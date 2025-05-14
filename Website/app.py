from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import google.generativeai as genai
import os
from dotenv import load_dotenv
import markdown
import json
from pydantic import BaseModel
import requests

# Load environment variables from .env file
load_dotenv()
genai.configure(api_key=os.getenv("API_KEY"))  # ✅ Correct way to configure GenAI
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

app = Flask(__name__)

# --- Load your data and model here ---
movies_df = pd.read_csv('./df_with_embeddings.csv', dtype={'imdbId': str})
try:
    movies_df['embeddings'] = movies_df['embeddings'].apply(lambda x: np.array(eval(x)))
    print("Embeddings successfully loaded and converted.")
except (SyntaxError, NameError, TypeError) as e:
    print(f"Error converting embeddings to numpy arrays. Check that 'embeddings' is a list of numbers: {e}")

if 'embeddings' not in movies_df.columns or not isinstance(movies_df['embeddings'].iloc[0], np.ndarray):
    print("Error: 'embeddings' column is missing or not in the expected format.")


def get_rag_recommendations(description, movies_df, top_n=40):
    if 'embeddings' not in movies_df.columns or not isinstance(movies_df['embeddings'].iloc[0], np.ndarray):
        raise ValueError("Embeddings column not properly loaded or formatted.")

    try:
        response = genai.embed_content(
            model="models/embedding-001",
            content=description,
            task_type="retrieval_query"
        )

        if not response or "embedding" not in response:
            raise ValueError("Embedding response from GenAI was empty or invalid.")

        query_embedding = response["embedding"]

    except Exception as e:
        print(f"Error getting embedding from GenAI: {e}")
        raise

    try:
        stacked_embeddings = np.stack(movies_df['embeddings'])
        dot_products = np.dot(stacked_embeddings, query_embedding)
        top_indices = np.argsort(dot_products)[::-1][:top_n]
    except Exception as e:
        print(f"Error calculating dot products or getting top indices: {e}")
        raise

    required_cols = ['imdbId', 'combined_text', 'rating']
    if not all(col in movies_df.columns for col in required_cols):
        print(f"Error: Missing required columns in DataFrame for results: {required_cols}")
        cols_to_return = [col for col in required_cols if col in movies_df.columns]
        if not cols_to_return:
            return []
        return movies_df.iloc[top_indices].sort_values(by='rating', ascending=False)[cols_to_return].to_dict(orient='records')

    return movies_df.iloc[top_indices].sort_values(by='rating', ascending=False)[required_cols].to_dict(orient='records')


def paraphrase_query(query):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        prompt = f"""
You are part of a RAG based movie recommendation system.
Your task is to paraphrase the following query to make it more suitable for the retrieval task.
The query is: {query}

The movies had all the information combined into a single string and embedded. The combined string looks like this:
Title: Iron Man. Genres: Action, Adventure, Sci-Fi. Tags: animation. actors/Actors: Robert Downey Jr., Gwyneth Paltrow. Plot: Plot of Iron Man
Missing parts are removed so a movie with no tags will not have "Tags:" in its combined string.
Genre List: 'Adventure', 'Animation', 'Children', 'Comedy', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'Mystery', 'Sci-Fi', 'War', 'Musical', 'Documentary', 'IMAX', 'Western', 'Film-Noir'
Tags can be anything and are not limited to the genre list so make up tags if you feel like it.
Include actors/Actors if the user mentions them in the query.

Based on the above information, paraphrase the query to match the combined string format and return it. Only return the paraphrased query without any additional text or explanation. You are part of the pipeline so DO NOT include a single word other than the paraphrased query.
"""
        response = model.generate_content(prompt)
        if not response or not response.text:
            raise ValueError("Paraphrasing response from GenAI was empty.")

        return response.text.strip()

    except Exception as e:
        print(f"Error paraphrasing query with GenAI: {e}")
        return query

import re

def get_movie_recommendation(query):
    recommendation_text = "Could not get recommendations at this time."

    try:
        paraphrased_query = paraphrase_query(query)
        print(f"Paraphrased Query: {paraphrased_query}")

        similar_movies_data = get_rag_recommendations(paraphrased_query, movies_df, 30)
        print(f"Retrieved {len(similar_movies_data)} similar movies.")
        print(f"Similar Movies Example: {similar_movies_data[0]}")

        prompt = f"""
You are a friendly and helpful movie recommendation assistant.
The user has provided the following query: "{query}"

A retrieval system has found the following movies that are potentially relevant:
{similar_movies_data}

Based on the user's query and the list of retrieved movies, recommend 10 relevant movies from the list.
Do not recommend movies not present in the retrieved list. Return a list of json objects containing the imdbId from the retrieved list, and a short reason for each recommendation.
Only return the list. Do not include any additional explanation or message.
"""

        model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")
        response = model.generate_content(prompt)

        print("GenAI raw response text:", response.text)  # Debug output

        # Extract JSON content from the response text
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("Response is not in valid JSON format.")

        json_text = json_match.group(0)
        recommendations = json.loads(json_text)

    except Exception as e:
        print(f"An error occurred during the recommendation process: {e}")
        recommendations = []

    return recommendations

def get_movie_details(imdb_id):
    url = f"http://www.omdbapi.com/?i=tt{imdb_id}&apikey={OMDB_API_KEY}"
    response = requests.get(url)
    if response.status_code == 200:
        print(f"OMDB API response received for IMDB ID: {imdb_id}")
        return response.json()
    return None


@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    recommendations = None

    if request.method == "POST":
        query = request.form.get("query")
        recommendation_results = get_movie_recommendation(query)

        enhanced_recommendations = []
        for rec in recommendation_results:
            imdb_id = rec['imdbId']
            movie_details = get_movie_details(imdb_id)
            if movie_details and movie_details.get("Response") == "True":
                enhanced_recommendations.append({
                    'details': movie_details,
                    'reason': rec['reason']
                })

        recommendations = enhanced_recommendations

    return render_template("index.html", query=query, recommendations=recommendations)


if __name__ == '__main__':
    if not os.getenv("API_KEY"):
        print("Warning: API_KEY not set. GenAI functions will likely fail.")

    app.run(debug=True)
