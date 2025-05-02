from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import markdown

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# --- Load your data and model here ---
# try:
movies_df = pd.read_csv('./df_with_embeddings.csv') # e.g., 'movies_with_embeddings.csv'
# Convert string representation of vectors to numpy arrays
try:
    movies_df['embeddings'] = movies_df['embeddings'].apply(lambda x: np.array(eval(x))) #convert from string to numpy
    print("Embeddings successfully loaded and converted.")
except (SyntaxError, NameError, TypeError) as e:
    print(f"Error converting embeddings to numpy arrays. Check that 'embeddings' is a list of numbers: {e}")
    # Handle this error appropriately, e.g., exit or log

# Check if embeddings column exists and is valid
if 'embeddings' not in movies_df.columns or not isinstance(movies_df['embeddings'].iloc[0], np.ndarray):
     print("Error: 'embeddings' column is missing or not in the expected format.")
     # You might want to stop the app or disable functionality here
     # For now, let's just continue, but expect errors if get_rag_recommendations is called


# --- Your existing functions ---
def get_rag_recommendations(description, movies_df, top_n=40):
    """Returns RAG-based recommendations sorted by collaborative filtering scores."""
    # Ensure embeddings are loaded correctly before calling
    if 'embeddings' not in movies_df.columns or not isinstance(movies_df['embeddings'].iloc[0], np.ndarray):
        raise ValueError("Embeddings column not properly loaded or formatted.")

    # Generate embedding for the input description using Google GenAI
    try:
        client = genai.Client(api_key=os.getenv("API_KEY"))
        if not client or not client.models: # Basic check if client initialized
             raise ConnectionError("Could not initialize Google GenAI client.")

        response = client.models.embed_content(
        model="text-embedding-004",
        contents=description,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")) # task type is retrieval query as part of RETRIEVAL DOCUMENT/RETRIEVAL QUERY embedding task pair from gemini

        if not response or not response.embeddings:
            raise ValueError("Embedding response from GenAI was empty or invalid.")

        query_embedding = response.embeddings[0].values

    except Exception as e:
        print(f"Error getting embedding from GenAI: {e}")
        # Depending on severity, you might raise the exception or return an empty list/error indication
        raise # Re-raise the exception to be caught in the route handler

    # Calculate dot product similarity scores
    # Ensure the stack operation is safe (e.g., handle empty or invalid embeddings in df)
    try:
        stacked_embeddings = np.stack(movies_df['embeddings'])
        dot_products = np.dot(stacked_embeddings, query_embedding)
        top_indices = np.argsort(dot_products)[::-1][:top_n]
    except Exception as e:
         print(f"Error calculating dot products or getting top indices: {e}")
         raise # Re-raise

    # return top movies
    # Ensure the columns exist before trying to access them
    required_cols = ['combined_text', 'rating']
    if not all(col in movies_df.columns for col in required_cols):
         print(f"Error: Missing required columns in DataFrame for results: {required_cols}")
         # You might want to return an empty list or raise an error
         # For now, let's attempt to return what's available, but be careful
         cols_to_return = [col for col in required_cols if col in movies_df.columns]
         if not cols_to_return: # If neither is available
              return []
         return movies_df.iloc[top_indices].sort_values(by='rating', ascending=False)[cols_to_return].to_dict(orient='records')

    return movies_df.iloc[top_indices].sort_values(by='rating', ascending=False)[required_cols].to_dict(orient='records')


def paraphrase_query(query):
    """Paraphrase the input query using Google GenAI."""
    try:
        client = genai.Client(api_key=os.getenv("API_KEY"))
        if not client or not client.models:
             raise ConnectionError("Could not initialize Google GenAI client.")

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

Based on the above information, paraphrase the query to match the combined string format and return it. Only return the paraphrased query without any additional text or explanation. You are part of the pipeline so DO NOT include anything other than the paraphrased query.
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
        )
        if not response or not response.text:
             raise ValueError("Paraphrasing response from GenAI was empty.")

        return response.text.strip() # Use strip to remove potential leading/trailing whitespace

    except Exception as e:
        print(f"Error paraphrasing query with GenAI: {e}")
        # Decide how to handle API errors - maybe return original query or an error message
        return query # Fallback: return the original query


def get_movie_recommendation(query):
    """Gets movie recommendations based on the query using RAG and GenAI."""
    recommendation_text = "Could not get recommendations at this time." # Default error message

    try:
        # Step 1: Paraphrase the query
        paraphrased_query = paraphrase_query(query)
        print(f"Paraphrased Query: {paraphrased_query}") # Log the paraphrased query

        # Step 2: Retrieve similar movies based on the paraphrased query
        # This call might raise errors if embeddings are bad or API fails
        similar_movies_data = get_rag_recommendations(paraphrased_query, movies_df, 40)
        print(f"Retrieved {len(similar_movies_data)} similar movies.")

        # Format the retrieved movies for the final LLM prompt
        # Create a simple string representation of the retrieved movies
        retrieved_movies_str = "\n".join([f"- Combined Text: {m.get('combined_text', 'N/A')}, Rating: {m.get('rating', 'N/A')}" for m in similar_movies_data])
        if not retrieved_movies_str:
             retrieved_movies_str = "No movies were retrieved based on your query."


        # Step 3: Use GenAI to formulate the final recommendation based on retrieved movies
        client = genai.Client(api_key=os.getenv("API_KEY"))
        if not client or not client.models:
             raise ConnectionError("Could not initialize Google GenAI client.")

        prompt = f"""
You are a friendly and helpful movie recommendation assistant.
The user has provided the following query: "{query}"

A retrieval system has found the following movies that are potentially relevant:
{retrieved_movies_str}

Based on the user's query and the list of retrieved movies, recommend 5 specific movies from the list.
Present the recommendations in a clear, bulleted list format, including the movie title and maybe a very brief reason based on the retrieved info (like genre or rating), but keep it concise. Do not recommend movies not present in the retrieved list. If fewer than 5 relevant movies are found, recommend fewer. If no relevant movies are retrieved, state that you couldn't find relevant movies based on the query.

Example format:
Based on your interest, here are some recommendations:
* Movie Title 1 (Genre, Rating: X.X)
* Movie Title 2 (Genre, Rating: X.X)
...
"""
        response = client.models.generate_content(
            model="gemini-2.0-flash", # Or "gemini-1.5-flash" if available and preferred
            contents=prompt,
        )
        if not response or not response.text:
             raise ValueError("Recommendation response from GenAI was empty.")

        recommendation_text = response.text.strip()

    except Exception as e:
        print(f"An error occurred during the recommendation process: {e}")
        recommendation_text = f"Sorry, an error occurred while getting recommendations: {e}"

    return recommendation_text


# Define routes
@app.route('/', methods=['GET', 'POST'])
def index():
    user_query = ""
    recommendations = None # Initialize recommendations to None

    if request.method == 'POST':
        user_query = request.form.get('query') # Use .get() for safety

        if user_query:
            # Call your recommendation function
            recommendations = get_movie_recommendation(user_query)
        else:
            recommendations = "Please enter a query to get recommendations."


    # Render the template, passing the query and recommendations (even if None)
    return render_template('index.html', query=user_query, recommendations=markdown.markdown(recommendations) if recommendations else None)


if __name__ == '__main__':
    # Ensure you have an API_KEY set in your .env file or environment variables
    if not os.getenv("API_KEY"):
         print("Warning: API_KEY not set. GenAI functions will likely fail.")

    # Set debug=True only for development, change to False for production
    app.run(debug=True)