# Movie Recommendation System

This project demonstrates a movie recommendation system utilizing a Retrieval Augmented Generation (RAG) approach based on semantic similarity. It leverages data from two different movie datasets, preprocesses and combines relevant information, and then uses Google's ⁠ text-embedding-004 ⁠ model to create embeddings for semantic search.

## Project Overview

The goal of this project is to build a recommendation system that can retrieve movies based on the semantic meaning of a user's query, rather than just keyword matching. This is achieved by:

1.  *Data Acquisition:* Downloading and loading two movie datasets: the MovieLens Small dataset and the CMU Movie Summary Corpus.
2.  *Data Preprocessing:* Cleaning and preparing the data from both datasets for merging, including extracting year, cleaning titles, and formatting textual features.
3.  *Data Linking:* Merging the two datasets based on cleaned movie titles and years to combine information like genres, ratings, tags (from MovieLens), and plot summaries and actors (from CMU).
4.  *Feature Engineering for RAG:* Combining relevant text features (title, genres, tags, actors, plot summary) into a single descriptive text string for each movie.
5.  *Embedding Generation:* Using the Google Gemini API's ⁠ text-embedding-004 ⁠ model to generate vector embeddings for the combined text features. These embeddings capture the semantic meaning of the movie descriptions.
6.  *Semantic Retrieval:* Implementing a mechanism to perform semantic search by embedding a user's query and finding movies with the most similar embeddings (using dot product similarity).

## Datasets Used

*   *MovieLens Small Dataset:* (Source: https://grouplens.org/datasets/movielens/) Provides movie metadata (title, genres) and user ratings and tags.
*   *CMU Movie Summary Corpus:* (Source: https://www.cs.cmu.edu/~ark/personas/) Provides movie metadata (including Wikipedia ID), plot summaries, and character/actor information.

## Requirements

*   Python 3.7+
*   Jupyter Notebook or JupyterLab
*   pandas
*   numpy
*   requests
*   ⁠ google-generativeai ⁠ library
*   An API key for the Google Gemini API. You can obtain one from the [Google Cloud Console](https://console.cloud.google.com/).
*   ⁠ python-dotenv ⁠ (for loading API key from a ⁠ .env ⁠ file)

You can install the required Python packages using pip:

⁠ bash
pip install pandas numpy requests google-generativeai python-dotenv
 ⁠

## Setup and Usage

1.  *Clone the repository:*
    ⁠ bash
    git clone <repository_url>
    cd <repository_name>
     ⁠
    (Replace ⁠ <repository_url> ⁠ and ⁠ <repository_name> ⁠ with your actual repository information)

2.  *Set up your Google Gemini API Key:*
    *   Create a file named ⁠ .env ⁠ in the root directory of the project.
    *   Add the following line to the ⁠ .env ⁠ file, replacing ⁠ YOUR_API_KEY ⁠ with your actual Google Gemini API key:
        ⁠ dotenv
        API_KEY=YOUR_API_KEY
         ⁠

3.  *Run the Jupyter Notebook:*
    *   Start a Jupyter Notebook or JupyterLab server in the project directory:
        ⁠ bash
        jupyter notebook
        # or
        jupyter lab
         ⁠
    *   Open the ⁠ main.ipynb ⁠ notebook.

4.  *Execute the notebook cells:*
    *   Run each cell in the notebook sequentially.
    *   The notebook will download the datasets, perform preprocessing, merge the data, generate embeddings using the Gemini API, save the resulting DataFrame with embeddings to ⁠ df_with_embeddings.csv ⁠, and demonstrate a basic semantic retrieval test.

## Notebook Breakdown

The ⁠ main.ipynb ⁠ notebook is structured as follows:

*   *Import Required Libraries and Dataset:* Imports necessary Python libraries.
*   *Downloading Datasets:* Downloads the MovieLens and CMU datasets from their respective URLs.
*   *Using MovieLens small dataset for efficient testing:* Loads the MovieLens movies, ratings, and tags data into pandas DataFrames.
*   *Importing CMU Movie Summary Dataset:* Loads the CMU movie metadata, character (actor) data, and plot summaries into pandas DataFrames.
*   *Linking MovieLens and CMU Movie Summary Corpus:*
    *   *Preprocessing MovieLens Data:* Cleans the MovieLens movie titles to extract the year and standardize the title format.
    *   *Preprocessing CMU Corpus data:* Extracts the year from the CMU release dates and cleans movie names.
    *   *Linking MovieLens rows to Wikipedia ID in CMU:* Merges the MovieLens data with the CMU metadata based on cleaned title and year to obtain Wikipedia IDs.
*   *Merging average ratings, tags, plot summaries, and actors:*
    *   *Calculating and Merging average Ratings from MovieLens:* Computes the average rating for each movie from the MovieLens ratings data and merges it.
    *   *Merging Tags from MovieLens:* Aggregates tags for each movie from the MovieLens tags data and merges them.
    *   *Merging Actors from CMU corpus:* Aggregates actor names for each movie from the CMU character data and merges them.
    *   *Merging Plot Summaries from CMU Corpus:* Merges the plot summaries from the CMU data.
    *   *Final Dataframe after Merge:* Displays the final merged DataFrame with all combined information.
*   *RAG - Retrieval Augmented Generation:*
    *   *Combine Text Features into a single column:* Creates a single text column ⁠ combined_text ⁠ by concatenating the title, genres, tags, actors, and plot for each movie.
    *   *Embed the Combined Text using Gemini API:* Uses the Google Gemini API to generate embeddings for the ⁠ combined_text ⁠ column in batches.
    *   *Write the embeddings to CSV for later use:* Saves the DataFrame, including the generated embeddings, to a CSV file.
    *   *Test Semantic Retrieval:* Demonstrates how to perform a semantic search by embedding a query and finding the most similar movie embeddings in the dataset.
