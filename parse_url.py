
import pandas as pd
import requests
from bs4 import BeautifulSoup
import json 
import os
import google.generativeai as genai 
from config import GENERATION_MODEL, PROJECT_ID, REGION , GOOGLE_API_KEY
from urllib.parse import urlparse, urlunparse

def clean_url(url: str) -> str:
    """Removes a trailing period from a URL if it exists."""
    if isinstance(url, str) and url.endswith('.'):
        return url[:-1]
    return url

def normalize_url(url: str) -> str:
    """Normalizes a URL by removing query parameters, fragment and trailing characters."""
    if not isinstance(url, str):
        return url
    # First, clean the url from trailing dots as before
    cleaned_url = clean_url(url)
    # Then, parse it
    parsed = urlparse(cleaned_url)
    # Reconstruct the URL without query parameters and fragment
    normalized = urlunparse((parsed.scheme, parsed.netloc, parsed.path, '', '', ''))
    return normalized

def _fetch_and_parse_single_url(url):
    """Fetches the content of a URL and returns the title and text from the main body."""
    normalized_url = normalize_url(url)
    try:
        response = requests.get(normalized_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        title = soup.find('title').text.strip() if soup.find('title') else "No Title Found"

        main_content = soup.find('div', {'class': 'devsite-article-body'}) or \
                       soup.find('article') or \
                       soup.find('main') or \
                       soup.find('div', {'id': 'content'}) or \
                       soup.find('body')

        if main_content:
            # Using '\n' preserves line breaks, which is better for diffing.
            text = main_content.get_text(separator='\n', strip=True)
        else:
            # This case should ideally not be hit if 'body' is the ultimate fallback
            print(f"Main content selectors not found for {normalized_url}, returning full body text.")
            text = soup.get_text(separator='\n', strip=True)

        return title, text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {normalized_url}: {e}")
        return None, None
    except Exception as e:
        print(f"Error processing {normalized_url}: {e}")
        return None, None


def get_page_content(input_csv_path, output_csv_path):
    """
    Loads URLs from a CSV, fetches their content, processes, and saves the results.
    """
    # 1. Load the CSV file into a DataFrame
    try:
        df = pd.read_csv(input_csv_path)
        print(f"CSV file '{input_csv_path}' successfully loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: CSV file not found at '{input_csv_path}'")
        return None
    except Exception as e:
        print(f"An error occurred while loading CSV '{input_csv_path}': {e}")
        return None

    if 'page_url' not in df.columns:
        print(f"Error: 'page_url' column not found in '{input_csv_path}'")
        return None

    # 2. Apply the fetching function to each URL
    print(f"\nStarting initial content fetching for {len(df)} URLs...")
    results = df['page_url'].apply(_fetch_and_parse_single_url)
    df['title'] = [result[0] if result else None for result in results]
    df['text'] = [result[1] if result else None for result in results]

    print("Initial content fetching complete.")
    # Save intermediate results (optional, but good for long processes)
    # df.to_csv(output_csv_path, index=False)
    # print(f"DataFrame with initial content saved to '{output_csv_path}'")

    # 3. Identify rows with None values and attempt to re-fetch
    rows_with_missing_data = df[df['title'].isnull() | df['text'].isnull()]
    if not rows_with_missing_data.empty:
        print(f"\nAttempting to re-fetch content for {len(rows_with_missing_data)} URLs with missing data...")
        for index, row in rows_with_missing_data.iterrows():
            url = row['page_url']
            # Only re-fetch if title or text is still None
            if pd.isnull(df.loc[index, 'title']) or pd.isnull(df.loc[index, 'text']):
                print(f"Re-fetching content for URL: {url}")
                title, text = _fetch_and_parse_single_url(url)
                if title is not None:
                    df.loc[index, 'title'] = title
                if text is not None:
                    df.loc[index, 'text'] = text
        print("Re-fetching attempt complete.")
    else:
        print("\nNo URLs required re-fetching based on initial pass.")

    # 4. Report on missing data after re-fetch attempt
    final_missing_count = df['title'].isnull().sum() + df['text'].isnull().sum()
    print(f"\nNumber of entries still missing title or text after re-fetch: {df[df['title'].isnull() | df['text'].isnull()].shape[0]}")

    # 5. Drop rows where 'title' or 'text' is still NA
    # These are critical for subsequent processing.
    original_row_count = len(df)
    df_cleaned = df.dropna(subset=['title', 'text'])
    dropped_rows_count = original_row_count - len(df_cleaned)
    if dropped_rows_count > 0:
        print(f"Dropped {dropped_rows_count} rows due to missing 'title' or 'text' after all attempts.")
    else:
        print("No rows dropped due to missing 'title' or 'text'.")

    # 6. Save the final DataFrame
    try:
        df_cleaned.to_csv(output_csv_path, index=False)
        print(f"\nFinal DataFrame saved to '{output_csv_path}'. Shape: {df_cleaned.shape}")
    except Exception as e:
        print(f"Error saving final DataFrame to '{output_csv_path}': {e}")
        return df_cleaned # Return the dataframe even if saving fails

    return df_cleaned

if __name__ == "__main__":
    # Example usage:
    # Ensure these paths are correct for your environment
    # You might want to get these from config.py or command-line arguments
    INPUT_CSV = '/home/varunsankuri/devrel-kg/files/url_list_wo_ref.csv'
    OUTPUT_CSV = '/home/varunsankuri/devrel-kg/files/url_list_with_content.csv'
    
    print(f"--- Starting URL Content Parsing ---")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output CSV: {OUTPUT_CSV}")

    final_dataframe = get_page_content(INPUT_CSV, OUTPUT_CSV)

    if final_dataframe is not None:
        print(f"\n--- Processing Complete ---")
        print(f"Final DataFrame has {final_dataframe.shape[0]} rows and {final_dataframe.shape[1]} columns.")
        print("Sample of final data (first 5 rows):")
        print(final_dataframe.head())
    else:
        print("\n--- Processing Failed or Aborted ---")
        