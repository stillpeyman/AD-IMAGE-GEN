import json
import requests
from dotenv import load_dotenv
import os
from collections import Counter


# Load API key & initialize client
load_dotenv()
bearer_token = os.getenv("X_BEARER_TOKEN")


def load_trending_keywords(file_path):
    """
    Load and parse the trending keywords JSON file.
    
    Args:
        file_path(str): Path to the trending keyword JSON file.
    
    Returns:
        dict: Parsed JSON data containing categorized trending keywords.
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def load_existing_keyword_mapping(filepath: str) -> dict:
    """
    Load existing keyword-to-hashtag mappings if file exists.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Existing mappings, or empty dict if file doesn't exist.
    """
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def collect_trending_keywords(data):
    """
    Collect all stable and increasing keywords from the parsed data into a single list.

    Args:
        data (dict): Dictionary of categorized keywords loaded from JSON.

    Returns:
        list[str]: Flattened list of all stable and increasing keywords.
    """
    keywords_list = []

    for category_name, category_data in data.items():
        # .extend() takes and adds each item individually to the list 
        keywords_list.extend(category_data.get("increasing_keywords", []))
        keywords_list.extend(category_data.get("stable_keywords", []))
    
    return keywords_list


def fetch_top_hashtags_per_kw(keywords_list: list[str], existing_mapping: dict = None, skip_existing: bool = True) -> dict:
    """
    For each keyword, send a request to the Twitter API, extract hashtags from recent tweets, count their frequency, and return the top 5 hashtags per keyword.

    If a response is received, it is also saved locally as a JSON file for offline reuse.

    Args:
        keywords_list (list[str]): 
            List of keywords to query on Twitter. If a single string is passed, it will be auto-wrapped into a list.

        existing_mapping (dict, optional): 
            An optional dictionary mapping previously processed keywords to their hashtags. If provided, keywords already in the mapping are skipped. This prevents duplicate requests and allows merging new hashtags into existing entries. If not provided (or set to None), an empty mapping will be used by default.

        skip_existing (bool, optional): 
            If True (default), keywords already in the existing mapping will be skipped to avoid duplicate API requests. If False, all keywords will be re-fetched and any new hashtags will be merged into the existing list.

    Returns:
        dict: Updated mapping of each keyword to a list of hashtags (e.g., {"Nike": ["#nike", "#justdoit", ...]}).

    Raises:
        ValueError: If any item in the list is not a string.
    """
    # Auto-wrap string in a list
    # And prevent looping through a string when sedning API requests (limited)
    if isinstance(keywords_list, str):
        print("You passed a string instead of a list! Converting to list...")
        keywords_list = [keywords_list]

    # Extra check: list must contain strings
    if not all(isinstance(keyword, str) for keyword in keywords_list):
        raise ValueError("All items in keywords_list must be strings.")

    # If existing_mapping is None, replace it with empty dict
    existing_mapping = existing_mapping or {}
    keyword_to_hashtags = existing_mapping.copy()

    for keyword in keywords_list:
        if skip_existing and keyword in keyword_to_hashtags:
            print(f"Skipping '{keyword}' - already fetched.")
            continue

        url = "https://api.twitter.com/2/tweets/search/recent"

        # Safe query string for exact match (treat 2-word-keyword as 1)
        query = f'"{keyword}"' if " " in keyword else keyword

        params = {
            "query": query,
            "max_results": 10,
            "tweet.fields": "entities"
        }

        headers = {"Authorization": f"Bearer {bearer_token}"}

        response = requests.get(url, headers=headers, params=params)

        # Check for response status and potential error
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            keyword_to_hashtags[keyword] = []
            # Move to next keyword
            continue

        data = response.json()

        # Save X data to JSON
        # replace spaces with underscores if, e.g., keyword = "running shoes"
        safe_filename = keyword.replace(" ", "_").lower()
        save_path = os.path.join("data", f"{safe_filename}_x_response.json")
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)

        # Extract hashtags and count them
        hashtag_counts = Counter()

        for tweet in data.get("data", []):
            entities = tweet.get("entities", {})
            hashtags = entities.get("hashtags", [])
            
            for tag in hashtags:
                hashtag_text = tag.get("tag", "").lower()
                hashtag_counts[hashtag_text] += 1

        # Collect top hashtags (top 5 for now)
        top_hashtags = [f"#{tag}" for tag, count in hashtag_counts.most_common(5)]

        # Add new hashtags to existing ones (if any), avoiding dublicates
        if keyword in keyword_to_hashtags:
            # Wrapping hashtags list in set() removes dublicates
            existing_tags = set(keyword_to_hashtags[keyword])
            # .union() merges both sets and removes dublicates
            combined = list(existing_tags.union(top_hashtags))
            keyword_to_hashtags[keyword] = combined

        else:
            keyword_to_hashtags[keyword] = top_hashtags
    
    return keyword_to_hashtags               


def main():
    """
    Main execution pipeline:
    - Loads trending keywords from file
    - Collects stable and increasing keywords
    - Fetches or saves top hashtags for each keyword
    - Writes the keyword-to-hashtag mapping to JSON
    """
    trend_kw_file = os.path.join(os.path.dirname(__file__), "..", "data", "trending_keywords.json")

    # File where final (updated) mappings are stored
    mapping_file = os.path.join("data", "keyword_to_hashtags.json")

    # Load data
    data = load_trending_keywords(trend_kw_file)
    keywords = collect_trending_keywords(data)
    # print(keywords)

    # Use only one keyword for now to avoid hitting rate limits
    test_keywords = [keywords[1]]

    # Load existing mapping if present
    existing_mapping = load_existing_keyword_mapping(mapping_file)

    # Fetch and merge hashtags
    updated_mapping = fetch_top_hashtags_per_kw(test_keywords, existing_mapping=existing_mapping)

    # Save mapping result
    with open(mapping_file, "w", encoding="utf-8") as handle:
        json.dump(updated_mapping, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()






# # Load saved mock response instead of calling the Twitter API
# with open("data/example_twitter_response.json", "r", encoding="utf-8") as handle:
#     data = json.load(handle)

# # Extract hashtags
# hashtag_counts = Counter()

# for tweet in data.get("data", []):
#     entities = tweet.get("entities", {})
#     hashtags = entities.get("hashtags", [])
    
#     for tag in hashtags:
#         hashtag_text = tag.get("tag", "").lower()
#         hashtag_counts[hashtag_text] += 1

# print(hashtag_counts)

# # Print top hashtags
# print("\nTop hashtags from mock data:")
# for tag, count in hashtag_counts.most_common(5):
    print(f"#{tag} — {count} times")
