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


def fetch_top_hashtags_per_kw(keywords_list: list[str]) -> dict:
    """
    For each keyword, send a request to the Twitter API, extract hashtags from recent tweets, count their frequency, and return the top 5 hashtags per keyword.

    If a response is received, it is also saved locally as a JSON file for offline reuse.

    Args:
        keywords_list (list[str]): List of keywords to query on Twitter.
            If a single string is passed, it will be auto-wrapped into a list.

    Returns:
        dict: Mapping of each keyword to a list of its top 5 hashtags (e.g., {"Nike": ["#nike", "#justdoit", ...]}).

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

    keyword_to_hashtags = {}

    for keyword in keywords_list:
        url = "https://api.twitter.com/2/tweets/search/recent"

        params = {
            "query": keyword,
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
    data = load_trending_keywords(trend_kw_file)
    keywords = collect_trending_keywords(data)

    # for now using 1 kw in order to preserve limited API requests on Xs Free tier
    test_keyword = [keywords[0]]

    kw_to_hashtags = fetch_top_hashtags_per_kw(test_keyword)

    # Save final keyword-hashtag mapping
    with open("data/keyword_to_hashtags.json", "w", encoding="utf-8") as handle:
        json.dump(kw_to_hashtags, handle, ensure_ascii=False, indent=2)


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
#     print(f"#{tag} — {count} times")
