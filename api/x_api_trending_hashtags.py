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
        file_path(str): Path to the keyword categorization JSON file
    
    Returns:
        dict: Parsed JSON data with cetegorized keywords
    """
    with open(file_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def collect_trending_keywords(data):
    """
    Collect all stable and increasing keywords into one list.
    """
    keywords_list = []

    for category_name, category_data in data.items():
        # .extend() takes and adds each item individually to the list 
        keywords_list.extend(category_data.get("increasing_keywords", []))
        keywords_list.extend(category_data.get("stable_keywords", []))
    
    return keywords_list


def fetch_top_hashtags_per_kw(keywords_list):
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
    # File path to keywords_categorization.json
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
