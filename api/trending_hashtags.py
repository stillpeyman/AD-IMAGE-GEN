import requests
from dotenv import load_dotenv
import json
import os
from openai import OpenAI


load_dotenv()
api_key = os.getenv("RAPID_API_KEY")


def fetch_trending_hashtags(period, countryCode):
    url = f"https://trending-hashtags.p.rapidapi.com/api/trends/{countryCode}/{period}"

    headers = {
        "x-rapidapi-key": api_key,
        "x-rapidapi-host": "trending-hashtags.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)

    return response.json()

hashtags = fetch_trending_hashtags("7", "WORLD")

with open("trending_hashtags.json", "w", encoding="utf-8") as handle:
    # ensure_ascii=False allows non-ASCII characters (e.g. emojis, ü, 漢字) to appear normally instead of being escaped
    json.dump(hashtags, handle, indent=2, ensure_ascii=False)

print("\n=== RAPID-API TRENDING-HASHTAGS WORLDWIDE ===")
print([item["hashtag"] for item in hashtags])

