import requests
from dotenv import load_dotenv
import os
import json


# Load API key & search engine ID
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
cse_id = os.getenv("SEARCH_ENGINE_ID")
query = "site:pinterest.com athletic wear"

url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&sort=date"

response = requests.get(url)
data = response.json()

with open("data/cse_pinterest.json", "w", encoding="utf-8") as handle:
    json.dump(data, handle, ensure_ascii=False, indent=2)
