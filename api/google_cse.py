import requests
from dotenv import load_dotenv
import os
import json
import time


# Load API key & search engine ID
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
cse_id = os.getenv("SEARCH_ENGINE_ID")


# Load product image analysis
with open("data/image_analysis.json", "r", encoding="utf-8") as handle:
    image_analysis = json.load(handle)

# Load moonboard analysis (list of dicts, key: image_path, analysis)
with open("data/moodboard_analysis.json", "r", encoding="utf-8") as handle:
    moodboard_analysis_list = json.load(handle)

# Extract keywords
product_keywords = image_analysis.get("suggested_keywords", [])

moodboard_keywords = []
for result in moodboard_analysis_list:
    keywords = result["analysis"].get("suggested_keywords", [])
    moodboard_keywords.extend(keywords)

# Combine and get rid off dublicates
all_keywords = list(set(product_keywords + moodboard_keywords))
# print(all_keywords)


target_sites = [
    "pinterest.com", "vogue.com", "highsnobiety.com", "hypebeast.com", "trendhunter.com"
    ]
# queries = [f"{kw} site:{site}" for kw in all_keywords for site in target_sites]
# sites_part = " OR ".join(f"site:{site}" for site in target_sites)
# print(sites_part)
# queries = [f"{kw} {sites_part}" for kw in all_keywords]
# print(queries)

results = []

for kw in all_keywords:
    for site in target_sites:
        query = f"{kw} site:{site}"
        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&sort=date"

        try:
            response = requests.get(url)
            data = response.json()

            if "items" in data:
                for item in data["items"]:
                    results.append(
                        {
                            "keyword": kw,
                            "site": site,
                            "title": item.get("title"),
                            "link": item.get("link"),
                            "snippet": item.get("snippet")
                        }
                    )

        except Exception as e:
            print(f"Error fetching results for query '{query}': {e}")
            continue

        time.sleep(1.5)


with open("data/cse_visual_reference_results.json", "w", encoding="utf-8") as handle:
    json.dump(results, handle, ensure_ascii=False, indent=2)
