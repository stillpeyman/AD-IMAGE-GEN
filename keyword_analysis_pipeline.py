"""
This is a sample code (for now!) to use the existing JSON files (image_analysis and moodboard_analysis) and in order to not make repeated API calls. To use this sample code and its result for the next steps:
- Build dynamic Google CSE query
- Generate Final Visual Prompt for Ad Image Gen giving GPT:
    - image_analysis
    - moodboard_analysis
    - kw_categoriztion
    - summaries from 3-5 Google CSE articles 
"""
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json


# Load API key & initialize client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


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


class KeywordCategorization(BaseModel):
    brand_terms: list[str]
    product_terms: list[str]
    lifestyle_terms: list[str]
    style_terms: list[str]
    uncategorized_terms: list[str]

def categorize_keywords(keywords: list[str]) -> KeywordCategorization:

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are an expert at categorizing marketing keywords for social media trend analysis."
            },
            {
                "role": "user",
                "content": f"""
                Categorize these keywords from a product image and moodboard analysis: {keywords}

Categories:
- brand_terms: Specific brand names, company names
- product_terms: Actual products, items, categories
- lifestyle_terms: Activities, behaviors, lifestyle concepts
- style_terms: Fashion, aesthetics, visual style descriptors
- uncategorized_terms: Anything that doesn't fit above
"""
            }
        ],
        text_format=KeywordCategorization
    )

    # Save structured results to a JSON file
    with open("data/keywords_categorization_structured.json", "w", encoding="utf-8") as handle:
        handle.write(response.model_dump_json(indent=2))

    # Access parsed response
    return response.output_parsed


kw_categorization = categorize_keywords(all_keywords)
with open("data/keywords_categorization.json", "w", encoding="utf-8") as handle:
    handle.write(kw_categorization.model_dump_json(indent=2))