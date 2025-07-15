from openai import OpenAI
from dotenv import load_dotenv
import os
from ad_image_gen.api.gpt_analysis import analyze_image_structured
from rapid_api_trending_hashtags import fetch_trending_hashtags


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Run image analysis, get suggested keywords
image_path = "test_images/nike-unsplash.jpg"
result = analyze_image_structured(image_path)
suggested_keywords = result.suggested_keywords

# Fetch trending hashtags
hashtags_data = fetch_trending_hashtags("7", "WORLD")
trending_hashtags = [item["hashtag"] for item in hashtags_data]

keyword_embeddings = client.embeddings.create(
    input=suggested_keywords,
    model="text-embedding-ada-002"
).data

hashtag_embeddings = client.embeddings.create(
    input=trending_hashtags,
    model="text-embedding-ada-002"
).data

print()
print(keyword_embeddings)
print()
print(hashtag_embeddings)


