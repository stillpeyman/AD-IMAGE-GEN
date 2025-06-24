from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json


# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# """PLAIN TEXT OUTPUT"""
# def build_prompt(keywords, optional_note, image_filename):
#     keywords_str = ", ".join(keywords)
#     prompt = f"""
# Assume the image is a close-up photo of white sneakers on concrete in natural sunlight.
# Keywords: {keywords_str}.
# Note: {optional_note}
# Filename: {image_filename}

# Please do the following:
# 1. Suggest 5 trending hashtags related to these keywords.
# 2. Describe the typical visual style seen in social media posts with those hashtags (colors, mood, composition).
# 3. Create a creative, detailed prompt for generating an ad image based on all the information above.
# """
#     return prompt


# def call_openai_gpt4o(prompt_text):
#     response = client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant that simulates social media trends and styles."},
#             {"role": "user", "content": prompt_text},
#         ],
#         temperature=0.7,
#         max_tokens=500,
#     )
    
#     # Save the full raw response to a JSON file
#     with open("openai_response.json", "w") as handle:
#         json.dump(response.model_dump(), handle, indent=2)

#     return response.choices[0].message.content


# keywords = ["urban", "youth", "minimal"]
# optional_note = "Close-up product photo of sneakers on concrete."
# image_filename = "sneakers.jpg"

# prompt_text = build_prompt(keywords, optional_note, image_filename)
# result = call_openai_gpt4o(prompt_text)

# print()
# print("=== GPT-4o-mini Response ===")
# print(result)


"""STRUCTURED OUTPUT"""

# Define structured output schema
class AdImagePrompt(BaseModel):
    hashtags: list[str]
    visual_style_description: str
    generation_prompt: str


def build_prompt(keywords: list[str], optional_note: str, image_filename: str) -> str:
    return f"""
    Assume the image is a close-up photo of white sneakers on concrete in natural sunlight.
    Keywords: {", ".join(keywords)}
    Note: {optional_note}
    Filename: {image_filename}

    Please do the following:
    1. Suggest 5 trending hashtags related to these keywords.
    2. Describe the typical visual style seen in social media posts with those hashtags (colors, mood, composition).
    3. Create a creative, detailed prompt for generating an ad image based on all the information above.
    """


keywords = ["urban", "youth", "minimal"]
optional_note = "Close-up product photo of sneakers on concrete."
image_filename = "sneakers.jpg"

prompt_text = build_prompt(keywords, optional_note, image_filename)


response = client.responses.parse(
    model="gpt-4o-mini",
    input=[
        {"role": "system", "content": "You are a helpful assistant that simulates social media trends and styles."},
        {"role": "user", "content": prompt_text},
    ],
    text_format=AdImagePrompt,
)

# Save the full raw response to a JSON file
with open("raw_response.json", "w") as handle:
    handle.write(response.model_dump_json(indent=2))

result = response.output_parsed


print()
print("=== GPT-4o-mini Response ===")
# print(result)
print(result.hashtags)
print(result.visual_style_description)
print(result.generation_prompt)