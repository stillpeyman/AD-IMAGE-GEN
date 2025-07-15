import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import json


# Load API key & initialize client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Path to product image
product_img_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "puma-sneakers-unsplash.jpg")
# Convert to absolute path
product_img_path = os.path.abspath(product_img_path)

# Path to moodboard folder
moodboard_paths = os.path.join(os.path.dirname(__file__), "..", "test_images", "moodboard")
# Convert to absolute path
moodboard_paths = os.path.abspath(moodboard_paths)
# Path to each moonboard image
moodboard_img_paths = [
    os.path.join(moodboard_paths, file) 
    for file in os.listdir(moodboard_paths) 
    if file.endswith((".jpg", ".jpeg", ".png"))
    ]


# Define structured output schema
class ImageAnalysis(BaseModel):
    main_subject: str
    dominant_colors: list[str]
    style_description: str
    composition_details: str
    mood_atmosphere: str
    suggested_keywords: list[str]

class MoodboardAnalysis(BaseModel):
    scene_description: str
    visual_style: str
    mood_atmosphere: str
    color_theme: list[str]
    composition_patterns: str
    suggested_keywords: list[str]

class KeywordCategorization(BaseModel):
    brand_terms: list[str]
    product_terms: list[str]
    lifestyle_terms: list[str]
    style_terms: list[str]
    uncategorized_terms: list[str]


# Function to encode the image
def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_structured(image_path: str) -> ImageAnalysis:
    """Anaylize image with structured output."""

    # Get Base64 string
    base64_image = encode_image(image_path)

    response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are an expert visual analyst for advertising and social media."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """Analyze this image for advertising purposes and provide:
                        1. Main subject/product in the image
                        2. 3-5 dominant colors (be specific, e.g., 'deep navy blue', 'warm cream')
                        3. Overall style description (e.g., minimalist, urban, vintage)
                        4. Composition details (e.g., centered, rule of thirds, close-up)
                        5. Mood/atmosphere (e.g., energetic, calm, luxurious)
                        6. 5-7 relevant keywords for social media marketing"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
            }
        ],
        text_format=ImageAnalysis,
    )
    
    # Save structured results to a JSON file
    with open("data/image_analysis_structured.json", "w", encoding="utf-8") as handle:
        handle.write(response.model_dump_json(indent=2))

    # Access parsed response
    return response.output_parsed


def analyze_moodboard_images(image_paths: list[str]) -> list[MoodboardAnalysis]:
    results = []
    for image_path in image_paths:
        base64_image = encode_image(image_path)

        response = client.responses.parse(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": "You are an expert visual analyst for advertising and social media."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": """Analyze this moodboard image and provide:
                        1. Brief scene description (what's happening in the image)
                        2. Visual style
                        3. Mood/atmosphere
                        4. Color theme (3-5 key tones)
                        5. Composition patterns
                        6. 5-7 relevant keywords that describe the visual aesthetics"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    }
                ]
            }
        ],
        text_format=MoodboardAnalysis,
    )
    
        results.append(response.output_parsed)

    return results


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
                Categorize these keywords from a product image analysis: {keywords}

Categories:
- brand_terms: Specific brand names, company names
- product_terms: Actual products, items, categories
- lifestyle_terms: Activities, behaviors, lifestyle concepts
- style_terms: Fashion, aesthetics, visual style descriptors
- uncategorized_terms: Anything that doesn't fit above

Explain your reasoning for each keyword placement, then provide the structured categorization.
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


image_analysis = analyze_image_structured(product_img_path)
with open("data/image_analysis.json", "w", encoding="utf-8") as handle:
    handle.write(image_analysis.model_dump_json(indent=2))

moodboard_analysis = analyze_moodboard_images(moodboard_img_paths)
with open("data/moodboard_analysis.json", "w", encoding="utf-8") as handle:
    json_data = [
        {
            "image_path": img_path,
            "analysis": result.model_dump()
        }
        # zip() pairs up elements from two lists position by position
        for img_path, result in zip(moodboard_img_paths, moodboard_analysis)
    ]
    json.dump(json_data, handle, indent=2)

all_keywords = image_analysis.suggested_keywords + [
    kw for result in moodboard_analysis for kw in result.suggested_keywords
]
kw_categorization = categorize_keywords(all_keywords)
with open("data/keywords_categorization.json", "w", encoding="utf-8") as handle:
    handle.write(kw_categorization.model_dump_json(indent=2))


