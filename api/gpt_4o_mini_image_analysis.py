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


# Path to test image
image_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "nike-unsplash.jpg")
# Convert to absolute path
image_path = os.path.abspath(image_path)


# Define structured output schema
class ImageAnalysis(BaseModel):
    main_subject: str
    dominant_colors: list[str]
    style_description: str
    composition_details: str
    mood_atmosphere: str
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
    with open("data/image_analysis_structured.json", "w") as handle:
        handle.write(response.model_dump_json(indent=2))

    # Access parsed response
    return response.output_parsed


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
    with open("data/keywords_categorization_structured.json", "w") as handle:
        handle.write(response.model_dump_json(indent=2))

    # Access parsed response
    return response.output_parsed


analysis = analyze_image_structured(image_path)
with open("data/image_analysis.json", "w") as handle:
    handle.write(analysis.model_dump_json(indent=2))
# print(analysis)


print()

kw_categorization = categorize_keywords(analysis.suggested_keywords)
with open("data/keywords_categorization.json", "w") as handle:
    handle.write(kw_categorization.model_dump_json(indent=2))
# print(kw_categorization)


# TEST
# if __name__ == "__main__":
#     try:
#         # Path to test image
#         image_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "nike-unsplash.jpg")
#         # Convert to absolute path
#         image_path = os.path.abspath(image_path)

#         print("Analyzing image with structured output...")

#         try:
#             analysis = analyze_image_structured(image_path)

#             print("\n=== STRUCTURED ANALYSIS RESULTS ===")
#             print(f"Main subject: {analysis.main_subject}")
#             print(f"Colors: {', '.join(analysis.dominant_colors)}")
#             print(f"Style: {analysis.style_description}")
#             print(f"Composition: {analysis.composition_details}")
#             print(f"Mood: {analysis.mood_atmosphere}")
#             print(f"Keywords: {', '.join(analysis.suggested_keywords)}")

#             # Save structured results to a JSON file
#             with open("data/image_analysis_structured.json", "w") as handle:
#                 handle.write(analysis.model_dump_json(indent=2))
            
#             print("\nStructured results saved to image_analysis_structured.json")
        
#         except Exception as e:
#             print(f"Structured output failed: {e}")
    
#     except FileNotFoundError:
#         print(f"Error: Image file not found at {image_path}")
    
#     except Exception as e:
#         print(f"Error analyzing image: {e}")

