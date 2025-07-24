import base64
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional
import os
import json


# Load API key & initialize client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


# Path to product image
product_img_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_images", "puma-sneakers-unsplash.jpg")
# Convert to absolute path
product_img_path = os.path.abspath(product_img_path)

# Path to moodboard folder
moodboard_paths = os.path.join(os.path.dirname(__file__), "..", "data", "test_images", "moodboard")
# Convert to absolute path
moodboard_paths = os.path.abspath(moodboard_paths)
# Path to each moonboard image
moodboard_img_paths = [
    os.path.join(moodboard_paths, file) 
    for file in os.listdir(moodboard_paths) 
    if file.endswith((".jpg", ".jpeg", ".png"))
    ]


# User text examples for testing
user_text1 = "young teenage girl skating, location similar to Venice Beach skatepark, blue hour"

user_text2 = "professional woman in business attire, urban coffee shop, morning golden hour, confident and relaxed"

user_text3 = "group of friends laughing, cozy living room, warm evening light, casual weekend vibe"


# Define structured output schema
class ImageAnalysis(BaseModel):
    product_type: str
    product_category: str
    style_descriptors: list[str]
    material_details: list[str]
    distinctive_features: list[str]
    primary_colors: list[str]
    accent_colors: list[str]
    brand_elements: list[str]
    advertising_keywords: list[str]
    # Optional means, it can be a string or None
    overall_aesthetic: Optional[str] = None

class MoodboardAnalysis(BaseModel):
    scene_description: str
    visual_style: str
    mood_atmosphere: str
    color_theme: list[str]
    composition_patterns: str
    suggested_keywords: list[str]

class UserVision(BaseModel):
    subjects: str
    action: str
    setting: str
    lighting: str
    mood_descriptors: list[str]
    additional_details: list[str]


# Function to encode the image
def encode_image(image_path):
    """Encode the image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_product_image(image_path: str) -> ImageAnalysis:
    """Anaylize image with structured output."""

    # Get Base64 string
    base64_image = encode_image(image_path)

    response = client.responses.parse(
        model="gpt-4.1",
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
                        "text": """
                        Analyze this product image for advertising purposes and provide:
                        1. Product type (e.g., 'sneakers', 'dress shirt', 'backpack')
                        2. Product category (e.g., 'footwear', 'apparel', 'accessories', 'electronics')
                        3. Style descriptors as list (e.g., ['minimalist', 'low-top'], ['vintage', 'elegant'])
                        4. Material details as list (e.g., ['leather', 'mesh'], ['cotton', 'denim'])
                        5. Distinctive features as list (e.g., ['white sole', 'perforated toe'])
                        6. Primary colors as list (e.g., ['black', 'white'], ['navy blue', 'gray'])
                        7. Accent colors as list (e.g., ['red accents', 'silver details'])
                        8. Brand elements as list (e.g., ['Puma logo', 'embossed text'], ['Nike logo, 'swoosh'])
                        9. Advertising keywords as list (e.g., ['urban', 'athletic', 'versatile'])
                        10. Overall aesthetic (optional) (e.g., 'luxury minimalist', 'urban casual')
                        """
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
    
    # Save raw response to a JSON file
    with open("data/image_analysis_response.json", "w", encoding="utf-8") as handle:
        handle.write(response.model_dump_json(indent=2))

    # Access parsed response
    return response.output_parsed


def analyze_moodboard(image_paths: list[str]) -> list[MoodboardAnalysis]:
    results = []
    for image_path in image_paths:
        base64_image = encode_image(image_path)

        response = client.responses.parse(
        model="gpt-4.1",
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


def parse_user_vision(user_text: str) -> UserVision:
    response = client.responses.parse(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": "You are an expert at extracting structured information from user description for advertising content creation."
            },
            {
                "role": "user",
                "content": f""" Extract structured information from this user description: '{user_text}'

                Please identify and extract:
                1. Who: People/subjects described (age, gender, count, etc.)
                2. What: Activities, actions, or behaviors mentioned
                3. Where: Locations, settings, or environments described
                4. When: Time of day, season, or temporal context
                5. Mood descriptors: Any mood, style, or atmosphere words
                6. Additional details: Any other specific requests or requirements

                If any category is not mentioned or unclear, leave it empty or mark as 'not specified'.
                """
            }
        ],
        text_format=UserVision,
    )

    # Access parsed response
    return response.output_parsed


def build_advertising_prompt(image_analysis_path: str, moodboard_analysis_path: str, user_vision_path: str, focus_slider: int) -> str:
    """
    Build an optimized Stable Diffusion prompt by loading analysis data from JSON files,
    using a more granular focus instruction, and saving the generated prompt as JSON.
    """
    # Load product analysis
    with open(image_analysis_path, "r", encoding="utf-8") as handle:
        product_data = json.load(handle)

    # Load moodboard analysis
    with open(moodboard_analysis_path, "r", encoding="utf-8") as handle:
        moodboard_data = json.load(handle)

    # Load user vision
    with open(user_vision_path, "r", encoding="utf-8") as handle:
        vision_data = json.load(handle)

    # More granular focus instruction
    if focus_slider == 0:
        focus_instruction = "The product is the sole focus, with minimal background or scene elements."
    elif focus_slider == 1:
        focus_instruction = "The product is the main focus, with a subtle hint of the scene for context."
    elif focus_slider == 2:
        focus_instruction = "The product is prominent, but the scene provides gentle support."
    elif focus_slider == 3:
        focus_instruction = "The product is clearly the hero, but the scene is present and meaningful."
    elif focus_slider == 4:
        focus_instruction = "The product and scene are balanced, each drawing equal attention."
    elif focus_slider == 5:
        focus_instruction = "The scene and product are equally important, blending together."
    elif focus_slider == 6:
        focus_instruction = "The scene is slightly more prominent, but the product remains clearly visible."
    elif focus_slider == 7:
        focus_instruction = "The scene is dominant, with the product naturally integrated and visible."
    elif focus_slider == 8:
        focus_instruction = "The scene is the main focus, with the product subtly present."
    elif focus_slider == 9:
        focus_instruction = "The scene is highly dominant, with the product as a supporting element."
    else:  
        focus_instruction = "The atmosphere and setting are the sole focus, with the product barely visible but still present."

    # Compose the prompt for GPT
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "system",
                "content": "You are an expert at creating optimized Stable Diffusion prompts for advertising imagery. You understand how to balance product visibility with atmospheric storytelling."
            },
            {
                "role": "user",
                "content": f"""Create an optimized Stable Diffusion prompt for advertising imagery using this data:

                PRODUCT ANALYSIS: {json.dumps(product_data, indent=2)}
                MOODBOARD INSPIRATION: {json.dumps(moodboard_data, indent=2)}
                USER VISION: {json.dumps(vision_data, indent=2)}
                FOCUS INSTRUCTION: {focus_instruction}

                Requirements:
                - Create a single, cohesive prompt (not multiple options)
                - Keep it 30-75 words for optimal Stable Diffusion performance
                - Include specific details that help the AI understand the product and scene
                - Ensure the actual product will be recognizable in the generated image
                - Use photography/cinematography terms when appropriate
                - The product MUST be visible and identifiable in the final image
                
                Return only the prompt text, no explanation."""
            }
        ]
    )

    prompt_text = response.output_text

    # Save the prompt and input data as JSON
    output_json = {
        "prompt": prompt_text,
        "product_data": product_data,
        "moodboard_data": moodboard_data,
        "vision_data": vision_data,
        "focus_slider": focus_slider
    }
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "generated_prompt.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2)

    return prompt_text


def main():
    # Check if cached analysis exists, otherwise analyze
    image_analysis_file = "data/image_analysis.json"
    if os.path.exists(image_analysis_file):
        print("Loading cached image analysis...")
        with open(image_analysis_file, "r", encoding="utf-8") as handle:
            image_analysis = ImageAnalysis.model_validate_json(handle.read())
    else:
        print("Analyzing product image...")
        image_analysis = analyze_product_image(product_img_path)
        with open(image_analysis_file, "w", encoding="utf-8") as handle:
            handle.write(image_analysis.model_dump_json(indent=2))

    # Same for moodboard
    moodboard_analysis_file = "data/moodboard_analysis.json"
    if os.path.exists(moodboard_analysis_file):
        print("Loading cached moodboard analysis...")
        with open(moodboard_analysis_file, "r", encoding="utf-8") as handle:
            # Load the list of dicts
            json_data = json.load(handle)
            # Convert each "analysis" part to MoodboardAnalysis
            moodboard_analysis = [
                MoodboardAnalysis.model_validate(item["analysis"])
                for item in json_data
                ]
    else:
        print("Analyzing moodboard image...")
        moodboard_analysis = analyze_moodboard(moodboard_img_paths)
        with open(moodboard_analysis_file, "w", encoding="utf-8") as handle:
            json_data = [
                {
                    "image_path": img_path,
                    "analysis": result.model_dump()
                }
                # zip() pairs up elements from two lists position by position
                for img_path, result in zip(moodboard_img_paths, moodboard_analysis)
            ]
            json.dump(json_data, handle, indent=2)
    
    # Same for user vision
    user_vision_file = "data/user_vision.json"
    if os.path.exists(user_vision_file):
        print("Loading cached user vision...")
        with open(user_vision_file, "r", encoding="utf-8") as handle:
            user_vision = UserVision.model_validate_json(handle.read())
    else:
        print("Parsing user vision...")
        user_vision = parse_user_vision(user_text1)
        with open(user_vision_file, "w", encoding="utf-8") as handle:
            handle.write(user_vision.model_dump_json(indent=2))

    # Call the new build_advertising_prompt function
    focus_slider = 5  # Example value, can be changed or made user-configurable
    
    prompt_text = build_advertising_prompt(
        image_analysis_file,
        moodboard_analysis_file,
        user_vision_file,
        focus_slider
    )

    print("Generated prompt:")
    print(prompt_text)


if __name__ == "__main__":
    main()


