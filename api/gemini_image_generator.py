"""
Image generation module using Gemini's image generation API.

Handles the actual image generation using Google's Gemini 2.5 Flash Image model.
This module provides the same interface as the GPT image generator for consistency
in the multi-model architecture.
"""

# stdlib imports
import base64

# third-party imports  
from google import genai
from google.genai import types


async def generate_image_data_url(
    prompt: str,
    product_image_bytes: bytes,
    model: str,
    api_key: str,
    reference_images_bytes: list[bytes] | None = None,
) -> str:
    """
    Generate an advertising image via Gemini's image generation API.

    Makes one API call and returns the image as a base64 data URL suitable for
    saving locally. This function maintains the same interface as the GPT image
    generator for consistency in the multi-model architecture.

    Args:
        prompt: Final advertising prompt text.
        product_image_bytes: Raw product image to include.
        model: Image generation model name (e.g., "gemini-2.5-flash-image").
        api_key: Google API key for Gemini.
        reference_images_bytes: Optional reference images to guide generation.

    Returns:
        Base64 data URL (e.g., "data:image/png;base64,...").

    Raises:
        ValueError: If the response lacks expected image data.
    """
    # Compose the final prompt with instructions for using reference images
    # This helps the model understand how to use the provided images effectively
    final_prompt = (
        f"{prompt}\n\n"
        "Use the provided reference images to ensure accuracy:\n"
        "- Product image: Use this for exact product details, colors, and branding\n"
        "- Character/reference images: Use these for pose, style, and scene elements\n"
        "Generate the image based on this description and the provided reference images."
    )
    
    # Convert product image bytes to base64 string (not data URL)
    # Gemini expects base64 strings directly, not data URLs like OpenAI
    product_image_b64 = base64.b64encode(product_image_bytes).decode('utf-8')
    
    # Create contents list with text prompt first, then images
    # Gemini's API expects: contents=[text, image1, image2, ...]
    contents = [final_prompt, product_image_b64]
    
    # Add reference images if provided
    # Each reference image is converted to base64 and appended to contents
    if reference_images_bytes:
        for img_bytes in reference_images_bytes:
            reference_image_b64 = base64.b64encode(img_bytes).decode('utf-8')
            contents.append(reference_image_b64)
    
    # Initialize Gemini client with the provided API key
    client = genai.Client(api_key=api_key)
    
    # Make the API call to Gemini for image generation
    # Uses the same generate_content method as text generation, but with image model
    response = client.models.generate_content(
        model=model,
        contents=contents
    )
    
    # Extract image data from response
    # Gemini response structure: response.candidates[0].content.parts
    # Each part can contain text OR image data (inline_data)
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            # Found the generated image data in base64 format
            b64_image_data = part.inline_data.data
            # Return as data URL format (consistent with GPT generator)
            # This ensures both generators return the same format for services.py
            return f"data:image/png;base64,{b64_image_data}"
    
    # If we get here, no image was found in the response
    # This indicates an API error or unexpected response format
    raise ValueError("Image generation did not return expected image data.")