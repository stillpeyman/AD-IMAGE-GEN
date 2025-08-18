"""
Image generation module using OpenAI's Responses API.
Handles the actual image generation with the image generation tool.
"""

# stdlib imports
import base64
import os

# third-party imports  
from openai import AsyncOpenAI


async def generate_image_data_url(
    prompt: str,
    product_image_bytes: bytes,
    reference_images_bytes: list[bytes] | None,
    model: str,
    api_key: str,
) -> str:
    """
    Generate an advertising image using OpenAI's Responses API (image_generation tool).
    
    This function makes a single API call to OpenAI and returns the generated image
    as a base64 data URL, ensuring a stable, non-expiring image that can be saved locally.
    
    Args:
        prompt (str): The advertising prompt text describing the desired image.
        product_image_bytes (bytes): Raw bytes of the product image to include.
        reference_images_bytes (list[bytes] | None): Optional list of reference image bytes.
        model (str): The OpenAI model name for image generation (e.g., "gpt-image-1").
        api_key (str): The OpenAI API key for image generation (MY_OPENAI_API_KEY).
        
    Returns:
        str: A base64 encoded data URL (e.g., "data:image/png;base64,...")
             representing the generated image.
             
    Raises:
        ValueError: If the API response doesn't contain expected image data.
    """
    # Compose the final prompt with instructions for using reference images
    final_prompt = (
        f"{prompt}\n\n"
        "Use the provided reference images to ensure accuracy:\n"
        "- Product image: Use this for exact product details, colors, and branding\n"
        "- Character/reference images: Use these for pose, style, and scene elements\n"
        "Generate the image based on this description and the provided reference images."
    )
    
    # Prepare content for the OpenAI Responses API
    # The API expects images as data URLs within the content array
    content = [{
        "role": "user",
        "content": [
            {"type": "input_text", "text": final_prompt},
            {
                "type": "input_image", 
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(product_image_bytes).decode('utf-8')}"
            },
        ],
    }]
    
    # Add reference images if provided
    if reference_images_bytes:
        for img_bytes in reference_images_bytes:
            content[0]["content"].append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{base64.b64encode(img_bytes).decode('utf-8')}",
            })
    
    # Initialize OpenAI client with the image generation API key
    client = AsyncOpenAI(api_key=api_key)
    
    # Make the single API call to OpenAI's Responses API for image generation
    resp = await client.responses.create(
        model=model,
        input=content,
        tools=[{"type": "image_generation", "input_fidelity": "high"}],
    )
    
    # Extract the base64 image data from the response
    b64_image = None
    for out in getattr(resp, "output", []):
        if getattr(out, "type", None) == "image_generation_call":
            b64_image = out.result
            break
    
    if b64_image:
        return f"data:image/png;base64,{b64_image}"
    else:
        # This shouldn't happen with a successful API call
        raise ValueError("Image generation did not return expected base64 data.")