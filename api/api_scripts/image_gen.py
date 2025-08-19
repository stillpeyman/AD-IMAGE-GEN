from openai import OpenAI
import base64
import os
from dotenv import load_dotenv


# Load API key & initialize client
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


product_img_path = os.path.join(os.path.dirname(__file__), "..", "data", "test_images", "puma-sneakers-unsplash.jpg")

product_img_path2 = os.path.join(os.path.dirname(__file__), "..", "data", "test_images", "rayssa_leal.png")

def encode_image(file_path):
    """Encode the image to base64."""
    with open(file_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


prompt = """A young teenage girl like in the provided image of a skatergirl skates through a graffiti-covered urban skatepark at blue hour, Venice Beach vibes. Cinematic angle highlights her minimalist white low-top Puma sneakers provided in the product image. Synthetic leather, bold black side branding with Puma logo, white perforated sole, tonal stitching, embossed PUMA text on the sole. The energetic scene blends streetwear, freedom, and city grit, with vibrant graffiti and concrete textures emphasizing urban minimalist style. Ensure the stance is correct for skateboarding and the sneakers with the Puma logo is in view and the sneakers are true to the Puma sneakers in the provided image. Generate the image now based on the description and references."""

base64_image1 = encode_image(product_img_path)
base64_image2 = encode_image(product_img_path2)

response = client.responses.create(
    model="gpt-4.1",
    input=[
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image1}",
                },
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{base64_image2}",
                }
            ],
        }
    ],
    tools=[{"type": "image_generation", "input_fidelity": "high"}],
)

image_generation_calls = [
    output
    for output in response.output
    if output.type == "image_generation_call"
]

# Save raw response to a JSON file
with open("data/image_generation_response.json", "w", encoding="utf-8") as handle:
    handle.write(response.model_dump_json(indent=2))

image_data = [output.result for output in image_generation_calls]

os.makedirs("output_images", exist_ok=True)

if image_data:
    image_base64 = image_data[0]
    with open("data/output_images/puma_skate_ad.png", "wb") as f:
        f.write(base64.b64decode(image_base64))
else:
    # Print all outputs for debugging
    for output in response.output:
        if hasattr(output, "content"):
            print(output.content)
        else:
            print(output)