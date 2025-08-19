from PIL import Image
import os
from transformers import BlipProcessor, BlipForConditionalGeneration


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = os.path.join(os.path.dirname(__file__), "..", "test_images", "nike-unsplash.jpg")

image = Image.open(os.path.abspath(image_path)).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)
print(caption)