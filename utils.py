"""
Utility functions for the ad generation application.
"""
# stdlib imports
import base64
import os
import uuid

# local imports
from constants import DEFAULT_IMAGE_EXTENSION, UPLOADS_DIR

# Image utilities
def save_uploaded_image(
    image_bytes: bytes,
    base_dir: str = UPLOADS_DIR,
    sub_dir: str | None = None,
    filename_prefix: str = "img",
    extension: str = DEFAULT_IMAGE_EXTENSION
) -> str:
    """
    Persist an uploaded image to disk in a configurable way.
    
    Args:
        image_bytes: Raw image data.
        base_dir: Top-level storage directory (defaults to UPLOADS_DIR).
        sub_dir: Optional subfolder (e.g., "product", "moodboards").
        filename_prefix: Prefix before UUID in the filename.
        extension: File extension (default: "jpg").

    Returns:
        The relative path as a string.
    """
    try: 
        target_dir = os.path.join(base_dir, sub_dir) if sub_dir else base_dir
        os.makedirs(target_dir, exist_ok=True)

        filename = f"{filename_prefix}_{uuid.uuid4().hex}.{extension}"
        path = os.path.join(target_dir, filename)

        with open(path, "wb") as f:
            f.write(image_bytes)

        return path
        
    except OSError as e:
        # <from e> = Preserve the original traceback via exception chaining
        raise RuntimeError(f"Failed to save image to {target_dir}: {e}") from e


def decode_data_url(data_url: str) -> tuple[str, bytes]:
    """
    Parse data URLs in the format 'data:<mime-type>;base64,<payload>'.

    Returns:
        (mime_type, raw_bytes)
    """
    if not isinstance(data_url, str) or not data_url.startswith("data:image"):
        raise ValueError("Invalid data URL: must start with 'data:image'.")

    try:
        # Extract the base64 data from the data URL
        # Format: "data:<mime-type>;base64,<payload>"
        header, b64data = data_url.split(",", 1)
        mime_type = header.split(";", 1)[0].split(":", 1)[1]

        # Decode base64 to raw bytes
        raw_bytes = base64.b64decode(b64data)
        
        return mime_type, raw_bytes

    except Exception as e:
        raise ValueError(f"Failed to parse and decode data URL: {e}") from e

