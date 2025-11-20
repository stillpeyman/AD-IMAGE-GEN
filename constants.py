"""
Shared constants for the ad image generation project.
Keeps directory names and configuration values centralized.
"""

# Storage directories
OUTPUT_IMAGES_DIR = "output_images"
UPLOADS_DIR = "uploads"

# Upload subdirectories
PRODUCT_UPLOAD_SUBDIR = "product"
MOODBOARD_UPLOAD_SUBDIR = "moodboards"
REFERENCE_UPLOAD_SUBDIR = "references"

# File defaults
DEFAULT_IMAGE_EXTENSION = "jpg"

# Mapping returned MIME types to file extensions
MIME_EXTENSION_MAP = {
    "image/png": "png",
    "image/jpeg": "jpg",
}



