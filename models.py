# stdlib imports
from datetime import datetime, timezone

# third-party imports
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON
from sqlmodel import SQLModel, Field


"""
NOTE TO MYSELF:
When using Field(sa_column=Column(JSON)) in SQLModel,
SQLModel automatically handles the conversion:
When saving to database: Python list[str] → JSON string
When reading from database: JSON string → Python list[str]
You don't need to manually call model_dump_json() or json.dumps()
- SQLModel does this automatically.
"""


class ImageAnalysis(SQLModel, table=True):
    """Stores analysis results for a product image."""
    id: int | None = Field(default=None, primary_key=True)
    product_type: str
    product_category: str
    style_descriptors: list[str] = Field(sa_column=Column(JSON))
    material_details: list[str] = Field(sa_column=Column(JSON))
    distinctive_features: list[str] = Field(sa_column=Column(JSON))
    primary_colors: list[str] = Field(sa_column=Column(JSON))
    accent_colors: list[str] = Field(sa_column=Column(JSON))
    brand_elements: list[str] = Field(sa_column=Column(JSON))
    advertising_keywords: list[str] = Field(sa_column=Column(JSON))
    overall_aesthetic: str | None = None
    image_path: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class MoodboardAnalysis(SQLModel, table=True):
    """Stores analysis results for a moodboard image."""
    id: int | None = Field(default=None, primary_key=True)
    scene_description: str
    visual_style: str
    mood_atmosphere: str
    color_theme: list[str] = Field(sa_column=Column(JSON))
    composition_patterns: str
    suggested_keywords: list[str] = Field(sa_column=Column(JSON))
    image_path: str | None = None
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserVision(SQLModel, table=True):
    """Stores structured user vision input."""
    id: int | None = Field(default=None, primary_key=True)
    subjects: str
    action: str
    setting: str
    lighting: str
    mood_descriptors: list[str] = Field(sa_column=Column(JSON))
    additional_details: list[str] = Field(sa_column=Column(JSON))
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Prompt(SQLModel, table=True):
    """Stores generated prompts for image generation."""
    id: int | None = Field(default=None, primary_key=True)
    prompt_text: str
    image_analysis_id: int | None = Field(default=None, foreign_key="imageanalysis.id")
    moodboard_analysis_ids: list[int] = Field(sa_column=Column(JSON))  # Changed to list of IDs
    user_vision_id: int | None = Field(default=None, foreign_key="uservision.id")
    focus_slider: int
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeneratedImage(SQLModel, table=True):
    """Stores final generated ad images and their metadata."""
    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id")
    image_url: str  # This is the final generated ad image
    input_images: list[str] = Field(sa_column=Column(JSON))  # All input images used for generation
    session_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))