from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON


class ImageAnalysis(SQLModel, table=True):
    """Stores analysis results for a product image."""
    id: Optional[int] = Field(default=None, primary_key=True)
    product_type: str
    product_category: str
    style_descriptors: list[str] = Field(sa_column=Column(JSON))
    material_details: list[str] = Field(sa_column=Column(JSON))
    distinctive_features: list[str] = Field(sa_column=Column(JSON))
    primary_colors: list[str] = Field(sa_column=Column(JSON))
    accent_colors: list[str] = Field(sa_column=Column(JSON))
    brand_elements: list[str] = Field(sa_column=Column(JSON))
    advertising_keywords: list[str] = Field(sa_column=Column(JSON))
    overall_aesthetic: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class MoodboardAnalysis(SQLModel, table=True):
    """Stores analysis results for a moodboard image."""
    id: Optional[int] = Field(default=None, primary_key=True)
    scene_description: str
    visual_style: str
    mood_atmosphere: str
    color_theme: list[str] = Field(sa_column=Column(JSON))
    composition_patterns: str
    suggested_keywords: list[str] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class UserVision(SQLModel, table=True):
    """Stores structured user vision input."""
    id: Optional[int] = Field(default=None, primary_key=True)
    subjects: str
    action: str
    setting: str
    lighting: str
    mood_descriptors: list[str] = Field(sa_column=Column(JSON))
    additional_details: list[str] = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Prompt(SQLModel, table=True):
    """Stores generated prompts for image generation."""
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_text: str
    image_analysis_id: Optional[int] = Field(default=None, foreign_key="imageanalysis.id")
    moodboard_analysis_id: Optional[int] = Field(default=None, foreign_key="moodboardanalysis.id")
    user_vision_id: Optional[int] = Field(default=None, foreign_key="uservision.id")
    focus_slider: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class GeneratedImage(SQLModel, table=True):
    """Stores final generated ad images and their metadata."""
    id: Optional[int] = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id")
    image_url: str  # This is the final generated ad image
    input_images: list[str] = Field(sa_column=Column(JSON))  # All input images used for generation
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))