# stdlib imports
from datetime import datetime, timezone

# third-party imports
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON
from sqlmodel import SQLModel, Field, Index


"""
NOTE TO MYSELF:
When using Field(sa_column=Column(JSON)) in SQLModel,
SQLModel automatically handles the conversion:
When saving to database: Python list[str] → JSON string
When reading from database: JSON string → Python list[str]
You don't need to manually call model_dump_json() or json.dumps()
- SQLModel does this automatically.

MULTI-MODEL ARCHITECTURE:
- Session table stores which AI model provider is used for a session
- Each analysis/generation record stores which model created it
- This enables model comparison and session consistency
- Prevents mixing different AI providers within the same workflow
"""


class UserSession(SQLModel, table=True):
    """
    Stores user session information for multi-model architecture.
    
    Each user session is bound to a specific AI model provider (OpenAI or Google)
    to ensure consistency throughout the workflow. This prevents mixing
    different AI providers within the same ad generation session.
    """
    id: str = Field(primary_key=True)  
    model_provider: str  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


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
    # REQUIRED foreign key to UserSession table
    session_id: str = Field(foreign_key="usersession.id") 
    model_provider: str  
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
    session_id: str = Field(foreign_key="usersession.id") 
    model_provider: str  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class UserVision(SQLModel, table=True):
    """Stores structured user vision input."""
    id: int | None = Field(default=None, primary_key=True)
    original_text: str  # Original user vision text input (for display after refresh)
    focus_subject: str
    action: str
    setting: str
    lighting: str
    mood_descriptors: list[str] = Field(sa_column=Column(JSON))
    additional_details: list[str] = Field(sa_column=Column(JSON))
    session_id: str = Field(foreign_key="usersession.id")
    model_provider: str  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Prompt(SQLModel, table=True):
    """Stores generated prompts for image generation."""
    id: int | None = Field(default=None, primary_key=True)
    prompt_text: str
    image_analysis_id: int | None = Field(default=None, foreign_key="imageanalysis.id")
    # No foreign key. Databases don't support FKs to lists 
    # We query each ID individually when needed
    moodboard_analysis_ids: list[int] = Field(sa_column=Column(JSON))
    user_vision_id: int | None = Field(default=None, foreign_key="uservision.id")
    focus_slider: int
    refinement_count: int = 0
    user_feedback: str | None = None 
    previous_prompt_id: int | None = Field(default=None, foreign_key="prompt.id")
    session_id: str = Field(foreign_key="usersession.id")  
    model_provider: str  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class GeneratedImage(SQLModel, table=True):
    """Stores final generated ad images and their metadata."""
    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int = Field(foreign_key="prompt.id")
    image_url: str  # This is the final generated ad image
    input_images: list[str] = Field(sa_column=Column(JSON))  # All input images used for generation
    session_id: str = Field(foreign_key="usersession.id") 
    model_provider: str  
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PromptExample(SQLModel, table=True):
    """Stores ad prompt examples by category for RAG"""
    id: int | None = Field(default=None, primary_key=True)
    prompt_id: int | None = Field(default=None, foreign_key="prompt.id")
    prompt_text: str
    # Matches ImageAnalysis.product_category for retrieval
    product_category: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HistoryEvent(SQLModel, table=True):
    """
    Stores history events for session workflow timeline.
    
    Each row represents one user-visible step in the ad generation workflow,
    enabling chat-style history display per session.
    """
    id: int | None = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="usersession.id")
    event_type: str
    related_type: str | None = None
    related_id: int | None = None
    actor: str
    snapshot_data: dict = Field(sa_column=Column(JSON))
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    __table_args__ = (Index("idx_session_created", "session_id", "created_at"),)
