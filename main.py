"""
FastAPI app wiring:
- Loads env keys for two OpenAI clients
- Sets up SQLite engine and table creation on startup
- Provides per-HTTP-request DB Session via Depends
- Provides AdGeneratorService via Depends for endpoints
"""

"""
ARCHITECTURE NOTE TO MYSELF:

Dual Provider, Multi-Model System:
- MY_OPENAI_API_KEY for OpenAI operations (gpt-4.1 for text, gpt-image-1 for images)
- GEMINI_API_KEY for Google operations (gemini-2.5-flash for text gemini-2.5-flash-image for images)
- User chooses model provider at session start (no defaults)
  
Text Analysis (via Agents class):
- OpenAI: gpt-4.1 (fallback maybe to gpt-4.1-mini)
- Google: gemini-2.5-flash (alternative provider)
- Jobs: Product analysis, moodboard analysis, user vision parsing, prompt building
- Validation: API keys checked at Agents.__init__() based on provider

Image Generation (via image_generator modules):  
- Model: gpt-image-1 or gemini-2.5-flash-image
- Jobs: Generate final ad images
- Validation: API key validated at AdGeneratorService.__init__()

Why this design:
- User choice between providers for text analysis
- Single image generation provider (OpenAI)
- Fail-fast validation prevents runtime errors
"""

"""
How DB connections and user sessions work:
    - Each HTTP request is isolated and receives a fresh SQLAlchemy Session from get_db_session; FastAPI auto-closes it after the response.
    - /session/create persists a UserSession and returns user_session_id to the frontend.
    - Subsequent workflow endpoints include user_session_id. FastAPI injects get_service, which itself depends on get_db_session.
    - get_service uses the per-request db_session to load the UserSession by user_session_id and constructs AdGeneratorService (and Agents) with the correct model_provider.
    - Net effect: new DB connection per request, but every request is bound to the same logical user session created at the start of the workflow.
"""

"""
NOTE TO MYSELF for better organizing code in separate files:
see https://fastapi.tiangolo.com/tutorial/bigger-applications/#import-apirouter
>>> "Multiple Files/APIRouter"
"""
# stdlib
import logging
import os
import uuid

# third-party
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import SQLModel, Session, create_engine, select
from sqlalchemy import event
import uvicorn

# local
from agents import Agents
from models import ImageAnalysis, MoodboardAnalysis, PromptExample, UserVision, Prompt, GeneratedImage, UserSession, HistoryEvent
from services import AdGeneratorService


# Configure application logging
# This enables logger.info() statements throughout services.py
# Format: timestamp - module - level - message
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Test session for MVP
# TEST_SESSION_ID = "test-session-123"


# 1) env
load_dotenv()
OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# Enable debug endpoint only when ENABLE_DB_PING=true in .env
ENABLE_DB_PING = os.getenv("ENABLE_DB_PING", "false").lower() == "true"


# 2) Database engine configuration
DATABASE_FILE = "database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"

# Create SQLAlchemy engine with SQLite-specific configurations
# - echo=False: Not logging all SQL statements (logging takes care of it for debugging)
# - check_same_thread=False: Allow connections to be used across async thread switches
#   (Required for FastAPI's async nature; see Python sqlite3 docs)
# - timeout=30: Wait up to 30 seconds for database lock before raising error
#   (Default is 5 seconds; increased to handle concurrent requests better)
engine = create_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False, "timeout": 30}
)

# Understand the flow and when @event.listens_for triggers:
# 1. App starts
# 2. create_engine() is called
# 3. Engine created, but no connections yet
# (now every time a NEW connection opens, 4.-11. runs)
# 4. First HTTP request comes in
# 5. get_db_session() is called
# 6. Session tries to get a connection from the engine
# 7. Engine opens a NEW database connection
# 8. @event.listens_for triggers! ← HERE
# 9. _sqlite_set_pragmas() is called automatically
# 10. WAL mode is set
# 11. Connection is now ready to use
@event.listens_for(engine, "connect")
def _sqlite_set_pragmas(dbapi_connection, connection_record):
    """
    Set pragmas on connection open:
    - Enable WAL (write-ahead log) for better concurrency.
    Note: WAL is persistent on the DB file; this sets it on each connection defensively.
    """
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.close()
    except Exception:
        # best-effort; we don't want to crash app creation if the pragma fails
        pass


def create_db_and_tables() -> None:
    """
    Create all tables defined on SQLModel metadata if they don't exist yet.
    Idempotent: calling multiple times will not overwrite existing tables.
    """
    SQLModel.metadata.create_all(engine)


# 3) App creation
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Serve generated images from local disk under /static
app.mount("/static", StaticFiles(directory="output_images"), name="static")


# 4) Dependency and Service functions
def get_db_session():
    """
    Provide a database connection for the current request.

    This is a SQLAlchemy connection (not a logical user session). It enables
    queries/inserts/updates and is auto-closed by FastAPI after the response.
    """
    with Session(engine) as db_session:
        # yield -> FastAPI will close the session after the response
        yield db_session


def get_service(user_session_id: str, db_session: Session = Depends(get_db_session)) -> AdGeneratorService:
    """
    Create AdGeneratorService with the model provider from the user session.

    Depends(get_db_session) means: "Before calling this, first call get_db_session() and give me its returned database connection."
    
    Args:
        user_session_id: ID returned by /session/create, used to load the session's model provider
        db_session: Database connection for looking up the UserSession
        
    Returns:
        AdGeneratorService: Configured with user session's model provider
    """
    # Get session record from database
    session_record = db_session.get(UserSession, user_session_id)
    if not session_record:
        raise HTTPException(status_code=404, detail="Session not found. Please create a session first.")
    
    # Create agents with session's model provider
    agents = Agents(
        openai_api_key=OPENAI_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        model_provider=session_record.model_provider
    )
    
    return AdGeneratorService(
        agents=agents,                 
        session=db_session,
        openai_api_key=OPENAI_API_KEY,
        gemini_api_key=GEMINI_API_KEY
    )


@app.on_event("startup")
def on_startup() -> None:
    """
    FastAPI startup event handler.
    
    Called once when the application starts, before accepting any requests.
    Creates all database tables if they don't exist yet (idempotent operation).
    """
    create_db_and_tables()


if ENABLE_DB_PING:
    @app.get("/db/ping")
    def db_ping(db_session: Session = Depends(get_db_session)):
        """
        Lightweight health check for the database connection.

        Executes a trivial SELECT to validate that the per-request db_session
        is open and functional.

        Returns:
            {"status": "ok"} on success.
        """
        # Simple health check - just test database connection
        db_session.exec(select(1)).first()
        return {"status": "ok"}


@app.post("/session/create")
async def create_session(model_provider: str, db_session: Session = Depends(get_db_session)):
    """
    Create and persist a UserSession bound to the chosen model provider.

    Establishes the logical workflow session (separate from the DB connection),
    records the selected provider, and returns a user_session_id that the
    frontend must include with subsequent requests.

    Args:
        model_provider: Provider to use for this user session ("openai" or "gemini").
        db_session: Per-request database connection used to write the session.

    Returns:
        {"user_session_id": str, "model_provider": str}
    """
    if model_provider not in ["openai", "gemini"]:
        raise HTTPException(status_code=400, detail="model_provider must be 'openai' or 'gemini'")
    
    try:
        user_session_id = str(uuid.uuid4())
        session_record = UserSession(id=user_session_id, model_provider=model_provider)
        db_session.add(session_record)
        db_session.commit()
        db_session.refresh(session_record)

        history_event = HistoryEvent(
            session_id=user_session_id,
            event_type="session_created",
            related_type="UserSession",
            related_id=user_session_id,
            actor="user",
            snapshot_data={"model_provider": model_provider}
        )

        db_session.add(history_event)
        db_session.commit()
        
        return {"user_session_id": user_session_id, "model_provider": model_provider}
        
    except Exception as e:
        db_session.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@app.post("/analyze/product-image", response_model=ImageAnalysis)
async def analyze_product_image(
    # Required parameters first
    file: UploadFile,
    user_session_id: str,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    """
    Analyze a single product image and persist structured analysis.

    Args:
        file: Product image file upload.
        user_session_id: ID returned by /session/create.

    Returns:
        Persisted ImageAnalysis.
    """
    try:
        # read() = async method from FastAPI's UploadFile class
        # Reads uploaded file's binary content into bytes
        # Always needs <await> because file I/O is async
        image_bytes = await file.read()
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed {str(e)}.")

    try:
        # Service layer might raise generic ValueError
        result = await service.analyze_product_image(image_bytes, user_session_id)
        return result

    # Endpoint layer catches service's ValueError and converts to HTTP responce
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/moodboard", response_model=list[MoodboardAnalysis])
async def analyze_moodboard_images(
    # Required parameters first
    user_session_id: str,
    # Optional parameters
    files: list[UploadFile] | None = File(default=None),
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    """
    Analyze one or more moodboard images and persist results.

    Args:
        user_session_id: ID returned by /session/create.
        files: Optional list of moodboard images; empty/None returns [].

    Returns:
        List of persisted MoodboardAnalysis.
    """
    try:
        image_bytes_list = None
        if files:
            image_bytes_list = []
            for file in files:
                image_bytes = await file.read()
                image_bytes_list.append(image_bytes)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed {str(e)}.")

    try:
        result = await service.analyze_moodboard_images(user_session_id, image_bytes_list)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/vision/parse", response_model=UserVision)
async def parse_user_vision(
    # Required parameters first
    text: str,
    user_session_id: str,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    """
    Parse user vision text into a structured brief and persist it.

    Args:
        text: User's scene/brand intent.
        user_session_id: ID returned by /session/create.

    Returns:
        Persisted UserVision.
    """
    try:
        result = await service.parse_user_vision(text, user_session_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt/build", response_model=Prompt)
async def build_advertising_prompt(
    # Required parameters first
    focus_slider: int,
    user_session_id: str,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    """
    Build and persist an advertising prompt using prior analyses.

    Args:
        focus_slider: Balance between product and scene (0–10).
        user_session_id: ID returned by /session/create.

    Returns:
        Persisted Prompt.
    """
    # Endpoint-level validation: Input validation failures are user errors, not system errors
    # "Fail fast, Fail clear": catch bad inputs before any processing
    if not (0 <= focus_slider <= 10):
        # In FastAPI endpoints: direct user input validation should raise HTTPException immediately
        # ValueError gets caught and converted, but it's cleaner to be explicit
        raise HTTPException(status_code=400, detail="Focus slider must be between 0 and 10")

    # Service operations
    try:
        # Find records for this session
        image_analysis = service.session.exec(
            select(ImageAnalysis).where(ImageAnalysis.user_session_id == user_session_id)
        ).first()
        if not image_analysis:
            raise HTTPException(status_code=404, detail="No product image analysis found for this session")
        
        # When no moodboard analyses exist, service.session.exec(...) returns an empty list [].
        # SQLModel/SQLAlchemy automatically returns an empty list when no records match the query,
        # rather than raising an error or returning None. This makes moodboard truly optional.
        moodboard_analyses = service.session.exec(
            select(MoodboardAnalysis).where(MoodboardAnalysis.user_session_id == user_session_id)
        ).all()
        
        user_vision = service.session.exec(
            select(UserVision).where(UserVision.user_session_id == user_session_id)
        ).first()
        if not user_vision:
            raise HTTPException(status_code=404, detail="No user vision found for this session.")
        
        # Extract IDs
        moodboard_ids = [analysis.id for analysis in moodboard_analyses]
        
        # Build prompt
        result = await service.build_advertising_prompt(
            image_analysis.id,
            user_vision.id,
            focus_slider,
            user_session_id,
            moodboard_ids,
            is_refinement=False,
            previous_prompt_id=None,
            user_feedback=None
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt/complete", response_model=Prompt)
async def create_final_prompt(
    # Required  parameters
    product_file: UploadFile,
    text: str,
    focus_slider: int,
    user_session_id: str,
    # Optional parameters
    moodboard_files: list[UploadFile] | None = File(default=None),
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
    """
    One-shot: analyze product, optional moodboards, parse vision, build prompt.

    Args:
        product_file: Product image file upload.
        text: User's scene/brand intent.
        focus_slider: Balance between product and scene (0–10).
        user_session_id: ID returned by /session/create.
        moodboard_files: Optional moodboard images.

    Returns:
        Persisted Prompt ready for image generation.
    """
    # Input validation
    if not text:
        raise HTTPException(status_code=400, detail="User vision required.")

    if not (0 <= focus_slider <= 10):
        raise HTTPException(status_code=400, detail="Focus slider must be between 0 and 10.")

    # File operations
    try:
        # Read product image
        product_image_bytes = await product_file.read()

        # Read moodboard images (optional)
        moodboard_image_bytes_list = None
        if moodboard_files:
            moodboard_image_bytes_list = []
            for file in moodboard_files:
                image_bytes = await file.read()
                moodboard_image_bytes_list.append(image_bytes)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed {str(e)}.")

    # Service operations
    try:
        result = await service.create_final_prompt(
            product_image_bytes,
            text,
            focus_slider,
            user_session_id,
            moodboard_image_bytes_list
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt/refine", response_model=Prompt)
async def refine_prompt(
    # Required parameters
    user_session_id: str,
    # Optional parameters
    text: str | None = None,
    focus_slider: int | None = None,
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
    """
    Refine the latest prompt for this user session and persist a new version.

    Args:
        user_session_id: ID returned by /session/create.
        text: Optional feedback/instructions for refinement.
        focus_slider: Optional updated focus.

    Returns:
        Persisted refined Prompt.
    """
    # Input validation
    # focus_slider validation depends on database data, see service method in services.py
    if not text:
        raise HTTPException(status_code=400, detail="User vision required.")

    # Service operations
    try:
        prompt = service.session.exec(
            select(Prompt).where(Prompt.session_id == user_session_id).order_by(Prompt.id.desc())
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session.")
        
        result = await service.refine_prompt(
            previous_prompt_id=prompt.id,
            user_session_id=user_session_id,
            user_feedback=text,
            focus_slider=focus_slider
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/images/generate", response_model=GeneratedImage)
async def generate_image(
    # Required  parameters
    user_session_id: str,
    image_model_choice: str,
    # Optional parameters
    reference_files: list[UploadFile] | None = File(default=None),
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
    """
    Generate the final ad image from the saved prompt and optional references.

    Args:
        user_session_id: ID returned by /session/create.
        image_model_choice: Image generation model choice ("openai" or "gemini").
        reference_files: Optional reference images.

    Returns:
        Persisted GeneratedImage (served via /static when saved locally).
    """
    # Input validation
    if image_model_choice not in ["openai", "gemini"]:
        raise HTTPException(status_code=400, detail="image_model_choice must be 'openai' or 'gemini'")
        
    # File operations
    try:
        reference_image_bytes_list = None
        if reference_files:
            reference_image_bytes_list = []
            for file in reference_files:
                image_bytes = await file.read()
                reference_image_bytes_list.append(image_bytes)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed {str(e)}.")

    # Service operations
    try:
        # Find the prompt for this session
        prompt = service.session.exec(
            select(Prompt).where(Prompt.session_id == user_session_id).order_by(Prompt.id.desc())
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session")
        
        result = await service.generate_image(prompt.id, image_model_choice, user_session_id, reference_image_bytes_list)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ad/complete", response_model=GeneratedImage)
async def create_complete_ad(
    # Required parameters
    user_vision_text: str,
    focus_slider: int,
    product_file: UploadFile,
    user_session_id: str,
    image_model_choice: str,
    # Optional parameters
    moodboard_files: list[UploadFile] | None = File(default=None),
    reference_files: list[UploadFile] | None = File(default=None),
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
    """
    Full workflow: analyze, parse, build prompt, and generate final image.

    Args:
        user_vision_text: User's scene/brand intent.
        focus_slider: Balance between product and scene (0–10).
        product_file: Product image file upload.
        user_session_id: ID returned by /session/create.
        image_model_choice: Image generation model choice ("openai" or "google").
        moodboard_files: Optional moodboard images.
        reference_files: Optional reference images for generation.

    Returns:
        Persisted GeneratedImage.
    """
    # Input validation
    if not user_vision_text:
        raise HTTPException(status_code=400, detail="User vision required.")

    if not (0 <= focus_slider <= 10):
        raise HTTPException(status_code=400, detail="Focus slider must be between 0 and 10.")
    
    if image_model_choice not in ["openai", "google"]:
        raise HTTPException(status_code=400, detail="image_model_choice must be 'openai' or 'gemini'")

    # File operations
    try:
        # Read product image
        product_image_bytes = await product_file.read()
        
        # Read moodboard images (optional)
        moodboard_image_bytes_list = None
        if moodboard_files:
            moodboard_image_bytes_list = []
            for file in moodboard_files:
                image_bytes = await file.read()
                moodboard_image_bytes_list.append(image_bytes)
        
        # Read reference images if provided
        reference_image_bytes_list = None
        if reference_files:
            reference_image_bytes_list = []
            for file in reference_files:
                image_bytes = await file.read()
                reference_image_bytes_list.append(image_bytes)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File processing failed {str(e)}.")

    # Service operations
    try:
        result = await service.create_complete_ad(
            product_image_bytes,
            user_vision_text,
            focus_slider,
            user_session_id,
            image_model_choice,
            moodboard_image_bytes_list,
            reference_image_bytes_list
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# Helper function for formatinng history events
def format_event_text(event: HistoryEvent) -> str:
    """Convert HistoryEvent to readable chat-style text."""
    snapshot = event.snapshot_data or {}

    if event.event_type == "session_created":
        provider = snapshot.get("model_provider", "unknown")
        return f"Session started with {provider} model."
    
    elif event.event_type == "product_image_upload":
        return "User uploaded product image."
    
    elif event.event_type == "product_image_analyzed":
        product_type = snapshot.get("product_type", "not specified")
        product_category = snapshot.get("product_category", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Product analyzed by {provider}.\n"
            f"Detected type: {product_type}, category: {product_category}."
        )
    
    elif event.event_type == "moodboard_upload":
        return "User uploaded moodboard image."
    
    elif event.event_type == "moodboard_image_analyzed":
        visual_style = snapshot.get("visual_style", "not specified")
        mood_atmo = snapshot.get("mood_atmosphere", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Moodboard image analyzed by {provider}.\n"
            f"Visual style: {visual_style}\n"
            f"Mood: {mood_atmo}"
        )
    
    elif event.event_type == "user_vision_submitted":
        preview_text = snapshot.get("preview_text", "")
        if preview_text:
            return (
                "User submitted their vision.\n"
                f"Preview: {preview_text}"
            )
        return "User submitted their vision."
    
    elif event.event_type == "vision_parsed":
        focus_subject = snapshot.get("focus_subject", "not specified")
        setting = snapshot.get("setting", "not specified")
        provider = snapshot.get("model_provider", "unknown")
        return (
            f"Vision parsed and structured by {provider}.\n"
            f"Focus subject: {focus_subject}\n"
            f"Setting: {setting}"
        )
    
    elif event.event_type == "prompt_built":
        focus_slider = snapshot.get("focus_slider", "unknown")
        used_rag_examples = snapshot.get("used_rag_examples", False)
        provider = snapshot.get("model_provider", "unknown")
        rag_status = "Yes" if used_rag_examples else "No"
        return (
            f"Advertising prompt built by {provider}.\n"
            f"Focus slider: {focus_slider}\n"
            f"Prompt examples used: {rag_status}"
        )
    
    elif event.event_type == "prompt_refined":
        focus_slider = snapshot.get("focus_slider", "unknown")
        refinement_count = snapshot.get("refinement_count", "unknown")
        used_rag_examples = snapshot.get("used_rag_examples", False)
        provider = snapshot.get("model_provider", "unknown")
        rag_status = "Yes" if used_rag_examples else "No"
        return (
            f"Advertising prompt refined by {provider}.\n"
            f"Focus slider: {focus_slider}\n"
            f"Refinement count: {refinement_count}\n"
            f"Prompt examples used: {rag_status}"
        )
    
    elif event.event_type == "prompt_refinement_request":
        feedback = snapshot.get("user_feedback")
        focus_slider = snapshot.get("focus_slider")
        
        parts = ["User requested prompt refinement."]
        if feedback:
            parts.append(f"\nFeedback: {feedback}")
        if focus_slider is not None:
            parts.append(f"\nFocus slider: {focus_slider}")
        
        return "".join(parts)
    
    elif event.event_type == "image_model_chosen":
        model = snapshot.get("image_model", "unknown")
        return f"Image generation model chosen: {model}."
    
    elif event.event_type == "reference_image_upload":
        return "User uploaded reference image."
    
    elif event.event_type == "image_generated":
        model = snapshot.get("model", "unknown")
        return f"Ad image generated successfully with {model} model."
    
    else:
        return f"Unknown event: {event.event_type}" 


@app.get("/sessions/{user_session_id}/history")
async def get_session_history(
    user_session_id: str,
    page: int = 1,
    limit: int = 20,
    db_session: Session = Depends(get_db_session)
):
    """
    Retrieve paginated history events for a user session.

    Args:
        user_session_id: Session ID from URL path.
        page: Page number (default: 1). Query parameter: ?page=1
        limit: Events per page (default: 20, max: 100). Query parameter: ?limit=20

    Returns:
        {
            "events": [list of formatted event strings],
            "total": int,
            "page": int,
            "limit": int,
            "has_more": bool
        }
    """
    # Validate pagination 
    if limit > 100:
        limit = 100
    if limit < 1:
        limit = 20
    if page < 1:
        page = 1

    offset = (page - 1) * limit

    # Query events with pagination
    stmt = (
        select(HistoryEvent)
        .where(HistoryEvent.session_id == user_session_id)
        .order_by(HistoryEvent.created_at.asc())
        .limit(limit)
        .offset(offset)
    )
    events = db_session.exec(stmt).all()

    # Count total events for pagination metadata
    total_stmt = select(HistoryEvent).where(
        HistoryEvent.session_id == user_session_id
    )
    total = len(db_session.exec(total_stmt).all())

    # Format events
    formatted = []
    for event in events:
        formatted.append(format_event_text(event))
    
    # Calculate has_more: True if there are more pages after current page
    # Example: total=100, page=1, limit=20 -> has_more=True (pages 2-5 exist)
    # Example: total=100, page=5, limit=20 -> has_more=False (no page 6)
    has_more = (offset + limit) < total
    
    return {
        "events": formatted,
        "total": total,      # Total events: used by frontend to show "Page 1 of 5"
        "page": page,        # Current page: used by frontend for navigation
        "limit": limit,      # Page size: used by frontend to calculate total pages
        "has_more": has_more  # Convenience flag: frontend can show "Load More" button
    }


@app.post("/examples/save", response_model=PromptExample)
async def save_prompt_example(
    prompt_id: int,
    user_session_id: str,
    service: AdGeneratorService = Depends(get_service)
):
    """
    Persist a PromptExample from an existing Prompt for RAG retrieval.

    Notes:
        - This endpoint requires user_session_id because get_service depends on it to construct a session-scoped service. We also validate that the Prompt belongs to this session to prevent cross-session writes.

    Args:
        prompt_id: ID of the Prompt to convert into a PromptExample.
        user_session_id: ID returned by /session/create (scopes the request and validation).

    Returns:
        PromptExample: Newly created example row linked to the prompt's category.

    Raises:
        HTTPException(400): If the prompt is missing, the session mismatches, or the linked analysis is not found.
    """
    try:
        return await service.save_prompt_example(prompt_id, user_session_id)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/test/text-only")
async def test_text_only(
    # Required user parameters first
    text: str,
    user_session_id: str,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    """
    Test-only: parse user vision text without images.

    Args:
        text: User-provided text.
        user_session_id: ID returned by /session/create.

    Returns:
        Persisted UserVision.
    """
    try:
        # Simple test - just parse user vision text (no images)
        result = await service.parse_user_vision(text, user_session_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5001,
        reload=True,  # Only for development
        log_level="info"
    )

