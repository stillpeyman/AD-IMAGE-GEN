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
- MY_OPENAI_API_KEY for OpenAI operations (gpt-4o-mini for text, gpt-image-1 for images)
- GEMINI_API_KEY for Google operations (gemini-1.5-flash for text)
- User chooses model provider at session start (no defaults)
  
Text Analysis (via Agents class):
- OpenAI: gpt-4o-mini (smart, cheap, fast)
- Google: gemini-1.5-flash (alternative provider)
- Jobs: Product analysis, moodboard analysis, user vision parsing, prompt building
- Validation: API keys checked at Agents.__init__() based on provider

Image Generation (via image_generator module):  
- Model: gpt-image-1 (OpenAI only - specialized for images)
- Jobs: Generate final ad images
- Validation: API key validated at AdGeneratorService.__init__()

Why this design:
- User choice between providers for text analysis
- Single image generation provider (OpenAI)
- Fail-fast validation prevents runtime errors
"""

"""
NOTE TO MYSELF for better organizing code in separate files:
see https://fastapi.tiangolo.com/tutorial/bigger-applications/#import-apirouter
>>> "Multiple Files/APIRouter"
"""
# stdlib
import os

# third-party
from dotenv import load_dotenv
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from sqlmodel import SQLModel, Session, create_engine, select

# local
from agents import Agents
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage, UserSession
from services import AdGeneratorService
import uuid


# Test session for MVP
TEST_SESSION_ID = "test-session-123"


# 1) env
load_dotenv()
OPENAI_API_KEY = os.getenv("MY_OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Image generation models (separate from text models)
OPENAI_IMG_MODEL = "gpt-image-1" 

# Enable debug endpoint only when ENABLE_DB_PING=true in .env
ENABLE_DB_PING = os.getenv("ENABLE_DB_PING", "false").lower() == "true"


# 2) database
DATABASE_FILE = "database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL, echo=True)


def create_db_and_tables() -> None:
    """
    Create all tables defined on SQLModel metadata if they don't exist yet.
    Idempotent: calling multiple times will not overwrite existing tables.
    """
    SQLModel.metadata.create_all(engine)


# 3) App creation
app = FastAPI()

# Serve generated images from local disk under /static
app.mount("/static", StaticFiles(directory="output_images"), name="static")


# 4) Dependency and Service functions
def get_db_session():
    """
    Provide a database connection for the current request.
    
    This is a SQLAlchemy database session - it's just a connection to the database.
    It's NOT a user session. It allows us to query/insert/update records.
    """
    with Session(engine) as db_session:
        # yield -> FastAPI will close the session after the response
        yield db_session


def get_service(user_session_id: str, db_session: Session = Depends(get_db_session)) -> AdGeneratorService:
    """
    Create AdGeneratorService with the model provider from the user session.

    Depends(get_db_session) means: "Before calling this, first call get_db_session()
    and give me its returned database connection."
    
    Args:
        user_session_id: The user session ID to get model provider from
        db_session: Database connection for querying user session
        
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
        img_model=OPENAI_IMG_MODEL,         
        openai_api_key=OPENAI_API_KEY 
    )


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


if ENABLE_DB_PING:
    @app.get("/db/ping")
    def db_ping(db_session: Session = Depends(get_db_session)):
        # Simple health check - just test database connection
        db_session.exec(select(1)).first()
        return {"status": "ok"}


@app.post("/session/create")
async def create_session(model_provider: str, db_session: Session = Depends(get_db_session)):
    """
    Create a new user session with the chosen model provider.
    
    This creates a user session (not a database session) that stores
    which AI model the user wants to use for their workflow.
    
    Args:
        model_provider: "openai" or "google" 
        db_session: Database connection for storing user session
        
    Returns:
        dict: user_session_id and model_provider
    """
    if model_provider not in ["openai", "google"]:
        raise HTTPException(status_code=400, detail="model_provider must be 'openai' or 'google'")
    
    try:
        user_session_id = str(uuid.uuid4())
        session_record = UserSession(id=user_session_id, model_provider=model_provider)
        db_session.add(session_record)
        db_session.commit()
        db_session.refresh(session_record)
        
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
    # Input validation
    # focus_slider validation depends on database data, see service method in services.py
    if not text:
        raise HTTPException(status_code=400, detail="User vision required.")

    # Service operations
    try:
        prompt = service.session.exec(
            select(Prompt).where(Prompt.session_id == user_session_id)
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session.")
        
        result = await service.refine_prompt(
            prompt.id,
            user_session_id,
            focus_slider,
            text
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/images/generate", response_model=GeneratedImage)
async def generate_image(
    # Required  parameters
    user_session_id: str,
    # Optional parameters
    reference_files: list[UploadFile] | None = File(default=None),
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
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
            select(Prompt).where(Prompt.session_id == user_session_id)
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session")
        
        result = await service.generate_image(prompt.id, reference_image_bytes_list, user_session_id)
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
    # Optional parameters
    moodboard_files: list[UploadFile] | None = File(default=None),
    reference_files: list[UploadFile] | None = File(default=None),
    # Dependency injection
    service: AdGeneratorService = Depends(get_service)
):
    # Input validation
    if not user_vision_text:
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
            moodboard_image_bytes_list,
            user_vision_text,
            focus_slider,
            user_session_id,
            reference_image_bytes_list
        )
        return result

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
    try:
        # Simple test - just parse user vision text (no images)
        result = await service.parse_user_vision(text, user_session_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

