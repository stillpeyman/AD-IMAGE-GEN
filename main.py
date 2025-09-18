"""
FastAPI app wiring:
- Loads env keys for two OpenAI clients
- Sets up SQLite engine and table creation on startup
- Provides per-request DB Session via Depends
- Provides AdGeneratorService via Depends for endpoints
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
from pydantic_core.core_schema import NoneSchema
from sqlmodel import SQLModel, Session, create_engine, select

# local
from agents import Agents
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage
from services import AdGeneratorService
import uuid


# Test session for MVP
TEST_SESSION_ID = "test-session-123"


# 1) env
load_dotenv()
TEXT_API_KEY = os.getenv("MS_OPENAI_API_KEY")
IMG_API_KEY = os.getenv("MY_OPENAI_API_KEY")
IMG_MODEL = "gpt-image-1"  # The only supported model for image generation

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


def get_session():
    """
    Provide a database Session for the current request.

    Plain English: “Open a fresh Session for this request, hand it to the endpoint,
    and after the response is sent, close it.”
    The `yield` hands control back to FastAPI; after the endpoint completes,
    FastAPI runs the code after `yield` (which closes the Session).
    """
    with Session(engine) as session:
        yield session  # FastAPI will close the session after the response


# 3) app + agents
app = FastAPI()

# Serve generated images from local disk under /static
app.mount("/static", StaticFiles(directory="output_images"), name="static")

agents = Agents(
    text_openai_api_key=TEXT_API_KEY,
    text_model_name="gpt-4o-mini",
)


def get_service(session: Session = Depends(get_session)) -> AdGeneratorService:
    """
    Build an AdGeneratorService bound to this request's Session.

    Depends(get_session) means: "Before calling this, first call get_session()
    and give me its returned Session."
    """
    return AdGeneratorService(
        agents=agents, 
        session=session,
        img_model=IMG_MODEL,
        img_api_key=IMG_API_KEY
    )


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


if ENABLE_DB_PING:
    @app.get("/db/ping")
    def db_ping(service: AdGeneratorService = Depends(get_service)):
        # Executes a harmless SQL statement using the injected service + session
        service.session.exec(select(1)).first()
        return {"status": "ok"}


@app.post("/session/start")
async def start_session():
    """Start a new session and return session_id for use in subsequent API calls."""
    session_id = str(uuid.uuid4())
    return {"session_id": session_id}


@app.post("/analyze/product-image", response_model=ImageAnalysis)
async def analyze_product_image(
    # Required parameters first
    file: UploadFile,
    session_id: str = TEST_SESSION_ID,
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
        result = await service.analyze_product_image(image_bytes, session_id)
        return result

    # Endpoint layer catches service's ValueError and converts to HTTP responce
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/moodboard", response_model=list[MoodboardAnalysis])
async def analyze_moodboard_images(
    # Required parameters first
    session_id: str = TEST_SESSION_ID,
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
        result = await service.analyze_moodboard_images(session_id, image_bytes_list)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/vision/parse", response_model=UserVision)
async def parse_user_vision(
    # Required parameters first
    text: str,
    session_id: str = TEST_SESSION_ID,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    try:
        result = await service.parse_user_vision(text, session_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt/build", response_model=Prompt)
async def build_advertising_prompt(
    # Required parameters first
    focus_slider: int,
    session_id: str = TEST_SESSION_ID,
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
            select(ImageAnalysis).where(ImageAnalysis.session_id == session_id)
        ).first()
        if not image_analysis:
            raise HTTPException(status_code=404, detail="No product image analysis found for this session")
        
        # When no moodboard analyses exist, service.session.exec(...) returns an empty list [].
        # SQLModel/SQLAlchemy automatically returns an empty list when no records match the query,
        # rather than raising an error or returning None. This makes moodboard truly optional.
        moodboard_analyses = service.session.exec(
            select(MoodboardAnalysis).where(MoodboardAnalysis.session_id == session_id)
        ).all()
        
        user_vision = service.session.exec(
            select(UserVision).where(UserVision.session_id == session_id)
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
            session_id,
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
    session_id: str = TEST_SESSION_ID,
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
            session_id,
            moodboard_image_bytes_list
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/prompt/refine", response_model=Prompt)
async def refine_prompt(
    # Required parameters
    session_id: str = TEST_SESSION_ID,
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
            select(Prompt).where(Prompt.session_id == session_id)
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session.")
        
        result = await service.refine_prompt(
            prompt.id,
            session_id,
            focus_slider,
            text
        )
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/images/generate", response_model=GeneratedImage)
async def generate_image(
    # Required  parameters
    session_id: str = TEST_SESSION_ID,
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
            select(Prompt).where(Prompt.session_id == session_id)
        ).first()
        if not prompt:
            raise HTTPException(status_code=404, detail="No prompt found for this session")
        
        result = await service.generate_image(prompt.id, reference_image_bytes_list, session_id)
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ad/complete", response_model=GeneratedImage)
async def create_complete_ad(
    # Required parameters
    user_vision_text: str,
    focus_slider: int,
    product_file: UploadFile,
    session_id: str = TEST_SESSION_ID,
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
            session_id,
            reference_image_bytes_list
        )
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/test/text-only")
async def test_text_only(
    # Required user parameters first
    text: str,
    session_id: str = TEST_SESSION_ID,
    # Dependency injection last
    service: AdGeneratorService = Depends(get_service)
):
    try:
        # Simple test - just parse user vision text (no images)
        result = await service.parse_user_vision(text, session_id)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

