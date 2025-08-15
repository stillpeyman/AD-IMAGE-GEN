"""
FastAPI app wiring:
- Loads env keys for two OpenAI clients
- Sets up SQLite engine and table creation on startup
- Provides per-request DB Session via Depends
- Provides AdGeneratorService via Depends for endpoints
"""

import os  # stdlib

from dotenv import load_dotenv  # third-party
from fastapi import FastAPI, Depends, File, UploadFile, HTTPException
from sqlmodel import SQLModel, Session, create_engine, select

from agents import Agents  # local
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage
from services import AdGeneratorService


# 1) env
load_dotenv()
TEXT_API_KEY = os.getenv("MS_OPENAI_API_KEY")
IMG_API_KEY = os.getenv("MY_OPENAI_API_KEY")

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
agents = Agents(
    text_openai_api_key=TEXT_API_KEY,
    img_openai_api_key=IMG_API_KEY,
    text_model_name="gpt-4o-mini",
    img_model_name="gpt-4.1",
    provider="openai",
)


def get_service(session: Session = Depends(get_session)) -> AdGeneratorService:
    """
    Build an AdGeneratorService bound to this request's Session.

    Depends(get_session) means: “Before calling this, first call get_session()
    and give me its returned Session.”
    """
    return AdGeneratorService(agents=agents, session=session)


@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()


if ENABLE_DB_PING:
    @app.get("/db/ping")
    def db_ping(service: AdGeneratorService = Depends(get_service)):
        # Executes a harmless SQL statement using the injected service + session
        service.session.exec(select(1)).first()
        return {"status": "ok"}


@app.post("/analyze/product-image", response_model=ImageAnalysis)
async def analyze_product_image(
    file: UploadFile,
    service: AdGeneratorService = Depends(get_service)
):
    try:
        image_bytes = await file.read()
        result = await service.analyze_product_image(image_bytes)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/test/text-only")
async def test_text_only(
    text: str,
    service: AdGeneratorService = Depends(get_service)
):
    try:
        # Simple test - just parse user vision text (no images)
        result = await service.parse_user_vision(text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

