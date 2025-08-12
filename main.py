import os  # stdlib

from dotenv import load_dotenv  # third-party
from fastapi import FastAPI, Depends
from sqlmodel import SQLModel, Session, create_engine

from agents import Agents  # local
from models import ImageAnalysis, MoodboardAnalysis, UserVision, Prompt, GeneratedImage

# 1) env
load_dotenv()
TEXT_API_KEY = os.getenv("MS_OPENAI_API_KEY")
IMG_API_KEY = os.getenv("MY_OPENAI_API_KEY")

# 2) database
DATABASE_FILE = "database.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session  # FastAPI will close the session after the response

# 3) app + agents
app = FastAPI()
agents = Agents(
    text_openai_api_key=TEXT_API_KEY,
    img_openai_api_key=IMG_API_KEY,
    model_name="openai:gpt-4o",
    provider="openai",
)

@app.on_event("startup")
def on_startup() -> None:
    create_db_and_tables()