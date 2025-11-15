# Ad Image Generator

An AI-powered advertising image generation application that creates compelling marketing visuals from product images, moodboards, and user vision descriptions. Built with FastAPI and Next.js, supporting both OpenAI and Google Gemini models.

## Features

- **Multi-Provider AI Support**: Choose between OpenAI (GPT-4.1) or Google Gemini (Gemini 2.5 Flash) for vision and text analysis tasks
- **Flexible Image Generation**: Generate images using either OpenAI's `gpt-image-1` or Google's `gemini-2.5-flash-image` models
- **Product Image Analysis**: Automated analysis of product images extracting style, colors, materials, and brand elements
- **Moodboard Analysis**: Analyze moodboard images to extract visual style, mood, and composition patterns
- **User Vision Parsing**: Convert natural language descriptions into structured advertising briefs
- **RAG-Enhanced Prompt Building**: Few-shot prompting with category-matched examples for better prompt quality
- **Prompt Refinement**: Iteratively refine prompts with user feedback (up to 2 refinements)
- **Session Management**: Persistent sessions with full workflow history tracking
- **Complete Workflow APIs**: Step-by-step or one-shot endpoints for different use cases

## Architecture

### Dual Provider, Multi-Model System

The application uses a dual-provider architecture where users select their preferred AI provider at session creation:

**Vision & Text Analysis (via `Agents` class):**
- **OpenAI**: `gpt-4.1` model for:
  - Product image analysis (vision + text)
  - Moodboard image analysis (vision + text)
  - User vision parsing (text only)
  - Prompt building (text only)
- **Google Gemini**: `gemini-2.5-flash` model as an alternative provider for the same tasks

**Image Generation (via `image_generator` modules):**
- **OpenAI**: `gpt-image-1` model via Responses API
- **Google Gemini**: `gemini-2.5-flash-image` model

**Key Design Principles:**
- User choice between providers for vision and text analysis
- Independent choice for image generation (can mix providers)
- Fail-fast validation prevents runtime errors
- Session-scoped provider consistency for analysis operations

### Database Architecture

- **SQLite** database with WAL (Write-Ahead Logging) mode for better concurrency
- **SQLModel** for type-safe database operations
- Per-request database sessions with automatic cleanup
- Session-based workflow tracking with history events

## Tech Stack

### Backend
- **FastAPI**: Modern Python web framework
- **SQLModel**: SQL database ORM with Pydantic integration
- **SQLite**: Lightweight database with WAL mode
- **pydantic-ai**: AI agent framework for structured outputs
- **OpenAI SDK**: OpenAI API integration
- **Google GenAI SDK**: Google Gemini API integration
- **Pillow**: Image processing

### Frontend
- **Next.js 15**: React framework with App Router
- **React 19**: UI library
- **Tailwind CSS**: Utility-first CSS framework
- **Radix UI**: Accessible component primitives
- **shadcn/ui**: Component library

## Project Structure

```
ad_image_gen/
├── main.py                 # FastAPI application and endpoints
├── services.py             # Business logic and service layer
├── agents.py               # AI agents for vision and text analysis tasks
├── models.py               # SQLModel database models
├── api/
│   ├── gpt_image1_generator.py    # OpenAI image generation
│   └── gemini_image_generator.py  # Google Gemini image generation
├── frontend/               # Next.js frontend application
│   ├── app/                # Next.js app directory
│   ├── components/         # React components
│   ├── hooks/              # Custom React hooks
│   ├── lib/                # Utility functions and API client
│   └── package.json       # Frontend dependencies
├── uploads/                # Uploaded images (product, moodboards, references)
├── output_images/          # Generated ad images
├── database.db             # SQLite database (created on first run)
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (create this)
```

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **OpenAI API Key** (for OpenAI operations)
- **Google Gemini API Key** (for Gemini operations)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ad_image_gen
```

### 2. Backend Setup

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd frontend
npm install
cd ..
```

### 4. Environment Variables

Create a `.env` file in the root directory:

```env
# OpenAI API Key (required for OpenAI vision/text analysis and image generation)
MY_OPENAI_API_KEY=your_openai_api_key_here

# Google Gemini API Key (required for Gemini vision/text analysis and image generation)
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Enable database ping endpoint for health checks
ENABLE_DB_PING=true
```

**Note**: You need at least one API key depending on which provider(s) you want to use. Both keys are required if you want to support both providers.

## Running the Application

### Start the Backend

```bash
# From the root directory
python main.py
```

The backend will start on `http://localhost:5001`

### Start the Frontend

```bash
# From the root directory
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

### Access the Application

Open your browser and navigate to `http://localhost:3000`

## API Endpoints

### Session Management

#### `POST /session/create`
Create a new user session with a selected model provider.

**Query Parameters:**
- `model_provider` (str): `"openai"` or `"gemini"`

**Response:**
```json
{
  "user_session_id": "uuid-string",
  "model_provider": "openai"
}
```

### Analysis Endpoints

#### `POST /analyze/product-image`
Analyze a product image and extract structured information.

**Form Data:**
- `file` (file): Product image file
- `user_session_id` (str): Session ID

**Response:** `ImageAnalysis` object

#### `POST /analyze/moodboard`
Analyze one or more moodboard images.

**Query Parameters:**
- `user_session_id` (str): Session ID

**Form Data:**
- `files` (list[file], optional): Moodboard image files

**Response:** List of `MoodboardAnalysis` objects

#### `POST /vision/parse`
Parse user vision text into structured brief.

**Query Parameters:**
- `text` (str): User's vision description
- `user_session_id` (str): Session ID

**Response:** `UserVision` object

### Prompt Building

#### `POST /prompt/build`
Build an advertising prompt from prior analyses.

**Query Parameters:**
- `focus_slider` (int): Balance between product and scene (0-10)
- `user_session_id` (str): Session ID

**Response:** `Prompt` object

#### `POST /prompt/complete`
One-shot endpoint: analyze product, optional moodboards, parse vision, and build prompt.

**Query Parameters:**
- `user_session_id` (str): Session ID

**Form Data:**
- `product_file` (file): Product image
- `text` (str): User vision text
- `focus_slider` (int): Focus balance (0-10)
- `moodboard_files` (list[file], optional): Moodboard images

**Response:** `Prompt` object

#### `POST /prompt/refine`
Refine an existing prompt with user feedback.

**Query Parameters:**
- `user_session_id` (str): Session ID
- `text` (str, optional): Refinement feedback
- `focus_slider` (int, optional): Updated focus value

**Response:** `Prompt` object (refined)

### Image Generation

#### `POST /images/generate`
Generate the final ad image from a saved prompt.

**Query Parameters:**
- `user_session_id` (str): Session ID
- `image_model_choice` (str): `"openai"` or `"gemini"`

**Form Data:**
- `reference_files` (list[file], optional): Reference images

**Response:** `GeneratedImage` object

#### `POST /ad/complete`
Full workflow: analyze, parse, build prompt, and generate image in one call.

**Query Parameters:**
- `user_session_id` (str): Session ID

**Form Data:**
- `user_vision_text` (str): User's vision description
- `focus_slider` (int): Focus balance (0-10)
- `product_file` (file): Product image
- `image_model_choice` (str): `"openai"` or `"gemini"`
- `moodboard_files` (list[file], optional): Moodboard images
- `reference_files` (list[file], optional): Reference images

**Response:** `GeneratedImage` object

### Session Status & History

#### `GET /sessions/{user_session_id}/status`
Get complete session status with all analyses, prompt, and generated image.

**Response:**
```json
{
  "session_exists": true,
  "session_id": "uuid-string",
  "model_provider": "openai",
  "image_analysis": {...},
  "moodboard_analyses": [...],
  "user_vision": {...},
  "prompt": {...},
  "generated_image": {...}
}
```

#### `GET /sessions/{user_session_id}/history`
Get paginated history events for a session.

**Query Parameters:**
- `page` (int, default: 1): Page number
- `limit` (int, default: 20, max: 100): Events per page

**Response:**
```json
{
  "events": [
    {
      "id": 1,
      "text": "Session started with openai model.",
      "created_at": "2024-01-01T00:00:00"
    }
  ],
  "total": 10,
  "page": 1,
  "limit": 20,
  "has_more": false
}
```

### Utility Endpoints

#### `POST /examples/save`
Save a prompt as an example for RAG retrieval.

**Query Parameters:**
- `prompt_id` (int): Prompt ID to save
- `user_session_id` (str): Session ID

**Response:** `PromptExample` object

#### `GET /db/ping` (if `ENABLE_DB_PING=true`)
Health check endpoint for database connection.

**Response:**
```json
{
  "status": "ok"
}
```

## Usage Examples

### Step-by-Step Workflow

```python
import requests

# 1. Create session
response = requests.post(
    "http://localhost:5001/session/create",
    params={"model_provider": "openai"}
)
session_id = response.json()["user_session_id"]

# 2. Analyze product image
with open("product.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5001/analyze/product-image",
        files={"file": f},
        params={"user_session_id": session_id}
    )
    product_analysis = response.json()

# 3. Parse user vision
response = requests.post(
    "http://localhost:5001/vision/parse",
    params={
        "text": "Professional woman in modern office during golden hour",
        "user_session_id": session_id
    }
)
user_vision = response.json()

# 4. Build prompt
response = requests.post(
    "http://localhost:5001/prompt/build",
    params={
        "focus_slider": 5,
        "user_session_id": session_id
    }
)
prompt = response.json()

# 5. Generate image
response = requests.post(
    "http://localhost:5001/images/generate",
    params={
        "user_session_id": session_id,
        "image_model_choice": "openai"
    }
)
generated_image = response.json()
```

### One-Shot Workflow

```python
# Complete workflow in one call
with open("product.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5001/ad/complete",
        files={
            "product_file": f,
            "moodboard_files": [open("mood1.jpg", "rb"), open("mood2.jpg", "rb")]
        },
        data={
            "user_vision_text": "Professional woman in modern office during golden hour",
            "focus_slider": 5,
            "user_session_id": session_id,
            "image_model_choice": "openai"
        }
    )
    result = response.json()
```

## Database Schema

### Core Models

- **UserSession**: Stores session information and model provider choice
- **ImageAnalysis**: Product image analysis results
- **MoodboardAnalysis**: Moodboard image analysis results
- **UserVision**: Parsed user vision structured data
- **Prompt**: Generated advertising prompts with metadata
- **GeneratedImage**: Final generated ad images
- **PromptExample**: Saved prompts for RAG retrieval
- **HistoryEvent**: Workflow history events for timeline display

### Relationships

- All analysis models reference `UserSession` via `session_id`
- `Prompt` references `ImageAnalysis`, `UserVision`, and optionally `MoodboardAnalysis`
- `GeneratedImage` references `Prompt`
- `PromptExample` references `Prompt` and uses `product_category` for RAG

## Development Guidelines

### Code Style

- **Python**: Follow PEP 8 standards
- **Import Ordering**: PEP 8 import ordering (stdlib, third-party, local)
- **Type Hints**: Use Python 3.10+ type hints (built-in generics, not `typing.List`)
- **Async/Await**: All AI operations are async to prevent blocking

### Database Best Practices

- Use per-request database sessions (FastAPI dependency injection)
- Commit and close sessions before long-running AI calls to prevent locks
- Use WAL mode for better SQLite concurrency
- Always handle rollbacks in exception handlers

### API Design

- Use FastAPI dependency injection for services and database sessions
- Validate inputs at the endpoint level
- Return structured error messages with appropriate HTTP status codes
- Use response models for type-safe API responses

### Frontend Development

- API calls are centralized in `frontend/lib/api.js`
- Use custom hooks (`useAdGenerator.js`) for state management
- Components use shadcn/ui for consistent UI
- Session state is persisted in localStorage

## Troubleshooting

### Database Lock Errors

If you encounter "database is locked" errors:
- Ensure WAL mode is enabled (automatic on connection)
- Check that sessions are properly closed after long operations
- Increase SQLite timeout in `main.py` if needed

### API Key Errors

- Verify API keys are set in `.env` file
- Check that the correct provider's key is set for your chosen model
- Ensure API keys have sufficient credits/permissions

### Image Generation Failures

- Verify image file formats are supported (JPEG, PNG)
- Check API response logs for detailed error messages
- Ensure product image is accessible and valid

### Frontend Connection Issues

- Verify backend is running on `http://localhost:5001`
- Check CORS settings in `main.py` if accessing from different origin
- Ensure API_BASE_URL in `frontend/lib/api.js` matches backend URL

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

