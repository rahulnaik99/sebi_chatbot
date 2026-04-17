import os
from dotenv import load_dotenv
load_dotenv()
os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"

from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
import tempfile

from app.chatbot import chat_bot
from app.services.ingestion import run_ingestion
from app.services.get_weaviate import close_weaviate_client
from app.core.logger import get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("LoanIQ starting up...")
    yield
    log.info("LoanIQ shutting down...")
    close_weaviate_client()


app = FastAPI(title="LoanIQ RAG API", version="2.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    status: str
    chunks_indexed: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "LoanIQ RAG API v2"}


@app.post("/ask", response_model=ChatResponse)
def ask(request: ChatRequest):
    """Ask a question — runs full hybrid retrieval + RAG."""
    try:
        result = chat_bot(request.question, request.session_id)
        return result
    except Exception as e:
        log.error(f"Error in /ask: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    """
    Upload a PDF, DOCX, or TXT file.
    Runs the full ingestion pipeline: load → chunk → embed → index.
    """
    allowed = {".pdf", ".docx", ".txt"}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed}",
        )

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        log.info(f"Ingesting uploaded file: {file.filename}")
        run_ingestion(tmp_path)
        return IngestResponse(
            status="success",
            chunks_indexed=f"File '{file.filename}' ingested successfully.",
        )
    except Exception as e:
        log.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
