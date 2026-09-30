import logging
from dotenv import load_dotenv

# Load backend/.env before anything else
load_dotenv()

# Configure standard structured application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("pdf_chatbot")

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import document_routes, search_routes, chat_routes

app = FastAPI(
    title="PDF Chatbot API",
    description="A RAG-based chatbot API for answering questions about PDF documents.",
    version="0.1.0",
)

# Build allowed origins from environment variable (comma-separated).
# Falls back to localhost:5173 for local development when not set.
origins = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173"
    ).split(",")
    if origin.strip()
]

# Allow the React frontend to communicate with this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(document_routes.router)
app.include_router(search_routes.router)
app.include_router(chat_routes.router)

logger.info("PDF Chatbot FastAPI application initialized with document, search, and chat routes.")


@app.get("/")
def root():
    """Health check endpoint."""
    return {"status": "ok", "message": "PDF Chatbot API is running."}
