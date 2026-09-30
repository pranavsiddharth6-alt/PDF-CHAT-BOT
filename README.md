# PDF Chatbot

A full-stack PDF Chatbot application that uses Retrieval-Augmented Generation (RAG) and Hugging Face models to answer questions about uploaded PDF documents.

## Project Structure

```
pdf-chatbot/
├── backend/          # FastAPI backend server
│   ├── main.py       # Application entry point
│   └── requirements.txt
├── frontend/         # React (Vite) frontend
├── data/             # Storage for uploaded PDFs and processed data
├── .env.example      # Environment variable template
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+

### Backend Setup

```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

### Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

## Tech Stack

- **Backend:** Python, FastAPI
- **Frontend:** React, Vite
- **AI/ML:** Hugging Face
- **Vector DB:** Chroma Cloud

