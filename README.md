# Multi-Model RAG System

Local multimodal Retrieval-Augmented Generation (RAG) app built with Streamlit.

The app lets you upload documents, index them into a local vector store, and ask grounded questions with source-backed answers.

## Features

- Supports multiple document types:
	- PDF (`.pdf`)
	- Text/Markdown (`.txt`, `.md`)
	- Images with OCR (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tif`, `.tiff`, `.webp`)
	- Office files (`.docx`, `.pptx`, `.xlsx`, `.odt`)
- Local persistent vector database using Chroma (`chroma_db/`)
- Hybrid retrieval (semantic + keyword reranking)
- Query expansion for better recall
- File-level indexing and deletion from UI
- Caching and basic usage metrics
- LLM answer generation via OpenRouter API
- Grounded fallback answers when LLM is unavailable/rate-limited

## Tech Stack

- Frontend: Streamlit
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2`
- Vector DB: Chroma (persistent local storage)
- OCR: Tesseract via `pytesseract`
- LLM Provider: OpenRouter (OpenAI-compatible chat completions)

## Prerequisites

- Python 3.10+
- Tesseract OCR installed and available in PATH (for image text extraction)
- OpenRouter API key (for AI-generated answers)

## Quick Start

1. Clone and enter project

```powershell
git clone https://github.com/rafayfarooq768/Multi-Model-RAG-System.git
cd Multi-Model-RAG-System
```

2. Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies

```powershell
pip install -r requirements.txt
```

4. Configure local secrets

Create `.streamlit/secrets.toml` (local only):

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
```

5. Run app

```powershell
streamlit run app.py
```

## Usage Flow

1. Open the app.
2. Upload files from sidebar `Add Files`.
3. Select search quality (`Fast`, `Balanced`, `Thorough`).
4. Ask questions in natural language.
5. Verify answers in `Sources` section.
6. Remove files from `Your Files` tab when needed.

## Security: API Keys

- Never hardcode API keys in source files.
- Keep secrets only in local `.streamlit/secrets.toml` or Streamlit Cloud secrets.
- This repo ignores secrets via `.gitignore`.
- If a key was pasted in chat/history, rotate it immediately.

## Project Structure

- `app.py` - Streamlit UI and user flow
- `src/config.py` - Constants and defaults
- `src/ingestion.py` - Loaders for PDF/text/image/office files
- `src/chunking.py` - Document chunking
- `src/embeddings.py` - Embedding function wrapper
- `src/vectorstore.py` - Chroma persistence and manifest logic
- `src/retrieval.py` - Query expansion, reranking, grounded fallback formatting
- `src/llm.py` - OpenRouter LLM client
- `src/pipeline.py` - End-to-end orchestration
- `data/` - Local input files (uploads live in `data/uploads/`)
- `tests/` - Core tests

## Running Tests

```powershell
pytest
```

## Deployment (Streamlit Cloud)

1. Push code to GitHub (without local secrets file).
2. Create Streamlit Cloud app from repo.
3. Add secret in Streamlit Cloud:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
```

4. Deploy.

## Troubleshooting

- `LLM generation error: HTTP Error 429`
	- OpenRouter/provider rate-limited the request. The app will fall back to grounded retrieval answer.

- Answers seem too short
	- Use `Balanced` or `Thorough` quality.
	- Rephrase as a specific question and verify relevant file is indexed.

- Removed file reappears
	- Uploaded files are now removed from both index and `data/uploads` when deleted from UI.

- OCR not working
	- Install Tesseract and verify it is available in PATH.

## Notes

- This project is a local MVP optimized for rapid experimentation.
- Auth/multi-tenant controls are intentionally not included.
