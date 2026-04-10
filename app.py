from __future__ import annotations

import os
import logging
from pathlib import Path

import streamlit as st

from src.config import SEARCH_QUALITY_PRESETS, SUPPORTED_DOC_TYPES, SUPPORTED_EXTENSIONS
from src.pipeline import LocalRAGPipeline

# Configure logging to avoid cluttering the app
logging.basicConfig(level=logging.WARNING)

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"


def get_startup_files() -> list[Path]:
    """Return persistent files that should be indexed automatically on startup.

    User uploads are kept under data/uploads and are indexed only when the user
    adds them. That prevents removed uploads from being re-attached on restart.
    """
    if not DATA_DIR.exists():
        return []

    files: list[Path] = []
    for path in DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        if UPLOAD_DIR in path.parents:
            continue
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(path)

    return sorted(files)


def get_pipeline() -> LocalRAGPipeline:
    if "pipeline" not in st.session_state:
        pipeline = LocalRAGPipeline(openrouter_api_key=get_openrouter_api_key())
        startup_files = get_startup_files()
        if startup_files:
            pipeline.ingest_paths(startup_files)
        st.session_state.pipeline = pipeline
    return st.session_state.pipeline


def get_openrouter_api_key() -> str | None:
    try:
        return st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        return os.getenv("OPENROUTER_API_KEY")


def save_upload(uploaded_file) -> Path:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target_path = UPLOAD_DIR / uploaded_file.name
    target_path.write_bytes(uploaded_file.getbuffer())
    return target_path


def remove_uploaded_file(source: str) -> bool:
    """Remove an uploaded file from disk if it exists in data/uploads."""
    target_path = UPLOAD_DIR / source
    if target_path.exists() and target_path.is_file():
        target_path.unlink()
        return True
    return False


def check_ocr_readiness() -> bool:
    """Check if Tesseract OCR is available on the system."""
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def format_file_size(path: Path) -> str:
    """Format file size in human-readable format."""
    size_bytes = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


st.set_page_config(page_title="MultiModal RAG System", layout="wide")
st.title("Ask Your Documents")
st.caption("Upload your files, ask a question, and get an answer with sources.")

pipeline = get_pipeline()

# Sidebar Navigation
with st.sidebar:
    st.header("Document Assistant")
    
    tab1, tab2 = st.tabs(["Add Files", "Your Files"])
    
    # TAB 1: Ingestion
    with tab1:
        st.subheader("Upload Files")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=sorted(extension.removeprefix(".") for extension in SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
        )
        
        if st.button("Add files", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Adding files..."):
                    saved_paths = [save_upload(uploaded_file) for uploaded_file in uploaded_files]
                    chunk_count = pipeline.ingest_paths(saved_paths)
                    st.success(f"Added {len(saved_paths)} file(s).")
                    st.rerun()
            else:
                st.warning("No files selected.")
        
        st.divider()
        
        # OCR Status
        ocr_ready = check_ocr_readiness()
        if ocr_ready:
            st.success("Image text reading is ready.")
        else:
            st.warning("Image text reading is not available yet.")

        st.divider()
        st.subheader("AI answer writing")
        llm_ready = pipeline.check_llm_status()
        llm_model = pipeline.get_llm_model_name()
        if llm_ready and llm_model:
            st.success(f"AI answer writing is ready ({llm_model}).")
        else:
            st.warning("AI answer writing is offline.")
        
        # Overall Stats
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed passages", pipeline.document_count())
        with col2:
            indexed_files = pipeline.list_indexed_files()
            st.metric("Files", len(indexed_files))

        usage = pipeline.get_usage_metrics()
        st.caption(
            f"Searches: {int(usage['queries'])} | Avg speed: {usage['average_latency_ms']} ms | Cache hit rate: {usage['cache_hit_rate']}%"
        )
    
    # TAB 2: File Management
    with tab2:
        st.subheader("Files in your library")
        
        indexed_files = pipeline.list_indexed_files()
        
        if not indexed_files:
            st.info("No files yet.")
        else:
            for file_name, metadata in indexed_files.items():
                col1, col2 = st.columns([3, 1], gap="small")
                
                with col1:
                    chunk_count = metadata.get("chunk_count", 0)
                    file_hash = metadata.get("file_hash", "unknown")[:8]
                    st.write(f"{file_name} ({chunk_count} passages)")
                    st.caption(f"ID: {file_hash}...")
                
                with col2:
                    if st.button("Remove", key=f"delete_{file_name}", help="Remove this file"):
                        with st.spinner(f"Deleting {file_name}..."):
                            deleted = pipeline.delete_indexed_file(file_name)
                            removed_from_disk = remove_uploaded_file(file_name)
                            st.success(f"Removed {deleted} passages")
                            if removed_from_disk:
                                st.caption("Uploaded file was also removed from disk.")
                            st.rerun()


# Main Content Area
col_search, col_settings = st.columns([3, 1], gap="large")

with col_search:
    query = st.text_input(
        "Ask a question",
        placeholder="What would you like to know?",
    )

with col_settings:
    quality = st.selectbox(
        "Search quality",
        options=list(SEARCH_QUALITY_PRESETS.keys()),
        index=1,
    )
    use_llm = st.checkbox("Use AI to write the answer", value=True)

with st.expander("Advanced search options"):
    use_query_expansion = st.checkbox("Try alternate wordings automatically", value=True)
    selected_types = st.multiselect(
        "Search only in these file types",
        options=SUPPORTED_DOC_TYPES,
        default=SUPPORTED_DOC_TYPES,
    )

preset = SEARCH_QUALITY_PRESETS[quality]
top_k = int(preset["top_k"])
if not use_query_expansion:
    use_query_expansion = bool(preset["use_query_expansion"])

if st.button("🚀 Search", type="primary", use_container_width=True):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        spinner_text = "Generating answer..." if use_llm else "Searching retrieved context..."
        with st.spinner(spinner_text):
            response = pipeline.ask(
                query.strip(),
                top_k=top_k,
                use_llm=use_llm,
                filter_types=selected_types,
                use_query_expansion=use_query_expansion,
            )

            speed_text = f"{response['latency_ms']} ms"
            if response.get("cached"):
                st.caption(f"Fast answer (from cache) - {speed_text}")
            else:
                st.caption(f"Response time: {speed_text}")
            
            st.divider()
            st.subheader("Answer")
            st.write(response["answer"])

            answer_export = (
                f"Question: {query.strip()}\n\n"
                f"{response['answer']}\n\n"
                "Sources:\n" + "\n".join(f"- {item}" for item in response["sources"])
            )
            st.download_button(
                "Download answer",
                data=answer_export,
                file_name="answer.txt",
                mime="text/plain",
                use_container_width=True,
            )
            
            st.divider()
            st.subheader("Sources")
            
            if response["sources"]:
                for index, source in enumerate(response["sources"], start=1):
                    with st.expander(f"Source {index}"):
                        st.write(source)
            else:
                st.info("No sources found for this query.")
