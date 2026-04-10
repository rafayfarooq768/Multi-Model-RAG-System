DEFAULT_COLLECTION_NAME = "multimodal_rag"
DEFAULT_CHROMA_PERSIST_DIR = "chroma_db"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CHUNK_SIZE = 700
DEFAULT_CHUNK_OVERLAP = 100
DEFAULT_TOP_K = 4

SUPPORTED_PDF_EXTENSIONS = {".pdf"}
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SUPPORTED_OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".odt"}
SUPPORTED_EXTENSIONS = (
    SUPPORTED_PDF_EXTENSIONS
    | SUPPORTED_TEXT_EXTENSIONS
    | SUPPORTED_IMAGE_EXTENSIONS
    | SUPPORTED_OFFICE_EXTENSIONS
)

LOG_LEVEL = "INFO"

# Phase 3: LLM configuration (OpenRouter)
DEFAULT_LLM_MODEL = "openai/gpt-4o-mini"
DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_LLM_TEMPERATURE = 0.7
DEFAULT_LLM_MAX_TOKENS = 512
MAX_LLM_CONTEXT_CHARS = 7000

# Phase 5: retrieval quality tuning
DEFAULT_RETRIEVAL_CANDIDATE_MULTIPLIER = 3
DEFAULT_HYBRID_SEMANTIC_WEIGHT = 0.7
DEFAULT_HYBRID_KEYWORD_WEIGHT = 0.3
DEFAULT_QUERY_EXPANSION_ENABLED = True
DEFAULT_QUERY_CACHE_SIZE = 100

SUPPORTED_DOC_TYPES = ["pdf", "text", "image", "docx", "pptx", "xlsx", "odt"]

SEARCH_QUALITY_PRESETS: dict[str, dict[str, object]] = {
    "Fast": {
        "top_k": 2,
        "use_query_expansion": False,
    },
    "Balanced": {
        "top_k": 4,
        "use_query_expansion": True,
    },
    "Thorough": {
        "top_k": 6,
        "use_query_expansion": True,
    },
}