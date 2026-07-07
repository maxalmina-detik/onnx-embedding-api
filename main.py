"""
FastAPI service exposing LangChain-compatible embeddings, with two
interchangeable backends:

  1. "endpoint" (default) — LangChain's HuggingFaceEndpointEmbeddings, calling
     out to a remote HF Inference Endpoint or the public HF Inference API.
     No model weights loaded locally; lightweight container.

  2. "onnx" — a local ONNX model (via Hugging Face Optimum + ONNX Runtime),
     run inside this container on CPU. No network round-trip per request,
     no HF rate limits, but a heavier container (torch + onnxruntime) and
     more CPU/RAM needed on Railway.

Select the backend with EMBEDDINGS_BACKEND=endpoint|onnx.

Env vars (see .env.example for the full list):
  EMBEDDINGS_BACKEND          - "endpoint" (default) or "onnx"

  # backend=endpoint
  HUGGINGFACEHUB_API_TOKEN    - your HF token (needed for private/dedicated
                                 endpoints and for higher rate limits on the
                                 public API)
  HF_EMBEDDINGS_ENDPOINT_URL  - full URL of your HF Inference Endpoint
                                 (leave unset to fall back to the public HF
                                 Inference API using HF_MODEL_ID)
  HF_MODEL_ID                 - model repo id, used only when
                                 HF_EMBEDDINGS_ENDPOINT_URL is not set
                                 (default: BAAI/bge-m3)

  # backend=onnx
  ONNX_MODEL_ID                - HF model id or local path to load
                                  (default: BAAI/bge-m3)
  ONNX_EXPORT                  - "true" to convert a regular HF checkpoint to
                                  ONNX on the fly; "false" (default) if
                                  ONNX_MODEL_ID already ships ONNX weights
                                  (e.g. onnx-community/embeddinggemma-300m-ONNX)
  ONNX_FILE_NAME                - specific .onnx file to load, e.g.
                                  "model_quantized.onnx"
  ONNX_DEVICE                   - "cpu" (default) or "cuda"; maps to an ONNX
                                  Runtime execution provider unless
                                  ONNX_PROVIDER is set explicitly
  ONNX_PROVIDER                 - explicit ONNX Runtime provider, e.g.
                                  "CUDAExecutionProvider" (overrides ONNX_DEVICE)
  ONNX_MAX_SEQ_LENGTH           - max tokens per input (default: 512)
  ONNX_NORMALIZE                - "true" (default) to L2-normalize embeddings
  ONNX_BATCH_SIZE                - batch size for embed_documents (default: 32)
  ONNX_QUERY_INSTRUCTION         - prefix applied before embedding a query,
                                  e.g. "task: search result | query: "
  ONNX_TEXT_INSTRUCTION          - prefix applied before embedding documents
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.embeddings import Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("embeddings-service")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDINGS_BACKEND = os.getenv("EMBEDDINGS_BACKEND", "endpoint").strip().lower()

# backend=endpoint
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
HF_ENDPOINT_URL = os.getenv("HF_EMBEDDINGS_ENDPOINT_URL")  # dedicated Inference Endpoint
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "BAAI/bge-m3")      # used if no dedicated endpoint

# backend=onnx
ONNX_MODEL_ID = os.getenv("ONNX_MODEL_ID", "BAAI/bge-m3")
ONNX_EXPORT = os.getenv("ONNX_EXPORT", "false").strip().lower() == "true"
ONNX_FILE_NAME = os.getenv("ONNX_FILE_NAME") or None
ONNX_DEVICE = os.getenv("ONNX_DEVICE", "cpu").strip().lower()
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER") or None  # overrides ONNX_DEVICE if set
ONNX_MAX_SEQ_LENGTH = int(os.getenv("ONNX_MAX_SEQ_LENGTH", "512"))
ONNX_NORMALIZE = os.getenv("ONNX_NORMALIZE", "true").strip().lower() == "true"
ONNX_BATCH_SIZE = int(os.getenv("ONNX_BATCH_SIZE", "32"))
ONNX_QUERY_INSTRUCTION = os.getenv("ONNX_QUERY_INSTRUCTION", "")
ONNX_TEXT_INSTRUCTION = os.getenv("ONNX_TEXT_INSTRUCTION", "")

if EMBEDDINGS_BACKEND not in ("endpoint", "onnx"):
    raise ValueError(
        f"Invalid EMBEDDINGS_BACKEND={EMBEDDINGS_BACKEND!r}; must be 'endpoint' or 'onnx'"
    )

if EMBEDDINGS_BACKEND == "endpoint" and not HF_TOKEN:
    logger.warning(
        "HUGGINGFACEHUB_API_TOKEN is not set. This will fail for dedicated "
        "endpoints and will be rate-limited on the public Inference API."
    )

# Embeddings client is created once at startup and reused across requests.
_embeddings: Optional[Embeddings] = None


def build_embeddings_client() -> Embeddings:
    """Build the active embeddings client based on EMBEDDINGS_BACKEND."""
    if EMBEDDINGS_BACKEND == "onnx":
        from onnx_embeddings import ONNXEmbeddings

        logger.info(
            "Using local ONNX backend: model=%s export=%s file_name=%s device=%s provider=%s",
            ONNX_MODEL_ID, ONNX_EXPORT, ONNX_FILE_NAME, ONNX_DEVICE, ONNX_PROVIDER,
        )
        return ONNXEmbeddings(
            model_id=ONNX_MODEL_ID,
            onnx_file_name=ONNX_FILE_NAME,
            export=ONNX_EXPORT,
            device=ONNX_DEVICE,
            provider=ONNX_PROVIDER,
            max_length=ONNX_MAX_SEQ_LENGTH,
            normalize=ONNX_NORMALIZE,
            batch_size=ONNX_BATCH_SIZE,
            query_instruction=ONNX_QUERY_INSTRUCTION,
            text_instruction=ONNX_TEXT_INSTRUCTION,
        )

    # backend == "endpoint"
    kwargs = {"huggingfacehub_api_token": HF_TOKEN}
    if HF_ENDPOINT_URL:
        kwargs["model"] = HF_ENDPOINT_URL
        logger.info("Using dedicated HF Inference Endpoint: %s", HF_ENDPOINT_URL)
    else:
        kwargs["model"] = HF_MODEL_ID
        logger.info("Using public HF Inference API for model: %s", HF_MODEL_ID)

    return HuggingFaceEndpointEmbeddings(**kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embeddings
    _embeddings = build_embeddings_client()
    yield
    _embeddings = None


app = FastAPI(
    title="Embeddings Service",
    description=(
        "LangChain-compatible embeddings over HTTP — either a remote "
        "HuggingFaceEndpointEmbeddings backend or a local ONNX backend — "
        "deployable on Railway."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single piece of text to embed (e.g. a search query).")


class EmbedDocumentsRequest(BaseModel):
    texts: List[str] = Field(..., description="Batch of documents/chunks to embed.")


class EmbeddingResponse(BaseModel):
    embedding: List[float]
    dimensions: int


class EmbeddingsResponse(BaseModel):
    embeddings: List[List[float]]
    dimensions: int
    count: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    if EMBEDDINGS_BACKEND == "onnx":
        return {
            "status": "ok",
            "backend": "onnx",
            "model": ONNX_MODEL_ID,
            "onnx_export_on_load": ONNX_EXPORT,
            "onnx_file_name": ONNX_FILE_NAME,
            "device": ONNX_DEVICE,
            "provider": ONNX_PROVIDER,
        }

    return {
        "status": "ok",
        "backend": "endpoint",
        "model": HF_ENDPOINT_URL or HF_MODEL_ID,
        "using_dedicated_endpoint": bool(HF_ENDPOINT_URL),
    }


@app.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(payload: EmbedQueryRequest):
    """Embed a single query string (uses embed_query, some models apply a
    query-specific prefix/instruction under the hood)."""
    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings client not initialized")
    try:
        vector = _embeddings.embed_query(payload.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_query failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingResponse(embedding=vector, dimensions=len(vector))


@app.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(payload: EmbedDocumentsRequest):
    """Embed a batch of documents/chunks (uses embed_documents)."""
    if _embeddings is None:
        raise HTTPException(status_code=503, detail="Embeddings client not initialized")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        vectors = _embeddings.embed_documents(payload.texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("embed_documents failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    dims = len(vectors[0]) if vectors else 0
    return EmbeddingsResponse(embeddings=vectors, dimensions=dims, count=len(vectors))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
