"""
FastAPI service exposing LlamaIndex's HuggingFaceEmbedding, configured to use
the sentence-transformers ONNX backend (via Optimum + ONNX Runtime).

Based on:

    embed_model = HuggingFaceEmbedding(
        model_name="onnx-community/embeddinggemma-300m-ONNX",
        device="cuda",
        backend="onnx",
        embed_batch_size=8,
        model_kwargs={
            "file_name": "model_quantized.onnx",
            "provider": "CUDAExecutionProvider",
        },
        query_instruction="task: search result | query: ",
    )

IMPORTANT — Railway + GPU: Railway does not currently offer GPU-backed
services, so this deploys with device="cpu" and provider="CPUExecutionProvider"
by default. All of the above stays fully configurable via env vars, so the
exact same code runs unchanged with device="cuda" / CUDAExecutionProvider if
you deploy this on a GPU host instead (see .env.example).

Since backend="onnx" is handled internally by sentence-transformers/Optimum,
this file has almost no custom ONNX code — LlamaIndex + sentence-transformers
does the tokenization, inference, and pooling.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("llamaindex-embeddings-service")

# ---------------------------------------------------------------------------
# Config (all overridable via env vars — see .env.example)
# ---------------------------------------------------------------------------
MODEL_NAME = os.getenv("MODEL_NAME", "onnx-community/embeddinggemma-300m-ONNX")
DEVICE = os.getenv("DEVICE", "cpu")  # "cpu" on Railway; "cuda" if deployed on a GPU host
BACKEND = os.getenv("BACKEND", "onnx")
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))

ONNX_FILE_NAME = os.getenv("ONNX_FILE_NAME", "model_quantized.onnx")
# CPUExecutionProvider by default (Railway has no GPU). Set to
# CUDAExecutionProvider only when DEVICE=cuda on a GPU-enabled host.
ONNX_PROVIDER = os.getenv("ONNX_PROVIDER", "CPUExecutionProvider")

QUERY_INSTRUCTION = os.getenv("QUERY_INSTRUCTION", "task: search result | query: ")
# Some models (embeddinggemma included) also support/expect a distinct
# instruction prefix for documents being indexed. Leave unset to use the
# model's default.
TEXT_INSTRUCTION = os.getenv("TEXT_INSTRUCTION") or None

if DEVICE == "cuda" and ONNX_PROVIDER == "CPUExecutionProvider":
    logger.warning(
        "DEVICE=cuda but ONNX_PROVIDER=CPUExecutionProvider — set "
        "ONNX_PROVIDER=CUDAExecutionProvider to actually use the GPU."
    )

# Embedding model is created once at startup and reused across requests.
_embed_model: Optional[HuggingFaceEmbedding] = None


def build_embed_model() -> HuggingFaceEmbedding:
    model_kwargs = {"file_name": ONNX_FILE_NAME, "provider": ONNX_PROVIDER}

    kwargs = dict(
        model_name=MODEL_NAME,
        device=DEVICE,
        backend=BACKEND,
        embed_batch_size=EMBED_BATCH_SIZE,
        model_kwargs=model_kwargs,
        query_instruction=QUERY_INSTRUCTION,
    )
    if TEXT_INSTRUCTION:
        kwargs["text_instruction"] = TEXT_INSTRUCTION

    logger.info(
        "Loading HuggingFaceEmbedding: model=%s device=%s backend=%s "
        "onnx_file=%s provider=%s",
        MODEL_NAME, DEVICE, BACKEND, ONNX_FILE_NAME, ONNX_PROVIDER,
    )
    return HuggingFaceEmbedding(**kwargs)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model
    _embed_model = build_embed_model()
    yield
    _embed_model = None


app = FastAPI(
    title="LlamaIndex ONNX Embeddings Service",
    description="LlamaIndex HuggingFaceEmbedding (ONNX backend) wrapped in FastAPI, deployable on Railway.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class EmbedQueryRequest(BaseModel):
    text: str = Field(..., description="Single query string (query_instruction is applied automatically).")


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
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "device": DEVICE,
        "backend": BACKEND,
        "onnx_file_name": ONNX_FILE_NAME,
        "onnx_provider": ONNX_PROVIDER,
    }


@app.post("/embed/query", response_model=EmbeddingResponse)
def embed_query(payload: EmbedQueryRequest):
    """Embed a single query string. LlamaIndex applies query_instruction
    automatically here (unlike get_text_embedding, which does not)."""
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    try:
        vector = _embed_model.get_query_embedding(payload.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_query_embedding failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    return EmbeddingResponse(embedding=vector, dimensions=len(vector))


@app.post("/embed/documents", response_model=EmbeddingsResponse)
def embed_documents(payload: EmbedDocumentsRequest):
    """Embed a batch of documents/chunks (uses get_text_embedding_batch,
    respecting embed_batch_size internally)."""
    if _embed_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not initialized")
    if not payload.texts:
        raise HTTPException(status_code=400, detail="texts must be a non-empty list")

    try:
        vectors = _embed_model.get_text_embedding_batch(payload.texts)
    except Exception as exc:  # noqa: BLE001
        logger.exception("get_text_embedding_batch failed")
        raise HTTPException(status_code=502, detail=f"Embedding request failed: {exc}") from exc

    dims = len(vectors[0]) if vectors else 0
    return EmbeddingsResponse(embeddings=vectors, dimensions=dims, count=len(vectors))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)