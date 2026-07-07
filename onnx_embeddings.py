"""
A LangChain-compatible Embeddings class that runs an ONNX embedding model
locally via Hugging Face Optimum + ONNX Runtime, instead of calling a remote
HF Inference Endpoint.

This mirrors the shape of LlamaIndex's `HuggingFaceEmbedding(..., backend="onnx")`
config, e.g.:

    # LlamaIndex
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

    # LangChain equivalent (this class)
    embeddings = ONNXEmbeddings(
        model_id="onnx-community/embeddinggemma-300m-ONNX",
        device="cuda",
        onnx_file_name="model_quantized.onnx",
        batch_size=8,
        query_instruction="task: search result | query: ",
    )

Notes:
  - `device` is a convenience that maps to an ONNX Runtime execution provider
    ("cuda" -> CUDAExecutionProvider, "cpu" -> CPUExecutionProvider). Pass
    `provider=` directly if you need a specific provider (e.g. TensorrtExecutionProvider).
  - GPU inference (CUDAExecutionProvider) requires the `onnxruntime-gpu`
    package instead of plain `onnxruntime` — see requirements.txt.
  - `query_instruction` / `text_instruction` prepend a task prefix before
    embedding, matching how asymmetric-retrieval models like EmbeddingGemma
    and BGE are meant to be used (different prefix for queries vs. documents).
"""

import logging
from typing import List, Optional

import numpy as np
from langchain_core.embeddings import Embeddings
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

logger = logging.getLogger("onnx-embeddings")

_DEVICE_TO_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
}


class ONNXEmbeddings(Embeddings):
    """Mean-pooled sentence embeddings from a local ONNX model.

    Implements the same embed_query / embed_documents interface as
    LangChain's other Embeddings classes (e.g. HuggingFaceEndpointEmbeddings),
    so it's a drop-in swap in this service or any other LangChain code.
    """

    def __init__(
        self,
        model_id: str,
        onnx_file_name: Optional[str] = None,
        export: bool = False,
        device: str = "cpu",
        provider: Optional[str] = None,
        max_length: int = 512,
        normalize: bool = True,
        batch_size: int = 32,
        query_instruction: str = "",
        text_instruction: str = "",
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.normalize = normalize
        self.batch_size = batch_size
        self.query_instruction = query_instruction
        self.text_instruction = text_instruction

        resolved_provider = provider or _DEVICE_TO_PROVIDER.get(
            device.lower(), "CPUExecutionProvider"
        )

        logger.info(
            "Loading ONNX embeddings model=%s export=%s onnx_file_name=%s provider=%s",
            model_id, export, onnx_file_name, resolved_provider,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs = {}
        if onnx_file_name:
            load_kwargs["file_name"] = onnx_file_name

        # export=True converts a regular HF checkpoint to ONNX on the fly
        # (and caches it under the HF cache dir for the life of the container).
        # export=False (default) expects the repo/path to already contain ONNX
        # weights, e.g. onnx-community/embeddinggemma-300m-ONNX.
        self.model = ORTModelForFeatureExtraction.from_pretrained(
            model_id,
            export=export,
            provider=resolved_provider,
            **load_kwargs,
        )

        logger.info("ONNX embeddings model ready: %s (provider=%s)", model_id, resolved_provider)

    @staticmethod
    def _mean_pool(last_hidden_state: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
        mask = attention_mask[..., None].astype(np.float32)
        summed = (last_hidden_state * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
        return summed / counts

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        outputs = self.model(**inputs)

        last_hidden = outputs.last_hidden_state.detach().cpu().numpy()
        attention_mask = inputs["attention_mask"].detach().cpu().numpy()

        pooled = self._mean_pool(last_hidden, attention_mask)

        if self.normalize:
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            norms[norms == 0] = 1e-9
            pooled = pooled / norms

        return pooled.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of documents/chunks, in batch_size-sized groups.

        Applies `text_instruction` as a prefix to each text, mirroring
        LlamaIndex's document-side instruction handling.
        """
        if not texts:
            return []

        prefixed = [f"{self.text_instruction}{t}" for t in texts]

        all_vectors: List[List[float]] = []
        for i in range(0, len(prefixed), self.batch_size):
            chunk = prefixed[i : i + self.batch_size]
            all_vectors.extend(self._embed_batch(chunk))
        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string, prefixed with `query_instruction`."""
        return self._embed_batch([f"{self.query_instruction}{text}"])[0]
