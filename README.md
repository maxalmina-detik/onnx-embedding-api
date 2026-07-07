# LlamaIndex ONNX Embeddings Service (Railway-ready)

A minimal FastAPI service that wraps LlamaIndex's `HuggingFaceEmbedding`,
configured to use the sentence-transformers **ONNX backend** (Optimum + ONNX
Runtime) — turning this into a callable HTTP API instead of an in-notebook
object:

```python
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
```

## ⚠️ Railway has no GPU support

As of this writing, **Railway does not offer GPU-backed services** — so
`device="cuda"` / `provider="CUDAExecutionProvider"` from your original
snippet won't run there. This service defaults to `DEVICE=cpu` and
`ONNX_PROVIDER=CPUExecutionProvider` instead, which works fine for a
300M-parameter model at moderate request volume. Every setting is still
env-driven, so if you later deploy this same code on a GPU host (a VM,
RunPod, Fly.io GPU, Northflank BYOC, etc.) you just flip `DEVICE=cuda`,
`ONNX_PROVIDER=CUDAExecutionProvider`, and swap `onnxruntime` for
`onnxruntime-gpu` in `requirements.txt` — no code changes needed.

## Endpoints

- `GET /health` — check the service is up and see the active model/backend/device
- `POST /embed/query` — embed a single query string. **`query_instruction` is
  applied automatically** (this uses `get_query_embedding`, matching how your
  original `embed_model` distinguishes queries from documents).
  ```json
  { "text": "berapa harga bbm hari ini" }
  ```
- `POST /embed/documents` — embed a batch of chunks (`get_text_embedding_batch`,
  respects `EMBED_BATCH_SIZE` internally).
  ```json
  { "texts": ["chunk 1 ...", "chunk 2 ..."] }
  ```

## 1. Local development

```bash
cd railway-llamaindex-embeddings
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # defaults already match your original snippet (minus device/provider)
uvicorn main:app --reload --port 8000
```

First request will download the model + ONNX weights from the Hub — expect a
slower first call, then fast responses after.

Test it:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "contoh teks berita dalam bahasa Indonesia"}'

curl -X POST http://localhost:8000/embed/documents \
  -H "Content-Type: application/json" \
  -d '{"texts": ["chunk pertama", "chunk kedua"]}'
```

## 2. Deploy to Railway

**Option A — Railway CLI**
```bash
npm install -g @railway/cli
railway login
railway init      # or `railway link` to an existing project
railway up
```

**Option B — GitHub deploy**
1. Push this directory to a GitHub repo.
2. Railway dashboard → New Project → Deploy from GitHub repo.
3. Railway auto-detects Python via Nixpacks and uses `railway.json` /
   `Procfile` for the start command.

**Environment variables** (Railway dashboard → your service → Variables, or
via CLI) — the defaults in `.env.example` already match your snippet aside
from device/provider:
```bash
railway variables --set "MODEL_NAME=onnx-community/embeddinggemma-300m-ONNX" \
                   --set "BACKEND=onnx" \
                   --set "DEVICE=cpu" \
                   --set "ONNX_FILE_NAME=model_quantized.onnx" \
                   --set "ONNX_PROVIDER=CPUExecutionProvider" \
                   --set "EMBED_BATCH_SIZE=8" \
                   --set "QUERY_INSTRUCTION=task: search result | query: "
```

Railway provides `$PORT` automatically; the start command in `railway.json`
and `Procfile` already binds to it. The healthcheck timeout is set to 180s to
allow for the model download on first boot.

**Resource sizing:** embeddinggemma-300m is small, but this container still
loads torch + onnxruntime + the model and runs real inference — give it at
least 2 vCPU / 2GB RAM on Railway and load-test from there.

## 3. Calling it from other code

```python
import requests

BASE_URL = "https://<your-app>.up.railway.app"

def embed_query(text: str) -> list[float]:
    r = requests.post(f"{BASE_URL}/embed/query", json={"text": text})
    r.raise_for_status()
    return r.json()["embedding"]

def embed_documents(texts: list[str]) -> list[list[float]]:
    r = requests.post(f"{BASE_URL}/embed/documents", json={"texts": texts})
    r.raise_for_status()
    return r.json()["embeddings"]
```

This works as a drop-in remote embedding source for any RAG pipeline (e.g.
feeding vectors into Elasticsearch/Qdrant), without needing LlamaIndex,
sentence-transformers, or the ONNX runtime installed in the calling service.

## Running with a GPU (elsewhere)

If you move this off Railway to a GPU host later:

```bash
DEVICE=cuda
ONNX_PROVIDER=CUDAExecutionProvider
```
and in `requirements.txt`, replace:
```
optimum[onnxruntime]==1.23.3
onnxruntime==1.20.1
```
with:
```
optimum[onnxruntime-gpu]==1.23.3
onnxruntime-gpu==1.20.1
```
(and use a CUDA-enabled base image/torch build appropriate to the host's
driver version — the CPU-only `--extra-index-url` line in `requirements.txt`
should be removed in that case).

## Notes

- `text_instruction` is supported alongside `query_instruction` if your model
  expects a different prefix for indexed documents vs. queries — set
  `TEXT_INSTRUCTION` if needed (left unset by default, matching your snippet).
- `onnx-community/*` models ship pre-exported ONNX weights, so there's no
  on-the-fly conversion step here — sentence-transformers just downloads and
  loads `model_quantized.onnx` directly, keeping cold starts predictable.