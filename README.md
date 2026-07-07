# Embeddings Service (Railway-ready)

A minimal FastAPI service exposing LangChain-compatible embeddings, with two
interchangeable backends selected via `EMBEDDINGS_BACKEND`:

- **`endpoint`** (default) — wraps LangChain's `HuggingFaceEndpointEmbeddings`,
  calling a remote HF Inference Endpoint (or the public HF Inference API).
  No model weights loaded locally — lightweight, fast-booting container.
- **`onnx`** — runs an ONNX embedding model locally on CPU via Hugging Face
  Optimum + ONNX Runtime. No network round-trip per request, no HF API rate
  limits/cost, but a heavier container and more CPU/RAM needed.

Both backends implement the same `embed_query` / `embed_documents` interface,
so the API surface below is identical regardless of which one is active.

## Endpoints

- `GET /health` — check the service is up and see which model/endpoint it's using
- `POST /embed/query` — embed a single string, e.g. a search query
  ```json
  { "text": "berapa harga bbm hari ini" }
  ```
- `POST /embed/documents` — embed a batch of strings, e.g. chunked article text
  ```json
  { "texts": ["chunk 1 ...", "chunk 2 ..."] }
  ```

## 1. Get a Hugging Face embedding endpoint

Two options:

**A. Dedicated Inference Endpoint (recommended for production)**
1. Go to https://ui.endpoints.huggingface.co/ and deploy an embedding model
   (e.g. `BAAI/bge-m3` for strong multilingual/Indonesian coverage, or
   `intfloat/multilingual-e5-large`).
2. Copy the endpoint URL — you'll set this as `HF_EMBEDDINGS_ENDPOINT_URL`.

**B. Public HF Inference API (fine for prototyping)**
- Just set `HF_MODEL_ID` (default `BAAI/bge-m3`) and leave
  `HF_EMBEDDINGS_ENDPOINT_URL` unset. Subject to public rate limits and cold
  starts — not meant for production traffic.

Either way, grab an HF access token from https://huggingface.co/settings/tokens.

## 2. Choosing a backend

### Backend: `endpoint` (default) — call HF remotely
Set `EMBEDDINGS_BACKEND=endpoint` (or leave unset) and follow "1. Get a
Hugging Face embedding endpoint" above.

### Backend: `onnx` — run the model locally in this container
Set `EMBEDDINGS_BACKEND=onnx`. Two ways to point at a model:

**A. Auto-export a normal HF checkpoint (simplest, slower cold start)**
```bash
EMBEDDINGS_BACKEND=onnx
ONNX_MODEL_ID=BAAI/bge-m3
ONNX_EXPORT=true
```
On first load, Optimum downloads the regular (non-ONNX) checkpoint and
converts it to ONNX in-process. This works with any HF model but adds
roughly 30-90s to cold start (model + framework dependent), and re-runs on
every fresh container start unless you're using a persistent volume.

**B. Use a pre-exported ONNX model (fast, predictable cold start)**
```bash
EMBEDDINGS_BACKEND=onnx
ONNX_MODEL_ID=some-org/some-model     # a repo that already ships onnx/ weights
ONNX_EXPORT=false
ONNX_FILE_NAME=onnx/model.onnx        # or e.g. onnx/model_quantized.onnx
```
Many popular embedding models have ONNX-exported mirrors (e.g. under the
`Xenova/*` or `onnx-community/*` orgs on the HF Hub), or you can export your
own once with the [`optimum-cli export onnx`](https://huggingface.co/docs/optimum/exporters/onnx/usage_guides/export_a_model)
command and push the result to a private HF repo. This is the recommended
path for production — no on-the-fly conversion, faster and more predictable
boots.

**Resource note:** the ONNX backend loads `torch` + `onnxruntime` and runs
real inference in this container, so give it a Railway plan with meaningfully
more CPU/RAM than the `endpoint` backend needs (which is just a thin HTTP
proxy). Start with at least 2 vCPU / 2GB RAM and load-test from there —
actual needs depend on the model size and your request volume. For GPU
(`ONNX_DEVICE=cuda`) you need a GPU-enabled Railway runtime and
`onnxruntime-gpu` instead of `onnxruntime` in requirements.txt (see the
comment there — the two packages conflict and only one can be installed).

### Direct translation from a LlamaIndex config

If you have a LlamaIndex `HuggingFaceEmbedding(backend="onnx", ...)` config,
here's the field-by-field equivalent using this repo's `ONNXEmbeddings` class:

```python
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

# LangChain (this repo's onnx_embeddings.py)
from onnx_embeddings import ONNXEmbeddings

embeddings = ONNXEmbeddings(
    model_id="onnx-community/embeddinggemma-300m-ONNX",
    device="cuda",                       # or provider="CUDAExecutionProvider" directly
    onnx_file_name="model_quantized.onnx",
    export=False,                        # this repo already ships ONNX weights
    batch_size=8,
    query_instruction="task: search result | query: ",
)

vector = embeddings.embed_query("contoh berita ekonomi terbaru")
vectors = embeddings.embed_documents(["chunk 1 ...", "chunk 2 ..."])
```

Or, to run this service configured that way, set in `.env` / Railway variables:
```bash
EMBEDDINGS_BACKEND=onnx
ONNX_MODEL_ID=onnx-community/embeddinggemma-300m-ONNX
ONNX_EXPORT=false
ONNX_FILE_NAME=model_quantized.onnx
ONNX_DEVICE=cuda
ONNX_BATCH_SIZE=8
ONNX_QUERY_INSTRUCTION=task: search result | query: 
```

Notes on the translation:
- LlamaIndex's `model_kwargs={"file_name": ..., "provider": ...}` maps to
  this class's top-level `onnx_file_name` / `provider` (or `device`) params.
- LlamaIndex's `query_instruction` only prefixes queries by default; this
  class mirrors that with `query_instruction` for `embed_query` and a
  separate `text_instruction` for `embed_documents` if the document side also
  needs a prefix (EmbeddingGemma's own convention is `"title: none | text: "`
  for documents — set `ONNX_TEXT_INSTRUCTION` to that if you want parity).
- `embed_batch_size` → `batch_size` (only affects `embed_documents`;
  `embed_query` always embeds a single string).

## 3. Local development

```bash
# both backends live in this same env; only the relevant vars need real values
cd railway-hf-embeddings
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in your token / endpoint
uvicorn main:app --reload --port 8000
```

Test it:
```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/embed/query \
  -H "Content-Type: application/json" \
  -d '{"text": "contoh teks berita dalam bahasa Indonesia"}'
```

## 4. Deploy to Railway

**Option A — Railway CLI**
```bash
npm install -g @railway/cli   # if you don't have it
railway login
railway init                 # creates a new Railway project (or `railway link` to an existing one)
railway up                    # builds and deploys this directory
```

**Option B — GitHub deploy**
1. Push this directory to a GitHub repo.
2. In the Railway dashboard: New Project → Deploy from GitHub repo → select it.
3. Railway auto-detects Python via Nixpacks and uses `railway.json` /
   `Procfile` for the start command.

**Set environment variables** (Railway dashboard → your service → Variables,
or via CLI):
```bash
# backend=endpoint, dedicated Inference Endpoint
railway variables --set "EMBEDDINGS_BACKEND=endpoint" \
                   --set "HUGGINGFACEHUB_API_TOKEN=hf_xxx" \
                   --set "HF_EMBEDDINGS_ENDPOINT_URL=https://xxxx.endpoints.huggingface.cloud"

# backend=endpoint, public HF Inference API
railway variables --set "EMBEDDINGS_BACKEND=endpoint" \
                   --set "HUGGINGFACEHUB_API_TOKEN=hf_xxx" \
                   --set "HF_MODEL_ID=BAAI/bge-m3"

# backend=onnx, auto-export on load
railway variables --set "EMBEDDINGS_BACKEND=onnx" \
                   --set "ONNX_MODEL_ID=BAAI/bge-m3" \
                   --set "ONNX_EXPORT=true"
```

Railway automatically provides `$PORT`; the start command in `railway.json`
and `Procfile` already binds to it. If using the `onnx` backend, remember to
size the service (vCPU/RAM) for local inference, not just a thin proxy.

## 5. Calling it from another LangChain app

Once deployed, treat this service as a normal HTTP embeddings API from any
client (Python, your RAG pipeline, etc.):

```python
import requests

def embed_documents(texts, base_url="https://<your-app>.up.railway.app"):
    resp = requests.post(f"{base_url}/embed/documents", json={"texts": texts})
    resp.raise_for_status()
    return resp.json()["embeddings"]
```

If instead you want to use `HuggingFaceEndpointEmbeddings` *directly* inside
another LangChain app (skipping this service entirely and calling HF directly),
it's just:

```python
from langchain_huggingface import HuggingFaceEndpointEmbeddings

embeddings = HuggingFaceEndpointEmbeddings(
    model="https://xxxx.endpoints.huggingface.cloud",  # or a model id for the public API
    huggingfacehub_api_token="hf_xxx",
)

vector = embeddings.embed_query("contoh teks")
vectors = embeddings.embed_documents(["chunk 1", "chunk 2"])
```

This repo just wraps that in a small service so other systems (e.g. your
Elasticsearch ingestion pipeline or agent router) can call it over HTTP
without needing the HF token or LangChain installed locally.

## Notes

- **Backend `endpoint`:** no GPU or model weights run in this container —
  it's a thin proxy to HF's infra, so Railway's smallest plan is plenty. For
  production RAG at scale, prefer a **dedicated** HF Inference Endpoint over
  the public Inference API, which is rate-limited and not meant for sustained
  throughput.
- **Backend `onnx`:** inference runs in-container on CPU. Prefer a
  pre-exported ONNX model (`ONNX_EXPORT=false`) for fast, predictable cold
  starts in production; use `ONNX_EXPORT=true` for quick prototyping with any
  HF model. Give the service real CPU/RAM and load-test before committing to
  a Railway plan size.
- `BAAI/bge-m3` and `intfloat/multilingual-e5-large` are both solid choices
  for Indonesian-language content, in either backend.
