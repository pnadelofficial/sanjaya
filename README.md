# sanjaya

`sanjaya` generates richly annotated reading environments for historical texts — currently Ancient Greek. It calls an OpenAI-compatible LLM endpoint to produce word-level glosses and sentence-level translations, then renders them into interactive HTML pages where readers can click any word to reveal its gloss. A built-in vocabulary index collects every unique word form across the text, with links back to every sentence in which it appears.

## Requirements

- Python 3.12
- [uv](https://github.com/astral-uv/uv)
- An OpenAI-compatible LLM endpoint (local or hosted — see below)
- A TEI XML source file

## Installation

```bash
git clone https://github.com/pnadelofficial/sanjaya.git
cd sanjaya
uv sync
uv pip install -e .
source .venv/bin/activate
```

## Usage

### 1. Start an LLM endpoint

**Local server** (e.g. [llama.cpp](https://github.com/ggerganov/llama.cpp)):
```bash
llama-server --model your-model.gguf --port 8080
```

Any server exposing a `/v1/chat/completions` endpoint works — LLaMA.cpp, LM Studio, and Ollama all qualify.

**Hosted API** (OpenAI, or any OpenAI-compatible provider): no server to start; supply your `api_key` and `model` in the annotator config instead.

### 2. Configure and run the pipeline

Edit `run.py` to point at your source file and LLM endpoint, then run:

```bash
python run.py path/to/your/tei.xml
```

**Local server example** (default in `run.py`):
```python
GlossAnnotator(
    base_url="http://localhost:8080",
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
)
```

**Hosted API example**:
```python
GlossAnnotator(
    base_url="https://api.openai.com",
    model="gpt-4o",
    api_key="sk-...",
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
)
```

Use `chunk_filter` on `Generator` to process only a subset of chunks while iterating (e.g. `chunk_filter="1.1"`).

### 3. View the output

Serve the output directory and open it in a browser:

```bash
cd output/thucydides/html
python -m http.server
```

Then open `http://localhost:8000` in your browser.

## Output structure

```
output/<work>/html/
  index.html          ← chunk table of contents
  1.1.html            ← chunk pages with clickable glosses
  vocab/
    index.html        ← alphabetical vocabulary list
    <form>.html       ← one page per unique word form
  _pagefind/          ← search index (built automatically)
```

## Features

**Clickable glosses** — click any word in a chunk page to reveal its gloss in a popup. Click again to close.

**Vocabulary index** — every unique word form gets its own page listing all collected glosses and every sentence in which it appears, with links back to the source chunk. Each token has a stable ID of the form `tk-[chunk]-[sentence]-[word]`.

**Highlight on navigation** — clicking an occurrence link from a vocab page highlights all instances of that word form on the destination chunk page.

**Full-text search** — the vocabulary pages are indexed by [pagefind](https://pagefind.app) at build time. The search box at the top of every page covers both Greek word forms and their English glosses.

## Caching

Annotation results are cached as JSON files under `output/<work>/annotations/`. Re-running the pipeline skips any chunk that has already been annotated, so you only pay LLM costs once per chunk.

## Extending

To add a new annotation type, subclass `Annotator` in `src/llm/annotations.py`, set a unique `role` string, and implement `annotate()`. Pass the new annotator to `Generator` alongside the existing ones. The Jinja template (`src/templates/chunk-page.html.jinja`) will render unknown roles as collapsible raw JSON until you add a dedicated template branch.
