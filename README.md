# sanjaya

`sanjaya` ([named for the describer of the Mahabharata war](https://en.wikipedia.org/wiki/Sanjaya)) generates richly annotated reading environments for historical texts. It produces word-level and sentence-level annotations — via an LLM, a classical NLP pipeline, or any custom backend — and renders them into interactive HTML pages where readers can click any word to reveal its gloss. A vocabulary index collects every unique word form across the text, with links back to every sentence in which it appears.

## Requirements

- Python 3.12
- [uv](https://github.com/astral-uv/uv)
- A TEI XML source file
- An annotation backend (LLM endpoint, spaCy model, etc.) — see below

## Installation

```bash
git clone https://github.com/pnadelofficial/sanjaya.git
cd sanjaya
uv sync
uv pip install -e .
source .venv/bin/activate
```

## Usage

### 1. Write a config file

Copy `config.yaml` and edit it for your text and annotators:

```yaml
work:
  title: "History of the Peloponnesian War"
  author: "Thucydides"

source:
  file: "data/thucydides.xml"   # path to TEI XML, relative to this file
  xpath: ".//tei:p"             # XPath selecting the subunits to annotate

output:
  dir: "output/thucydides"

annotators:
  - class: sanjaya.llm.annotators.GlossAnnotator
    args:
      base_url: "http://localhost:8080"
      language: "Ancient Greek"
      author: "Thucydides"
      work: "History of the Peloponnesian War"

  - class: sanjaya.llm.annotators.TranslationAnnotator
    args:
      base_url: "http://localhost:8080"
      language: "Ancient Greek"
      author: "Thucydides"
      work: "History of the Peloponnesian War"
```

The `class` field accepts any dotted Python path, so you can point at your own annotators without modifying sanjaya itself.

### 2. Start your annotation backend (if using an LLM)

**Local server** (e.g. [llama.cpp](https://github.com/ggerganov/llama.cpp)):
```bash
llama-server --model your-model.gguf --port 8080
```

Any server exposing a `/v1/chat/completions` endpoint works — LLaMA.cpp, LM Studio, and Ollama all qualify.

**Hosted API** (OpenAI or any compatible provider): no server to start. Supply `model` and `api_key` under `args` in the config instead of `base_url`.

### 3. Run

```bash
sanjaya --config config.yaml
```

To process only specific chunks:
```bash
sanjaya --config config.yaml --chunk 1.1 2.1
```

Each chunk ID is a prefix match, so `1.1` also processes `1.1.1`, `1.1.2`, etc. `--chunk` overrides `chunk_filter` in the config file.

### 4. View the output

```bash
cd output/thucydides/html
python -m http.server
```

Then open `http://localhost:8000`.

## Output structure

```
output/<work>/html/
  index.html        ← chunk table of contents
  1.1.html          ← chunk pages with clickable glosses
  vocab/
    index.html      ← alphabetical vocabulary list
    <form>.html     ← one page per unique word form
```

## Features

**Clickable glosses** — click any word in a chunk page to reveal its gloss in a popup.

**Vocabulary index** — every unique word form gets its own page listing all collected glosses and every sentence in which it appears, with links back to the source chunk. Each token has a stable ID of the form `tk-[chunk]-[sentence]-[word]`.

**Highlight on navigation** — clicking an occurrence link from a vocab page highlights all instances of that word form on the destination chunk page.

## Caching

Annotation results are cached as JSON files under `output/<work>/annotations/`. Re-running the pipeline skips any chunk that has already been annotated.

## Extending

### Custom annotators

Sanjaya has two annotator base classes:

- **`WordAnnotator`** — processes a sentence and returns one `Annotation` per token. The annotation dict must include a `"label"` key (the gloss displayed in the UI). Override `tokenize()` for language-specific tokenisation.
- **`SentenceAnnotator`** — processes a sentence and returns a single `Annotation` (or `None` on failure). The annotation dict must include a `"summary"` key (the main text displayed in the UI).

Implement either base class in your own module, then reference it by dotted path in the config:

```python
# myproject/annotators.py
from sanjaya.llm.annotations import WordAnnotator, Annotation

class POSAnnotator(WordAnnotator):
    role = "pos"

    def annotate(self, sentence: str) -> list[Annotation]:
        tokens = self.tokenize(sentence)
        return [Annotation(text=t, annotation={"label": tag_token(t)}) for t in tokens]
```

```yaml
annotators:
  - class: myproject.annotators.POSAnnotator
    args: {}
```

All extra fields in the annotation dict beyond `label` / `summary` are rendered in a collapsible block on the chunk page.
