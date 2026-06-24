# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`sanjaya` (package name: `dynamic-reading-lists`) generates richly annotated reading environments for historical texts — currently Ancient Greek. It uses a locally-hosted LLM (LLaMA.cpp-compatible server) to produce word-level glosses and sentence-level translations, then renders them into interactive HTML pages.

## Environment setup

Managed with [uv](https://github.com/astral-uv/uv). Python 3.12 required.

```bash
uv sync          # install dependencies into .venv
source .venv/bin/activate
```

For development, install the package in editable mode to avoid the hard-coded path in `generator.py`:

```bash
uv pip install -e .
```

The external dependency `perseus-cts` is installed directly from GitHub:
```
perseus-cts @ git+https://github.com/PerseusDLCode/perseus-cts.git
```

## Running the pipeline

`run.py` is the reference entry point. It requires a running LLM server and TEI source data in `test-data/`:

```bash
python run.py
```

Use `chunk_filter` on `Generator` to limit processing to a subset of chunks during development (e.g. `chunk_filter="1.1"` matches chunks whose stem starts with `"1.1"`).

There are currently no automated tests.

## Architecture

### Data flow

```
TEI XML files (test-data/)
  → perseus_cts.Corpus / Chunker   # parse & chunk the source text
  → Chunker.compile(output_dir)    # writes per-chunk XML files
  → Generator._get_all()           # reads chunked XMLs via lxml
  → Annotator.annotate_and_save()  # calls local LLM, caches JSON
  → Generator.write_html()         # renders Jinja2 templates to HTML
```

### Key modules

**`src/llm/`** — LLM annotation layer

- `annotations.py` — `Annotator` base class + `call_model()` (HTTP POST to `localhost:{port}/v1/chat/completions`). Handles save/load of annotation JSON.
- `annotators.py` — `GlossAnnotator` (word-by-word, uses NLTK tokenization) and `TranslationAnnotator` (sentence-level). Both extend `Annotator` and override `annotate()`.
- `prompts.py` — `string.Template`-based prompt builders for gloss and translation tasks, returned as OpenAI-style `messages` lists.
- `validator.py` — extracts JSON from LLM responses via regex, repairs malformed JSON with `json_repair`, and optionally validates against a JSON schema.

**`src/site/generator.py`** — `Generator` class orchestrates the full pipeline: parse XML → annotate (with caching to avoid re-calling the LLM) → collate gloss/translation pairs by sentence → render HTML.

**`src/templates/`** — Jinja2 templates. `chunk-page.html.jinja` extends `base.html.jinja` and renders each sentence with inline gloss spans (`data-gloss`, `data-form`) plus translation and collapsible notes. The template branches on annotator `role` strings (`"gloss"`, `"translation"`); unknown roles render as collapsible raw JSON.

### LLM server

All annotation calls go to a local LLaMA.cpp-compatible server. Instantiate annotators with the port it is listening on:

```python
GlossAnnotator(port=8080, language="Ancient Greek", author="Thucydides", work="...")
TranslationAnnotator(port=8080, language="Ancient Greek", author="Thucydides", work="...")
```

The `language` parameter is interpolated directly into prompt templates, so use the human-readable name (e.g. `"Ancient Greek"`) rather than a language code.

Annotation results are cached to JSON files; `Generator._create_annotations()` skips files that already exist.

### Annotator role system

Each `Annotator` subclass declares a unique `role` string (e.g. `"gloss"`, `"translation"`). `Generator` validates uniqueness at construction time. The role controls both the cache directory name and which rendering branch `chunk-page.html.jinja` uses.

### TEI namespace

All XPath queries use the TEI namespace `http://www.tei-c.org/ns/1.0` aliased as `tei`.

## Known issues / TODOs

- `src/site/generator.py` line 9 hard-codes an absolute path (`HERE = Path("/Users/pnadel01/Desktop/dynamic-reading-lists")`). Fix by installing in editable mode (`uv pip install -e .`).
- `TranslationAnnotator` has a `# @TODO come back for drama` note regarding speaker handling for dramatic texts.
- `create_annotation_prompt()` in `prompts.py` is a stub (`pass`).
