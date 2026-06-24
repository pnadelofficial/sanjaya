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

The external dependency `perseus-cts` is installed directly from GitHub:
```
perseus-cts @ git+https://github.com/PerseusDLCode/perseus-cts.git
```

## Running the notebook

```bash
jupyter lab notebooks/testing.ipynb
```

The notebook imports from `src/` using an absolute path (`sys.path.insert(0, "/Users/pnadel01/Desktop/dynamic-reading-lists")`). If working on a different machine, update that path or install the package in editable mode.

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

**`src/templates/`** — Jinja2 templates. `chunk-page.html.jinja` extends `base.html.jinja` and renders each sentence with inline gloss spans (`data-gloss`, `data-form`) plus translation and collapsible notes.

### LLM server

All annotation calls go to a local LLaMA.cpp-compatible server. Instantiate annotators with the port it is listening on:

```python
GlossAnnotator(port=8080, language="Ancient Greek", author="Thucydides", work="...")
TranslationAnnotator(port=8080, language="Ancient Greek", author="Thucydides", work="...")
```

Annotation results are cached to JSON files; `Generator._create_annotations()` skips files that already exist.

### TEI namespace

All XPath queries use the TEI namespace `http://www.tei-c.org/ns/1.0` aliased as `tei`.

## Known issues / TODOs

- `src/site/generator.py` line 9 hard-codes an absolute path (`HERE = Path("/Users/pnadel01/Desktop/dynamic-reading-lists")`).
- `TranslationAnnotator` has a `# @TODO come back for drama` note regarding speaker handling for dramatic texts.
- `create_annotation_prompt()` in `prompts.py` is a stub (`pass`).
