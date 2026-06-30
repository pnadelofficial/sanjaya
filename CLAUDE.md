# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

`sanjaya` generates richly annotated reading environments for historical texts — currently Ancient Greek. It supports word-level and sentence-level annotations from any backend (LLM, classical NLP, rule-based), and renders them into interactive HTML pages. The CLI accepts a YAML config file; annotator classes are loaded dynamically by dotted path, so users can supply their own without modifying sanjaya itself.

## Environment setup

Managed with [uv](https://github.com/astral-uv/uv). Python 3.12 required.

```bash
uv sync
uv pip install -e .
source .venv/bin/activate
```

The editable install is required: it wires up the `sanjaya` entry point and makes `sanjaya.*` imports resolve correctly.

The external dependency `perseus-cts` is installed directly from GitHub:
```
perseus-cts @ git+https://github.com/PerseusDLCode/perseus-cts.git
```

## Running the pipeline

The primary entry point is the CLI:

```bash
sanjaya --config config.yaml
```

To limit processing to a subset of chunks during development:

```bash
sanjaya --config config.yaml --chunk 1.1
```

`run.py` at the project root is a direct Python API example and can be used as an alternative to the CLI.

There are currently no automated tests.

## Architecture

### Data flow

```
config.yaml
  → sanjaya.cli            # loads config, dynamically imports annotator classes
  → TEIDocument            # parses TEI XML source
  → Generator.__init__()   # chunks the document (writes per-chunk XML files)
  → Generator._get_all()   # reads chunked XMLs via lxml
  → WordAnnotator / SentenceAnnotator  # annotate each subunit, cache JSON
  → Generator.write_html() # renders Jinja2 templates to HTML
```

### Package layout

```
src/sanjaya/
  cli.py              ← entry point: config loading + dynamic annotator import
  llm/
    annotations.py    ← WordAnnotator and SentenceAnnotator ABCs + call_model()
    annotators.py     ← GlossAnnotator (word-level) and TranslationAnnotator (sentence-level)
    prompts.py        ← string.Template prompt builders
    validator.py      ← JSON extraction, repair, and validation
  site/
    generator.py      ← Generator: orchestrates chunking, annotation, HTML rendering
    chunker.py        ← DocumentChunker: thin wrapper around perseus_cts.Chunker
  templates/          ← Jinja2 templates (bundled with the package)
    base.html.jinja
    chunk-page.html.jinja
    index.html.jinja
    vocab-page.html.jinja
    vocab-index.html.jinja
```

### Key modules

**`sanjaya/llm/annotations.py`** — Two abstract base classes and shared utilities:

- `WordAnnotator` — abstract; subclasses implement `annotate(sentence) -> List[Annotation]`, returning one `Annotation` per token. The annotation dict must include `"label"`. Provides a default `tokenize()` backed by NLTK; override for language-specific tokenisation. No LLM machinery in the base.
- `SentenceAnnotator` — abstract; subclasses implement `annotate(sentence) -> Optional[Annotation]`. The annotation dict must include `"summary"`. No LLM machinery in the base.
- `call_model()` — HTTP POST to any OpenAI-compatible `/v1/chat/completions` endpoint. Available for use by LLM-backed annotator subclasses.
- Both bases provide `annotate_and_save()`, `save_as_json()`, and `load_annotations_from_json()`.

**`sanjaya/llm/annotators.py`** — Concrete LLM-backed implementations:

- `GlossAnnotator(WordAnnotator)` — calls the model once per token; maps the LLM's `"gloss"` field to the canonical `"label"` key.
- `TranslationAnnotator(SentenceAnnotator)` — calls the model once per sentence; maps the LLM's `"translation"` field to the canonical `"summary"` key.

Each subclass owns all its LLM machinery (`base_url`, `model`, `api_key`) with no intermediate base class enforcing an LLM pattern.

**`sanjaya/llm/prompts.py`** — `string.Template`-based prompt builders for gloss and translation tasks, returned as OpenAI-style `messages` lists.

**`sanjaya/llm/validator.py`** — Extracts JSON from LLM responses via regex, repairs malformed JSON with `json_repair`, and optionally validates against a JSON schema.

**`sanjaya/site/generator.py`** — `Generator` class orchestrates the full pipeline. It partitions the annotator list into `word_annotators` and `sentence_annotators` via `isinstance` checks and uses that distinction throughout: token normalisation (adding stable IDs, filtering empty annotations) runs for all word-level annotators; the template receives `word_roles` and `sentence_roles` lists rather than hardcoded role names.

**`sanjaya/cli.py`** — Loads the YAML config, resolves paths relative to the config file, dynamically imports annotator classes via `importlib.import_module`, and drives `Generator`. The `--chunk` flag overrides `chunk_filter` from the config.

### Annotator contract

| Base class | `annotate()` return | Required annotation key | Rendered as |
|---|---|---|---|
| `WordAnnotator` | `List[Annotation]` | `"label"` | Clickable token spans |
| `SentenceAnnotator` | `Optional[Annotation]` | `"summary"` | Inline text + collapsible extras |

All other keys in the annotation dict are rendered in a collapsible block beneath the primary field.

### Template rendering

`chunk-page.html.jinja` receives `word_roles` and `sentence_roles` lists from `Generator.write_html()` and loops over them generically — no role names are hardcoded in the template. Word-level tokens are rendered as `<span>` elements with `data-label` and `data-form` attributes; sentence-level annotations render their `summary` field inline with remaining fields in a `<details>` block.

### TEI namespace

All XPath queries use the TEI namespace `http://www.tei-c.org/ns/1.0` aliased as `tei`.

## Known issues / TODOs

- `TranslationAnnotator` has a `# @TODO come back for drama` note regarding speaker handling for dramatic texts.
- `create_annotation_prompt()` in `prompts.py` is a stub (`pass`).
- Search and vocabulary are placeholder implementations. The pagefind-based search is removed; a SQLite + sql.js replacement is planned with a dynamic schema tied to the annotator configuration.
