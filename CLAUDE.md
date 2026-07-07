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
  → Generator.write_db()   # writes SQLite DB (skipped with --no-db)
```

### Package layout

```
src/sanjaya/
  cli.py              ← entry point: config loading + dynamic annotator import
  llm/
    annotations.py    ← WordAnnotator and SentenceAnnotator ABCs + call_model()
    annotators.py     ← GlossAnnotator (word-level) and TranslationAnnotator (sentence-level)
    prompts.py        ← string.Template prompt builders
    validator.py      ← JSON extraction, repair, validation, and output_schema checking
  site/
    generator.py      ← Generator: orchestrates chunking, annotation, HTML + DB rendering
    db.py             ← AnnotationDB: schema creation and row writing (usable standalone)
    chunker.py        ← DocumentChunker: thin wrapper around perseus_cts.Chunker
  templates/          ← Jinja2 templates (bundled with the package)
    base.html.jinja
    chunk-page.html.jinja
    index.html.jinja
    search.html.jinja
    vocab-page.html.jinja
    vocab-index.html.jinja
```

### Key modules

**`sanjaya/llm/annotations.py`** — Two abstract base classes and shared utilities:

- `WordAnnotator` — abstract; subclasses implement `annotate(sentence) -> List[Annotation]`, returning one `Annotation` per token. The annotation dict must include `"label"`. Provides a default `tokenize()` backed by NLTK; override for language-specific tokenisation. No LLM machinery in the base. Declares `output_schema: ClassVar[dict[str, type] | None] = None` for optional per-field type validation before DB writes.
- `SentenceAnnotator` — abstract; subclasses implement `annotate(sentence) -> Optional[Annotation]`. The annotation dict must include `"summary"`. No LLM machinery in the base. Same `output_schema` class variable.
- `call_model()` — HTTP POST to any OpenAI-compatible `/v1/chat/completions` endpoint. Available for use by LLM-backed annotator subclasses.
- Both bases provide `annotate_and_save()`, `save_as_json()`, and `load_annotations_from_json()`.

**`sanjaya/llm/annotators.py`** — Concrete LLM-backed implementations:

- `GlossAnnotator(WordAnnotator)` — calls the model once per token; maps the LLM's `"gloss"` field to the canonical `"label"` key.
- `TranslationAnnotator(SentenceAnnotator)` — calls the model once per sentence; maps the LLM's `"translation"` field to the canonical `"summary"` key.

Each subclass owns all its LLM machinery (`base_url`, `model`, `api_key`) with no intermediate base class enforcing an LLM pattern.

**`sanjaya/llm/prompts.py`** — `string.Template`-based prompt builders for gloss and translation tasks, returned as OpenAI-style `messages` lists.

**`sanjaya/llm/validator.py`** — Extracts JSON from LLM responses via regex, repairs malformed JSON with `json_repair`, and optionally validates against a JSON schema. `validate_output_schema()` checks that the required primary key (`"label"` or `"summary"`) is a non-empty string, then validates any additional fields declared in `output_schema`.

**`sanjaya/site/generator.py`** — `Generator` class orchestrates the full pipeline. It partitions the annotator list into `word_annotators` and `sentence_annotators` via `isinstance` checks and uses that distinction throughout: token normalisation (adding stable IDs, filtering empty annotations) runs for all word-level annotators; the template receives `word_roles` and `sentence_roles` lists rather than hardcoded role names. `write_db()` writes per-chunk SQLite transactions; `write_search()` renders the search page. `generate_site(write_db=True)` drives the full build — pass `write_db=False` to skip the DB.

**`sanjaya/site/db.py`** — `AnnotationDB` class. Schema creation (`create_schema()`) and row writing (`write_chunk/sentence/token/word_annotation/word_annotation_features/sentence_annotation`). `word_annotations` stores the canonical `label` per `(token_id, annotator)`; every other field an annotator returns (lemma, part_of_speech, morphology, …) goes into the generic key/value `word_annotation_features` table `(token_id, annotator, key, value)`, indexed on `(key, value)` for lemma/field lookups — no schema migration when annotator fields change. Usable independently of `Generator`. Writes are not auto-committed; call `commit()` once per chunk. Foreign key write order: chunks → sentences → tokens → word_annotations / word_annotation_features / sentence_annotations.

**`sanjaya/cli.py`** — Loads the YAML config, resolves paths relative to the config file, dynamically imports annotator classes via `importlib.import_module`, and drives `Generator`. `--chunk` overrides `chunk_filter` from the config; `--no-db` skips the SQLite write pass.

### Annotator contract

| Base class | `annotate()` return | Required annotation key | Rendered as |
|---|---|---|---|
| `WordAnnotator` | `List[Annotation]` | `"label"` | Clickable token spans |
| `SentenceAnnotator` | `Optional[Annotation]` | `"summary"` | Inline text + collapsible extras |

All other keys in the annotation dict are rendered in a collapsible block beneath the primary field.

`output_schema` (optional, `ClassVar[dict[str, type] | None]`) declares additional fields and their expected Python types. When set, `annotate_and_save()` drops any annotation that fails validation before it reaches the DB. Set to `None` (the default) for plain-string-only outputs like `GlossAnnotator` and `TranslationAnnotator`.

### Template rendering

`chunk-page.html.jinja` receives `word_roles` and `sentence_roles` lists from `Generator.write_html()` and loops over them generically — no role names are hardcoded in the template. Word-level tokens are rendered as `<span>` elements with `data-label` and `data-form` attributes; sentence-level annotations render their `summary` field inline with remaining fields in a `<details>` block.

`base.html.jinja` loads sql.js from the CDN and fetches `data/annotations.db` relative to the current page's depth prefix. The resolved `SQL.Database` instance is stored in `window.annotationsDB` and exposed as the `window.dbReady` Promise so any page can `await` it.

`search.html.jinja` searches four dimensions, all generic over annotator fields: **surface forms** (`tokens.form LIKE ?`), **lemmata** (`word_annotation_features` where `key='lemma'`, grouping all inflected forms + glosses + other fields under a lemma), **field values** (any other `word_annotation_features.value LIKE ?`, e.g. part of speech / morphology), and **annotations** (`word_annotations.value LIKE ?`). Gloss/occurrence counts and feature fields are fetched in separate queries and merged in JS — joining all features into one query would fan out rows and inflate counts. Chips pivot the search (form ↔ lemma ↔ field value ↔ gloss). Minimum 2 characters, 250 ms debounce.

### TEI namespace

All XPath queries use the TEI namespace `http://www.tei-c.org/ns/1.0` aliased as `tei`.

## Known issues / TODOs

- `TranslationAnnotator` has a `# @TODO come back for drama` note regarding speaker handling for dramatic texts.
- `create_annotation_prompt()` in `prompts.py` is a stub (`pass`).
- Vocabulary pages are a static HTML implementation; they are not yet backed by the SQLite DB.
- The sql.js search covers word-level data (forms, lemmata, field values, glosses); sentence-level annotation search is not yet implemented.
