# sanjaya

`sanjaya` generates richly annotated reading environments for historical texts — currently Ancient Greek. It calls a locally-hosted LLM to produce word-level glosses and sentence-level translations, then renders them into interactive HTML pages where readers can click on any word to reveal its gloss.

## Requirements

- Python 3.12
- [uv](https://github.com/astral-uv/uv)
- A running LLaMA.cpp-compatible LLM server (e.g. [llama.cpp](https://github.com/ggerganov/llama.cpp) with `--server`)
- TEI XML source files

## Installation

```bash
git clone https://github.com/pnadelofficial/sanjaya.git
cd sanjaya
uv sync
uv pip install -e .   # editable install — keeps imports working without hard-coded paths
source .venv/bin/activate
```

## Usage

### 1. Prepare source data

Place your TEI XML files in `test-data/`. The pipeline uses `perseus_cts.Corpus` to discover and chunk documents automatically.

### 2. Start the LLM server

Start a LLaMA.cpp-compatible server on a local port (default `8080`):

```bash
llama-server --model your-model.gguf --port 8080
```

Any server that exposes a `/v1/chat/completions` endpoint works.

### 3. Run the pipeline

Edit `run.py` to point at your corpus, then run:

```bash
python run.py
```

`run.py` shows the full wiring: create annotators, load a corpus document, configure a `Generator`, and call `generate_site()`. Use `chunk_filter` on `Generator` to process only a subset of chunks while iterating (e.g. `chunk_filter="1.1"`).

```python
from perseus_cts.models import Corpus
from src.site.generator import Generator
from src.llm.annotators import GlossAnnotator, TranslationAnnotator
from pathlib import Path

gloss_annotator = GlossAnnotator(
    port=8080,
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
)
translation_annotator = TranslationAnnotator(
    port=8080,
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
)

corpus = Corpus("test-data")
doc = next(corpus.documents())

gen = Generator(
    document=doc,
    template_dir=Path("src/templates"),
    subunit_xpath=".//tei:p",
    annotator_list=[gloss_annotator, translation_annotator],
    work="History of the Peloponnesian War",
    author="Thucydides",
    output_dir=Path("output/thucydides"),
)
gen.generate_site()
```

### 4. View the output

Open `output/<work>/index.html` in a browser. Each chunk page renders sentences with clickable glossed words; clicking a word reveals its gloss in a popup. As a tip: use the Python utility `python -m http.server` from the `output/<work>` directory, to render the HTML in that directory on `localhost:8000`. 

## Caching

Annotation results are cached as JSON files under the output directory. Re-running the pipeline skips any chunk that has already been annotated, so you only pay LLM costs once per chunk.

## Extending

To add a new annotation type, subclass `Annotator` in `src/llm/annotations.py`, set a unique `role` string, and implement `annotate()`. Pass the new annotator to `Generator` alongside the existing ones. The Jinja template (`src/templates/chunk-page.html.jinja`) will render unknown roles as collapsible raw JSON until you add a dedicated template branch.
