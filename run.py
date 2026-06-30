# run.py — direct Python API usage example.
# For normal use, prefer the CLI:  sanjaya --config config.yaml

from pathlib import Path
from perseus_cts.models import TEIDocument

from sanjaya.site.generator import Generator
from sanjaya.llm.annotators import GlossAnnotator, TranslationAnnotator

doc = TEIDocument("data/thucydides.xml")

gloss_annotator = GlossAnnotator(
    base_url="http://localhost:8080",
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
    # model="gpt-4o",       # uncomment for hosted APIs
    # api_key="sk-...",     # omit for local servers
)

translation_annotator = TranslationAnnotator(
    base_url="http://localhost:8080",
    language="Ancient Greek",
    author="Thucydides",
    work="The History of the Peloponnesian War",
    # model="gpt-4o",
    # api_key="sk-...",
)

gen = Generator(
    document=doc,
    template_dir=Path("src/sanjaya/templates"),
    subunit_xpath=".//tei:p",
    annotator_list=[gloss_annotator, translation_annotator],
    work="History of the Peloponnesian War",
    author="Thucydides",
    output_dir=Path("output/thucydides"),
    chunk_filter=None,
)
gen.generate_site()
