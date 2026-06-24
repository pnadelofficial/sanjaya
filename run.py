from perseus_cts.models import Corpus
from src.site.generator import Generator
from pathlib import Path
from src.llm.annotators import GlossAnnotator, TranslationAnnotator

gloss_annotator = GlossAnnotator(
    port=8080,
    language="grc",
    author="Thucydides",
    work="The History of the Peloponnesian War",
)

translation_annotator = TranslationAnnotator(
    port=8080,
    language="grc",
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
    chunk_filter="1.1"
)
gen.generate_site()