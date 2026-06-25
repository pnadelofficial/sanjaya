import argparse
from pathlib import Path

from src.site.chunker import DocumentChunker
from src.site.generator import Generator
from src.llm.annotators import GlossAnnotator, TranslationAnnotator

parser = argparse.ArgumentParser(description="Generate an annotated reading site from a TEI XML file.")
parser.add_argument("file", help="Path to the TEI XML source file")
args = parser.parse_args()

chunker = DocumentChunker(
    xml_path=args.file,
    chunk_dir=Path("chunks"),
)
doc = chunker.chunk()

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

output_dir = Path("output/thucydides")
gen = Generator(
    document=doc,
    template_dir=Path("src/templates"),
    subunit_xpath=".//tei:p",
    annotator_list=[gloss_annotator, translation_annotator],
    work="History of the Peloponnesian War",
    author="Thucydides",
    output_dir=output_dir,
    chunk_filter="1.1",
)
gen.generate_site()
