import argparse
from pathlib import Path

from src.site.chunker import DocumentChunker
from src.site.generator import Generator
from src.llm.annotators import GlossAnnotator, TranslationAnnotator

parser = argparse.ArgumentParser(description="Generate an annotated reading site from a TEI XML file.")
parser.add_argument("--file", help="Path to the TEI XML source file")
parser.add_argument("chunk", nargs="*", help="Chunk identifier(s) to process")
args = parser.parse_args()

chunker = DocumentChunker(
    xml_path=args.file,
    chunk_dir=Path("chunks"),
)
doc = chunker.chunk()

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

output_dir = Path("output/thucydides")
gen = Generator(
    document=doc,
    template_dir=Path("src/templates"),
    subunit_xpath=".//tei:p",
    annotator_list=[gloss_annotator, translation_annotator],
    work="History of the Peloponnesian War",
    author="Thucydides",
    output_dir=output_dir,
    chunk_filter=args.chunk,
)
gen.generate_site()
