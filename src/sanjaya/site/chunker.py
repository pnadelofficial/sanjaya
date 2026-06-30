from pathlib import Path
from perseus_cts.models import TEIDocument
from perseus_cts.chunker import Chunker


class DocumentChunker:
    """Load a TEI XML file and compile it into per-passage XML files."""

    def __init__(self, xml_path: Path, chunk_dir: Path):
        self.xml_path = Path(xml_path)
        self.chunk_dir = Path(chunk_dir)

    def chunk(self) -> TEIDocument:
        doc = TEIDocument(str(self.xml_path))
        Chunker(doc).compile(self.chunk_dir)
        return doc
