from ..llm import annotators as llm_annotators
from .search import build_search_index
from perseus_cts.models import TEIDocument
from perseus_cts.chunker import Chunker

from lxml import etree
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Optional
from tqdm import tqdm


class Generator:
    def __init__(
        self,
        document: TEIDocument,
        template_dir: Path,
        subunit_xpath: str,
        annotator_list: List[llm_annotators.Annotator],
        work: str,
        author: str,
        output_dir: Path,
        chunk_filter: Optional[str] = None,
    ):
        roles = [a.role for a in annotator_list]
        if any(not r for r in roles):
            raise ValueError(f"All annotators must define a non-empty role attribute: {roles}")
        if len(roles) != len(set(roles)):
            raise ValueError(f"Annotators must have unique roles, got duplicates in: {roles}")

        self.template_dir = Path(template_dir)
        self.subunit_xpath = subunit_xpath
        self.annotator_list = annotator_list
        self.work = work
        self.author = author
        self.output_dir = Path(output_dir)
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        self.chunk_filter = chunk_filter
        self.chunk_dir = self.output_dir / "chunks"
        Chunker(document).compile(self.chunk_dir)

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            autoescape=select_autoescape()
        )
        self.chunk_template = self.env.get_template("chunk-page.html.jinja")
        self.all_raw_pages = self._get_all()

    def _get_one(self, xml_path: Path) -> List[etree._Element]:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        return root.findall(self.subunit_xpath, namespaces=self.ns)

    def _matches_filter(self, stem: str) -> bool:
        if self.chunk_filter is None:
            return True
        # exact match (e.g. "1.1" matches only 1.1, not 1.10)
        # or subsection prefix (e.g. "1.1." matches 1.1.1, 1.1.2, ...)
        return stem == self.chunk_filter or stem.startswith(self.chunk_filter + ".")

    def _get_all(self) -> Dict[Path, List[etree._Element]]:
        xml_files = [p for p in self.chunk_dir.glob("*.xml") if self._matches_filter(p.stem)]
        return {xml_file: self._get_one(xml_file) for xml_file in xml_files}

    def _create_annotations(self, annotation_dir: Path) -> Dict:
        annotation_dir.mkdir(parents=True, exist_ok=True)
        annotes = {}
        chunks = list(self.all_raw_pages.items())
        for xml_file, subunits in tqdm(chunks, desc="Chunks", unit="chunk"):
            texts = [s.text for s in subunits]
            chunk_annotes = {}
            for annotator in self.annotator_list:
                cache_file = annotation_dir / f"{xml_file.stem}_{annotator.role}.json"
                if cache_file.exists():
                    tqdm.write(f"  [{xml_file.stem}] {annotator.role}: loading from cache")
                    result = annotator.load_annotations_from_json(cache_file)
                else:
                    tqdm.write(f"  [{xml_file.stem}] {annotator.role}: annotating {len(texts)} subunit(s)")
                    result = annotator.annotate_and_save(texts=texts, filename=cache_file)
                chunk_annotes[annotator.role] = result
            annotes[xml_file] = chunk_annotes
        return annotes

    def _create_sentences(self, annotes: Dict) -> Dict:
        sentences = {}
        first_role = self.annotator_list[0].role
        for xml_file, chunk_annotes in annotes.items():
            n = len(chunk_annotes[first_role])
            chunk_sentences = []
            for i in range(n):
                sentence_data = {role: outputs[i].annotation for role, outputs in chunk_annotes.items()}
                sentence_data["base_text"] = chunk_annotes[first_role][i].text
                chunk_sentences.append(sentence_data)
            sentences[xml_file] = chunk_sentences
        return sentences

    def write_html(self, sentences: Dict, html_dir: Path) -> None:
        html_dir.mkdir(parents=True, exist_ok=True)
        for xml_file, chunk_sentences in sentences.items():
            title = f"{self.work} - {self.author} - {xml_file.stem}"
            html = self.chunk_template.render(title=title, sentences=chunk_sentences)
            out_path = html_dir / f"{xml_file.stem}.html"
            out_path.write_text(html)
            print(f"Wrote {out_path}")

    def write_index(self, html_dir: Path) -> None:
        all_stems = sorted(
            [p.stem for p in html_dir.glob("*.html") if p.stem != "index"],
            key=lambda s: [int(x) for x in s.split(".")],
        )
        sections = [{"label": stem, "href": f"{stem}.html"} for stem in all_stems]

        index_template = self.env.get_template("index.html.jinja")
        html = index_template.render(work=self.work, author=self.author, sections=sections)
        out_path = html_dir / "index.html"
        out_path.write_text(html)
        print(f"Wrote {out_path}")

    def generate_site(self) -> None:
        annotation_dir = self.output_dir / "annotations"
        html_dir = self.output_dir / "html"
        annotes = self._create_annotations(annotation_dir=annotation_dir)
        sentences = self._create_sentences(annotes)
        self.write_html(sentences, html_dir=html_dir)
        self.write_index(html_dir)
        build_search_index(html_dir)
