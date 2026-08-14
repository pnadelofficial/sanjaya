import copy
import unicodedata
from urllib.parse import quote
from ..llm.annotations import WordAnnotator, SentenceAnnotator, Annotation


def _stem_sort_key(stem: str) -> list:
    """Sort key for chunk stems: numeric parts sort as ints, non-numeric as strings after."""
    result = []
    for part in stem.split("."):
        try:
            result.append((0, int(part)))
        except ValueError:
            result.append((1, part))
    return result
from .db import AnnotationDB, normalize_form
from perseus_cts.models import TEIDocument
from perseus_cts.chunker import Chunker

from lxml import etree
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import List, Dict, Optional, Union
from rich.progress import Progress, BarColumn, MofNCompleteColumn, TextColumn, TimeElapsedColumn
from rich.console import Console

console = Console()


class Generator:
    def __init__(
        self,
        document: TEIDocument,
        template_dir: Path,
        subunit_xpath: str,
        annotator_list: List[Union[WordAnnotator, SentenceAnnotator]],
        work: str,
        author: str,
        output_dir: Path,
        chunk_filter: Optional[Union[str, List[str]]] = None,
    ):
        roles = [a.role for a in annotator_list]
        if any(not r for r in roles):
            raise ValueError(f"All annotators must define a non-empty role attribute: {roles}")
        if len(roles) != len(set(roles)):
            raise ValueError(f"Annotators must have unique roles, got duplicates in: {roles}")

        self.template_dir = Path(template_dir)
        self.subunit_xpath = subunit_xpath
        self.annotator_list = annotator_list
        self.word_annotators = [a for a in annotator_list if isinstance(a, WordAnnotator)]
        self.sentence_annotators = [a for a in annotator_list if isinstance(a, SentenceAnnotator)]
        self.work = work
        self.author = author
        self.output_dir = Path(output_dir)
        self.ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

        self.chunk_filter = chunk_filter
        self.chunk_dir = self.output_dir / "chunks"
        Chunker(document).compile(self.chunk_dir)
        if not any(self.chunk_dir.glob("*.xml")):
            print("Warning: CTS chunker produced 0 chunks — falling back to body-div chunking.")
            self._compile_fallback_chunks(document, self.chunk_dir)

        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            # Templates are named *.jinja, which select_autoescape() with default
            # settings would leave unescaped; force autoescaping on so untrusted
            # annotator output (glosses, translations) can't break attributes or
            # inject markup. No template renders trusted HTML through a variable.
            autoescape=select_autoescape(default=True, default_for_string=True),
        )
        self.chunk_template = self.env.get_template("chunk-page.html.jinja")
        self.all_raw_pages = self._get_all()

    def _compile_fallback_chunks(self, document: TEIDocument, chunk_dir: Path) -> None:
        """Write one chunk file per direct div child of tei:body when the CTS chunker fails."""
        TEI_NS = "http://www.tei-c.org/ns/1.0"
        ns = {"tei": TEI_NS}
        root = document.root
        body = root.find(".//tei:body", ns)
        if body is None:
            raise RuntimeError("Cannot find tei:body in source document for fallback chunking.")
        divs = [el for el in body if etree.QName(el.tag).localname == "div"]
        if not divs:
            raise RuntimeError("Fallback chunker: no div children found under tei:body.")
        for idx, div in enumerate(divs):
            n = div.get("n") or str(idx + 1)
            unit = div.get("type", "div")
            chunk_el = etree.Element("citationChunk")
            chunk_el.set("unit", unit)
            chunk_el.set("n", n)
            elements_el = etree.SubElement(chunk_el, "elements")
            # deepcopy: appending the live node would reparent it out of the
            # source document's <body>, emptying the in-memory tree.
            elements_el.append(copy.deepcopy(div))
            chunk_dir.mkdir(parents=True, exist_ok=True)
            (chunk_dir / f"{n}.xml").write_bytes(
                etree.tostring(chunk_el, encoding="utf-8", xml_declaration=True, pretty_print=True)
            )
        print(f"Fallback chunker: wrote {len(divs)} chunk(s) to {chunk_dir}")

    # @TODO: fallback chunk files root at <citationChunk>, not a TEI element; XPath
    # patterns that require a specific ancestor (e.g. .//tei:body//tei:p) silently
    # return 0 elements per fallback chunk. A shared chunking abstraction that
    # normalises the XML root would remove this constraint.
    def _get_one(self, xml_path: Path) -> List[etree._Element]:
        tree = etree.parse(str(xml_path))
        root = tree.getroot()
        return root.findall(self.subunit_xpath, namespaces=self.ns)

    def _matches_filter(self, stem: str) -> bool:
        if not self.chunk_filter:
            return True
        filters = [self.chunk_filter] if isinstance(self.chunk_filter, str) else self.chunk_filter
        return any(stem == f or stem.startswith(f + ".") for f in filters)

    def _get_all(self) -> Dict[Path, List[etree._Element]]:
        xml_files = [p for p in self.chunk_dir.glob("*.xml") if self._matches_filter(p.stem)]
        return {xml_file: self._get_one(xml_file) for xml_file in xml_files}

    def _create_annotations(self, annotation_dir: Path) -> Dict:
        annotation_dir.mkdir(parents=True, exist_ok=True)
        annotes = {}
        chunks = list(self.all_raw_pages.items())

        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            chunk_task = progress.add_task("[bold]Chunks", total=len(chunks))
            sent_task = progress.add_task("", total=0, visible=False)

            for xml_file, subunits in chunks:
                chunk_id = xml_file.stem
                texts = ["".join(s.itertext()) for s in subunits]
                chunk_annotes = {}

                for annotator in self.annotator_list:
                    cache_file = annotation_dir / f"{chunk_id}_{annotator.role}.json"
                    if cache_file.exists():
                        progress.log(f"[{chunk_id}] {annotator.role}: cache")
                        result = annotator.load_annotations_from_json(cache_file)
                    else:
                        progress.update(
                            sent_task,
                            description=f"  [dim]{annotator.role}[/] [{chunk_id}]",
                            total=len(texts),
                            completed=0,
                            visible=True,
                        )
                        result = annotator.annotate_and_save(
                            texts=texts,
                            filename=cache_file,
                            on_sentence=lambda: progress.advance(sent_task),
                        )
                        progress.update(sent_task, visible=False)

                    chunk_annotes[annotator.role] = result

                annotes[xml_file] = chunk_annotes
                progress.advance(chunk_task)

        return annotes

    def _create_sentences(self, annotes: Dict) -> Dict:
        sentences = {}
        first_role = self.annotator_list[0].role
        for xml_file, chunk_annotes in annotes.items():
            chunk_id = xml_file.stem
            n = len(chunk_annotes[first_role])
            chunk_sentences = []
            for i in range(n):
                sentence_data = {
                    role: outputs[i].annotation
                    for role, outputs in chunk_annotes.items()
                    if i < len(outputs)
                }
                sentence_data["base_text"] = chunk_annotes[first_role][i].text
                for role in [a.role for a in self.annotator_list]:
                    if sentence_data.get(role) is None:
                        sentence_data.pop(role, None)
                        sentence_data[f"{role}_failed"] = True
                for annotator in self.word_annotators:
                    role = annotator.role
                    if role not in sentence_data:
                        continue
                    normalized = []
                    for t_idx, token in enumerate(sentence_data[role]):
                        if not token.get("annotation"):
                            continue
                        # Role is part of the id: two word annotators may tokenize
                        # the same sentence differently, so a role-free id would
                        # collide in the shared tokens table (position stays last).
                        token["id"] = f"tk-{chunk_id}-{i}-{role}-{t_idx}"
                        # NFC here keeps display, data-form, DB form, and the
                        # ?highlight= key identical so frequency counts and
                        # highlighting all match on the same canonical string.
                        token["text"] = normalize_form(token.get("text", ""))
                        normalized.append(token)
                    sentence_data[role] = normalized
                chunk_sentences.append(sentence_data)
            sentences[xml_file] = chunk_sentences
        return sentences

    def _collect_vocab(self, sentences: Dict) -> Dict:
        if not self.word_annotators:
            return {}
        word_role = self.word_annotators[0].role
        vocab = {}
        for xml_file, chunk_sentences in sentences.items():
            chunk_id = xml_file.stem
            for sentence in chunk_sentences:
                if word_role not in sentence:
                    continue
                context = sentence.get("base_text", "")
                for token in sentence[word_role]:
                    form = token["text"]
                    if not any(unicodedata.category(c).startswith("L") for c in form):
                        continue
                    label_val = token["annotation"].get("label", "")
                    token_id = token.get("id", "")
                    if form not in vocab:
                        vocab[form] = {"glosses": [], "occurrences": []}
                    if label_val and label_val not in vocab[form]["glosses"]:
                        vocab[form]["glosses"].append(label_val)
                    vocab[form]["occurrences"].append({
                        "token_id": token_id,
                        "chunk": chunk_id,
                        "href": f"../{chunk_id}.html?highlight={quote(form)}#{token_id}",
                        "context": context,
                    })
        return vocab

    def write_html(self, sentences: Dict, html_dir: Path) -> None:
        html_dir.mkdir(parents=True, exist_ok=True)
        word_roles = [a.role for a in self.word_annotators]
        sentence_roles = [a.role for a in self.sentence_annotators]
        for xml_file, chunk_sentences in sentences.items():
            title = f"{self.work} - {self.author} - {xml_file.stem}"
            html = self.chunk_template.render(
                title=title,
                sentences=chunk_sentences,
                chunk_id=xml_file.stem,
                depth_prefix="",
                word_roles=word_roles,
                sentence_roles=sentence_roles,
            )
            out_path = html_dir / f"{xml_file.stem}.html"
            out_path.write_text(html)
            print(f"Wrote {out_path}")

    def write_db(self, sentences: Dict, db: AnnotationDB) -> None:
        sorted_chunks = sorted(
            sentences.items(),
            key=lambda kv: _stem_sort_key(kv[0].stem),
        )
        failures = 0
        with Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[bold]Writing DB", total=len(sorted_chunks))

            for chunk_seq, (xml_file, chunk_sentences) in enumerate(sorted_chunks):
                chunk_id = xml_file.stem

                if db.chunk_exists(chunk_id):
                    progress.log(f"[{chunk_id}] db: already present, skipping")
                    progress.advance(task)
                    continue

                try:
                    db.write_chunk(chunk_id, sequence=chunk_seq, source_path=str(xml_file))

                    for sent_i, sentence in enumerate(chunk_sentences):
                        sentence_id = f"{chunk_id}-{sent_i}"
                        db.write_sentence(sentence_id, chunk_id=chunk_id, position=sent_i)

                        for annotator in self.word_annotators:
                            role = annotator.role
                            if role not in sentence:
                                continue
                            for token in sentence[role]:
                                token_id = token["id"]
                                position = int(token_id.rsplit("-", 1)[-1])
                                label = token["annotation"].get("label")
                                if not label:
                                    continue
                                db.write_token(
                                    token_id=token_id,
                                    sentence_id=sentence_id,
                                    chunk_id=chunk_id,
                                    position=position,
                                    form=token["text"],
                                )
                                db.write_word_annotation(
                                    token_id=token_id,
                                    annotator=role,
                                    value=label,
                                    confidence=token["annotation"].get("confidence"),
                                )
                                # Persist every other returned field (lemma,
                                # part_of_speech, morphology, …) as queryable
                                # features. Mirror the popup: drop the canonical
                                # key and any field that just repeats the gloss.
                                features = {
                                    k: v
                                    for k, v in token["annotation"].items()
                                    if k not in ("label", "confidence") and v != label
                                }
                                db.write_word_annotation_features(
                                    token_id=token_id,
                                    annotator=role,
                                    features=features,
                                )

                        for annotator in self.sentence_annotators:
                            role = annotator.role
                            if role not in sentence:
                                continue
                            ann = sentence[role]
                            summary = ann.get("summary")
                            if not summary:
                                continue
                            db.write_sentence_annotation(
                                sentence_id=sentence_id,
                                annotator=role,
                                value=summary,
                                confidence=ann.get("confidence"),
                            )
                            # Persist every other returned field (e.g.
                            # CommentAnnotator's per-comment tagging) as
                            # queryable features, mirroring word_annotation_features.
                            features = {
                                k: v
                                for k, v in ann.items()
                                if k not in ("summary", "confidence") and v != summary
                            }
                            db.write_sentence_annotation_features(
                                sentence_id=sentence_id,
                                annotator=role,
                                features=features,
                            )

                    db.commit()
                    progress.log(f"[{chunk_id}] db: wrote {len(chunk_sentences)} sentences")

                except Exception as e:
                    db.rollback()
                    progress.log(f"[{chunk_id}] db: write failed, rolled back — {e}")
                    failures += 1

                progress.advance(task)

        if failures:
            console.print(f"[yellow]DB write: {failures}/{len(sorted_chunks)} chunk(s) failed (see log above).[/yellow]")

    def write_search(self, html_dir: Path) -> None:
        search_template = self.env.get_template("search.html.jinja")
        html = search_template.render(
            title=f"Search — {self.work}",
            work=self.work,
            author=self.author,
            depth_prefix="",
        )
        out_path = html_dir / "search.html"
        out_path.write_text(html)
        print(f"Wrote {out_path}")

    def write_index(self, html_dir: Path) -> None:
        all_stems = sorted(
            [p.stem for p in html_dir.glob("*.html") if p.stem not in ("index", "search")],
            key=_stem_sort_key,
        )
        sections = [{"label": stem, "href": f"{stem}.html"} for stem in all_stems]
        index_template = self.env.get_template("index.html.jinja")
        html = index_template.render(work=self.work, author=self.author, sections=sections, depth_prefix="")
        out_path = html_dir / "index.html"
        out_path.write_text(html)
        print(f"Wrote {out_path}")

    def write_vocab(self, vocab: Dict, vocab_dir: Path) -> None:
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_template = self.env.get_template("vocab-page.html.jinja")
        for form, data in vocab.items():
            html = vocab_template.render(
                title=form,
                form=form,
                glosses=data["glosses"],
                occurrences=data["occurrences"],
                work=self.work,
                author=self.author,
                depth_prefix="../",
            )
            out_path = vocab_dir / f"{form}.html"
            out_path.write_text(html)
        print(f"Wrote {len(vocab)} vocab pages to {vocab_dir}")

    def write_vocab_index(self, vocab: Dict, vocab_dir: Path) -> None:
        forms = sorted(vocab.keys())
        vocab_index_template = self.env.get_template("vocab-index.html.jinja")
        html = vocab_index_template.render(
            title=f"Vocabulary — {self.work}",
            forms=forms,
            work=self.work,
            author=self.author,
            depth_prefix="../",
        )
        out_path = vocab_dir / "index.html"
        out_path.write_text(html)
        print(f"Wrote vocab index to {out_path}")

    def generate_site(self, write_db: bool = True, write_html: bool = True) -> None:
        annotation_dir = self.output_dir / "annotations"
        html_dir = self.output_dir / "html"
        annotes = self._create_annotations(annotation_dir=annotation_dir)
        sentences = self._create_sentences(annotes)
        if write_html:
            self.write_html(sentences, html_dir=html_dir)
            self.write_index(html_dir)
            self.write_search(html_dir)
            vocab = self._collect_vocab(sentences)
            vocab_dir = html_dir / "vocab"
            self.write_vocab(vocab, vocab_dir)
            self.write_vocab_index(vocab, vocab_dir)
        if write_db:
            db_path = html_dir / "data" / "annotations.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            with AnnotationDB(db_path) as db:
                db.create_schema()
                db.register_annotators(self.annotator_list)
                self.write_db(sentences, db)
