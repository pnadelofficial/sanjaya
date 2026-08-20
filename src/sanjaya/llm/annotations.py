import requests
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Callable, ClassVar, List, Optional, Dict, Any, Sequence, Union

from . import validator


@dataclass
class Annotation:
    text: str
    annotation: Any  # List[dict] for word-level, dict for sentence-level, None on failure


def call_model(
    base_url: str,
    messages: List[Dict[str, str]],
    model: str = "default",
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Calls an OpenAI-compatible chat completions endpoint.

    For local servers (LLaMA.cpp, LM Studio, Ollama) pass the server's base URL
    and omit api_key. For hosted APIs (OpenAI, etc.) pass the provider's base URL
    and your api_key.
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, **kwargs}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()


def call_model_with_retry(
    base_url: str,
    messages: List[Dict[str, str]],
    required_keys: Sequence[str],
    model: str = "default",
    api_key: Optional[str] = None,
    max_retries: int = 5,
    extract: Optional[Callable[[str], Optional[str]]] = None,
) -> Optional[Union[dict, list]]:
    """
    Call the model and validate that its JSON response contains every key
    in required_keys, retrying (blind resubmission of the same messages —
    no corrective feedback yet) up to max_retries times on malformed JSON
    or a response missing one of required_keys.

    required_keys is checked against the model's own raw response, before
    any merging with other data sources (e.g. NLPGlossAnnotator merging in
    nlp_pipeline's linguistic features) — a caller should perform that
    merge only after this returns successfully.

    extract: given the raw response content, return the JSON substring to
    parse (or None if none found). Defaults to
    validator.extract_json_from_annotation's default (single JSON object)
    behavior; pass a custom callable for array responses or other
    extraction strategies (e.g. CommentAnnotator's array-with-bare-object
    fallback).

    Returns the parsed dict/list on success, or None if every attempt
    failed.
    """
    extract = extract or validator.extract_json_from_annotation
    last_raw = None
    for attempt in range(1, max_retries + 1):
        response = call_model(base_url, messages, model=model, api_key=api_key)
        content = response.get("choices", [{}])[0].get("message", {}).get("content", "")
        raw = extract(content)
        last_raw = raw
        parsed = validator.validate_annotation(raw)
        if not parsed:
            print(f"[attempt {attempt}/{max_retries}] invalid JSON response: {raw!r}")
            continue
        missing = validator.find_missing_keys(parsed, required_keys)
        if missing:
            print(f"[attempt {attempt}/{max_retries}] response missing key(s) {missing}: {parsed!r}")
            continue
        return parsed
    print(f"call_model_with_retry: all {max_retries} attempts failed; last raw response: {last_raw!r}")
    return None


class WordAnnotator(ABC):
    """
    Abstract base for word-level annotators.

    Subclasses must define a non-empty `role` class attribute and implement
    `annotate()`, which processes a full sentence and returns one Annotation
    per token. Each token's annotation dict must include a "label" key — the
    value displayed in the reading interface.

    output_schema declares the structure of the annotation dict beyond "label":
      None  → output is a plain string (the label itself, nothing else validated)
      dict  → output has label + structured metadata fields declared here;
              "label" is always implicit and never declared in output_schema

    output_schema is also used at build time to derive the SQLite column schema
    for this annotator's data.

    Subclasses may override `tokenize()` for language-specific behaviour;
    the default uses NLTK's word_tokenize.

    The annotation backend is entirely up to the subclass (LLM, spaCy, rule-based,
    morphology lookup, etc.). No LLM machinery lives in this base class.
    """
    role: str = ""
    output_schema: ClassVar[dict[str, type] | None] = None

    @abstractmethod
    def annotate(self, sentence: str) -> List[Annotation]:
        """Return one Annotation per token. Each annotation dict must contain 'label'."""
        ...

    def tokenize(self, sentence: str) -> List[str]:
        from nltk import word_tokenize
        return word_tokenize(sentence)

    def save_as_json(self, annotations: List[Annotation], filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in annotations], f, ensure_ascii=False, indent=4)

    def load_annotations_from_json(self, filename: str) -> List[Annotation]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Annotation(text=item["text"], annotation=item["annotation"]) for item in data]

    def annotate_and_save(
        self,
        texts: List[str],
        filename: str,
        on_sentence: Optional[Callable[[], None]] = None,
    ) -> List[Annotation]:
        results = []
        for text in texts:
            # A subunit with no text (e.g. a <l> containing only a <gap/>
            # lacuna marker) has nothing to tokenize or gloss, and some
            # backends (e.g. NLPGlossAnnotator's stanza pipeline) error out
            # on an empty/blank string rather than returning zero tokens.
            token_annotations = self.annotate(text) if text.strip() else []
            valid_tokens = [
                a for a in token_annotations
                if validator.validate_output_schema(a.annotation, self.output_schema, "label")
            ]
            results.append(Annotation(text=text, annotation=[asdict(a) for a in valid_tokens]))
            if on_sentence is not None:
                on_sentence()
        self.save_as_json(results, filename)
        return results


class SentenceAnnotator(ABC):
    """
    Abstract base for sentence-level annotators.

    Subclasses must define a non-empty `role` class attribute and implement
    `annotate()`, which processes a full sentence and returns a single Annotation
    or None on failure. The annotation dict must include a "summary" key — the
    main text displayed in the reading interface.

    annotate() also receives an optional `speaker` — the name Generator
    resolved from an enclosing TEI <sp>'s <speaker> child, for text where the
    sentence is a line of dialogue (e.g. drama). It is None for text with no
    such markup. Subclasses that have no use for it (most do not) can ignore
    the parameter entirely.

    output_schema declares the structure of the annotation dict beyond "summary":
      None  → output is a plain string (the summary itself, nothing else validated)
      dict  → output has summary + structured metadata fields declared here;
              "summary" is always implicit and never declared in output_schema

    output_schema is also used at build time to derive the SQLite column schema
    for this annotator's data.

    The annotation backend is entirely up to the subclass. No LLM machinery
    lives in this base class.
    """
    role: str = ""
    output_schema: ClassVar[dict[str, type] | None] = None

    @abstractmethod
    def annotate(self, sentence: str, speaker: Optional[str] = None) -> Optional[Annotation]:
        """Return a single Annotation for the sentence, or None on failure. Must include 'summary'."""
        ...

    def save_as_json(self, annotations: List[Annotation], filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in annotations], f, ensure_ascii=False, indent=4)

    def load_annotations_from_json(self, filename: str) -> List[Annotation]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Annotation(text=item["text"], annotation=item["annotation"]) for item in data]

    def annotate_and_save(
        self,
        texts: List[str],
        filename: str,
        speakers: Optional[List[Optional[str]]] = None,
        on_sentence: Optional[Callable[[], None]] = None,
    ) -> List[Annotation]:
        results = []
        speakers = speakers or [None] * len(texts)
        for text, speaker in zip(texts, speakers):
            # See the matching comment in WordAnnotator.annotate_and_save():
            # a blank subunit has nothing to summarize and some backends
            # error out on empty input rather than declining gracefully.
            ann = self.annotate(text, speaker) if text.strip() else None
            if ann is not None and not validator.validate_output_schema(
                ann.annotation, self.output_schema, "summary"
            ):
                ann = None
            results.append(Annotation(text=text, annotation=ann.annotation if ann else None))
            if on_sentence is not None:
                on_sentence()
        self.save_as_json(results, filename)
        return results
