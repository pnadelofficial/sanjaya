import requests
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any, Union


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


class WordAnnotator(ABC):
    """
    Abstract base for word-level annotators.

    Subclasses must define a non-empty `role` class attribute and implement
    `annotate()`, which processes a full sentence and returns one Annotation
    per token. Each token's annotation dict must include a "label" key — the
    value displayed in the reading interface.

    Subclasses may override `tokenize()` for language-specific behaviour;
    the default uses NLTK's word_tokenize.

    The annotation backend is entirely up to the subclass (LLM, spaCy, rule-based,
    morphology lookup, etc.). No LLM machinery lives in this base class.
    """
    role: str = ""

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

    def annotate_and_save(self, texts: List[str], filename: str) -> List[Annotation]:
        results = []
        for text in texts:
            token_annotations = self.annotate(text)
            results.append(Annotation(text=text, annotation=[asdict(a) for a in token_annotations]))
        self.save_as_json(results, filename)
        return results


class SentenceAnnotator(ABC):
    """
    Abstract base for sentence-level annotators.

    Subclasses must define a non-empty `role` class attribute and implement
    `annotate()`, which processes a full sentence and returns a single Annotation
    or None on failure. The annotation dict must include a "summary" key — the
    main text displayed in the reading interface.

    The annotation backend is entirely up to the subclass. No LLM machinery
    lives in this base class.
    """
    role: str = ""

    @abstractmethod
    def annotate(self, sentence: str) -> Optional[Annotation]:
        """Return a single Annotation for the sentence, or None on failure. Must include 'summary'."""
        ...

    def save_as_json(self, annotations: List[Annotation], filename: str) -> None:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump([asdict(a) for a in annotations], f, ensure_ascii=False, indent=4)

    def load_annotations_from_json(self, filename: str) -> List[Annotation]:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [Annotation(text=item["text"], annotation=item["annotation"]) for item in data]

    def annotate_and_save(self, texts: List[str], filename: str) -> List[Annotation]:
        results = []
        for text in texts:
            ann = self.annotate(text)
            results.append(Annotation(text=text, annotation=ann.annotation if ann else None))
        self.save_as_json(results, filename)
        return results
