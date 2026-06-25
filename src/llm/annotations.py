import requests
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any

from . import prompts
from . import validator

@dataclass
class Annotation:
    text: str
    annotation: Dict[str, Any]

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
    payload = {
        "model": model,
        "messages": messages,
        **kwargs,
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    return response.json()

class Annotator:
    """
    Base class for annotating text using an OpenAI-compatible chat completions endpoint.

    Subclasses must define a non-empty `role` class attribute (e.g. role = "gloss").
    The role is used as the sentence dict key in Generator and as the template variable name.
    """
    role: str = ""

    def __init__(self, base_url: str, model: str = "default", api_key: Optional[str] = None):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def annotate(self, text: str, annotation_fn: callable = prompts.create_annotation_prompt) -> Optional[str]:
        messages = annotation_fn(text)
        response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
        return validator.extract_json_from_annotation(
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )

    def batch_annotate(self, texts: List[str]) -> List[Optional[Annotation]]:
        annotations = []
        for text in texts:
            annotation = self.annotate(text)
            annotations.append(Annotation(text=text, annotation=annotation))
        return annotations

    def save_as_json(self, annotations: List[Annotation], filename: str) -> None:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([asdict(annotation) for annotation in annotations], f, ensure_ascii=False, indent=4)

    def load_annotations_from_json(self, filename: str) -> List[Annotation]:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [Annotation(text=item["text"], annotation=item["annotation"]) for item in data]

    def annotate_and_save(self, texts: List[str], filename: str) -> None:
        annotations = self.batch_annotate(texts)
        self.save_as_json(annotations, filename)
        return annotations
