from .annotations import WordAnnotator, SentenceAnnotator, Annotation, call_model
from . import prompts
from . import validator

from typing import List, Optional


class GlossAnnotator(WordAnnotator):
    """
    Word-level gloss annotator backed by an LLM.

    Calls the model once per token and returns an Annotation whose dict
    includes all fields returned by the LLM plus a canonical "label" key
    (mapped from the LLM's "gloss" field) required by WordAnnotator.

    Uses NLTK word_tokenize by default; override tokenize() for different
    language-specific tokenisation.
    """
    role = "gloss"

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key

    def annotate(self, sentence: str) -> List[Annotation]:
        tokens = self.tokenize(sentence)
        annotations = []
        for i, token in enumerate(tokens):
            messages = prompts.create_gloss_messages(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=sentence,
                language_id=i,
                language_word=token,
            )
            response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
            raw = validator.extract_json_from_annotation(
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            annotation = validator.validate_annotation(raw)
            if not annotation:
                print(f"Invalid annotation for token: {token}\nAnnotation: {raw}")
                continue
            annotation["label"] = annotation.get("gloss", "")
            annotations.append(Annotation(text=token, annotation=annotation))
        return annotations


class TranslationAnnotator(SentenceAnnotator):
    """
    Sentence-level translation annotator backed by an LLM.

    Returns an Annotation whose dict includes all fields returned by the LLM
    plus a canonical "summary" key (mapped from the LLM's "translation" field)
    required by SentenceAnnotator.
    """
    role = "translation"

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key

    def annotate(self, sentence: str) -> Optional[Annotation]:
        messages = prompts.create_translation_messages(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=sentence,
            speaker=None,  # @TODO come back for drama
        )
        response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
        raw = validator.extract_json_from_annotation(
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        annotation = validator.validate_annotation(raw)
        if not annotation:
            print(f"Invalid annotation for text: {sentence}\nAnnotation: {raw}")
            return None
        annotation["summary"] = annotation.get("translation", "")
        return Annotation(text=sentence, annotation=annotation)
