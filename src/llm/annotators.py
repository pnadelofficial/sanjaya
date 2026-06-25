from .annotations import Annotator, Annotation, call_model
from . import prompts
from . import validator

from typing import List, Optional
from nltk import word_tokenize


class GlossAnnotator(Annotator):
    """Word-level gloss annotator. Tokenizes each subunit and calls the model once per token."""
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
        super().__init__(base_url, model=model, api_key=api_key)
        self.language = language
        self.author = author
        self.work = work

    def tokenize(self, sentence: str) -> List[str]:
        return word_tokenize(sentence)

    def annotate(self, text: str, annotation_fn: callable = prompts.create_gloss_messages) -> List[Annotation]:
        tokens = self.tokenize(text)
        word_by_word_annotations = []
        for i, token in enumerate(tokens):
            messages = annotation_fn(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=text,
                language_id=i,
                language_word=token,
            )
            response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
            annotation = validator.extract_json_from_annotation(
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            valid_annotation = validator.validate_annotation(annotation)
            if not valid_annotation:
                print(f"Invalid annotation for token: {token}\nAnnotation: {annotation}")
                continue
            word_by_word_annotations.append(Annotation(text=token, annotation=valid_annotation))
        return word_by_word_annotations


class TranslationAnnotator(Annotator):
    """Sentence-level translation annotator. Returns a translation and contextual notes per subunit."""
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
        super().__init__(base_url, model=model, api_key=api_key)
        self.language = language
        self.author = author
        self.work = work

    def annotate(self, text: str, annotation_fn: callable = prompts.create_translation_messages) -> Optional[Annotation]:
        messages = annotation_fn(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=text,
            speaker=None,  # @TODO come back for drama
        )
        response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
        annotation = validator.extract_json_from_annotation(
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        valid_annotation = validator.validate_annotation(annotation)
        if not valid_annotation:
            print(f"Invalid annotation for text: {text}\nAnnotation: {annotation}")
            return None
        return Annotation(text=text, annotation=valid_annotation)
