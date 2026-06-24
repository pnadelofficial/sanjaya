"""
File for defining custom annotators. 
By default:
- GlossAnnotator: annotates words with their glosses
- TranslationAnnotator: annotates sentences with their English translations and contextual notes
"""

from .annotations import Annotator, Annotation, call_model
from . import prompts
from . import validator

from typing import List, Optional
from nltk import word_tokenize

class GlossAnnotator(Annotator):
    """
    Needs access to tokenization so that we have word-level glosses.
    """
    role = "gloss"

    def __init__(self, port: str, language: str, author: str, work: str):
        super().__init__(port)
        self.language = language
        self.author = author
        self.work = work
    
    def tokenize(self, sentence: str) -> List[str]:
        return word_tokenize(sentence)

    def annotate(self, text: str, annotation_fn: callable = prompts.create_gloss_messages) -> List[Annotation]:
        # text will be a sentence
        tokens = self.tokenize(text)
        
        word_by_word_annotations = []
        for i, token in enumerate(tokens):
            messages = annotation_fn(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=text,
                language_id=i,
                language_word=token
            )
            response = call_model(port=self.port, messages=messages)
            annotation = validator.extract_json_from_annotation(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
            valid_annotation = validator.validate_annotation(annotation)
            if not valid_annotation:
                print(f"Invalid annotation for token: {token}\nAnnotation: {annotation}")
                continue
            word_by_word_annotations.append(Annotation(text=token, annotation=valid_annotation))
        return word_by_word_annotations

class TranslationAnnotator(Annotator):
    role = "translation"

    def __init__(self, port: str, language: str, author: str, work: str):
        super().__init__(port)
        self.language = language
        self.author = author
        self.work = work

    def annotate(self, text: str, annotation_fn: callable = prompts.create_translation_messages) -> Optional[Annotation]:
        messages = annotation_fn(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=text,
            speaker=None # @TODO come back for drama
        )
        response = call_model(port=self.port, messages=messages)
        annotation = validator.extract_json_from_annotation(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        valid_annotation = validator.validate_annotation(annotation)
        if not valid_annotation:
            print(f"Invalid annotation for text: {text}\nAnnotation: {annotation}")
            return None
        return Annotation(text=text, annotation=valid_annotation)

