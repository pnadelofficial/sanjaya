from .annotations import WordAnnotator, SentenceAnnotator, Annotation, call_model_with_retry
from . import validator
from sanjaya.utils import load_object

from typing import List, Optional


class CommentAnnotator(SentenceAnnotator):
    """
    Sentence-level commenter annotator that determines if a sentence needs
    comments/notes and generates them with criteria tagging.

    Runs as two separate model calls so the "does this need a comment"
    decision doesn't share a call (and a prompt) with comment composition:

    1. classifier_prompt asks only for a needs_comment boolean. This call
       runs for every sentence. When it says no, annotate() returns None
       immediately — no comment is composed and no second call is made.
    2. writer_prompt composes the comment(s) and only runs on sentences the
       classifier flagged. It returns a JSON array of comment objects (each
       with "comment" and "comment_type"), since a sentence may warrant more
       than one. It may still decline (comment == "None", per the prompt) as
       a final check on the classifier's judgment; if none remain after that
       filter, annotate() returns None.

    Surviving comments are joined into the canonical "summary" field required
    by SentenceAnnotator, and kept in full (text + type) under a "comments"
    key for the detail view.

    output_schema declares "comments" as a list, so a malformed (missing or
    wrongly-shaped) comments field is caught by validate_output_schema()
    before the annotation reaches the cache or the DB.

    Both model calls go through call_model_with_retry(), which retries
    (blind resubmission, up to max_retries times) on malformed JSON or a
    response missing a key declared on the relevant prompt's
    required_keys — see classifier_prompt/writer_prompt in prompts.py.
    """
    role = "comment"
    output_schema = {"comments": list}

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
        classifier_prompt: str = "sanjaya.llm.prompts.comment_classifier_prompt",
        writer_prompt: str = "sanjaya.llm.prompts.comment_writer_prompt",
        max_retries: int = 5,
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key
        self.classifier_prompt = load_object(classifier_prompt)
        self.writer_prompt = load_object(writer_prompt)
        self.max_retries = max_retries

    @staticmethod
    def _extract_writer_json(content: str) -> Optional[str]:
        raw = validator.extract_json_from_annotation(content, regex_pattern=r"\[.*\]")
        if raw is not None:
            return raw
        # Some models ignore the array-wrapping instruction and return a
        # bare object; treat that as a single comment rather than
        # conflating a malformed response with a legitimate "no comment
        # needed" reply (both would otherwise extract to None).
        bare = validator.extract_json_from_annotation(content, regex_pattern=r"\{.*\}")
        return f"[{bare}]" if bare is not None else None

    def _needs_comment(self, sentence: str) -> bool:
        messages = self.classifier_prompt.create_messages(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=sentence,
        )
        parsed = call_model_with_retry(
            self.base_url, messages, self.classifier_prompt.required_keys,
            model=self.model, api_key=self.api_key, max_retries=self.max_retries,
        )
        if not isinstance(parsed, dict):
            print(f"Invalid classifier annotation for text: {sentence}")
            return False
        value = parsed.get("needs_comment")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() == "true"
        return False

    def annotate(self, sentence: str, speaker: Optional[str] = None) -> Optional[Annotation]:
        if not self._needs_comment(sentence):
            return None

        messages = self.writer_prompt.create_messages(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=sentence,
        )
        parsed = call_model_with_retry(
            self.base_url, messages, self.writer_prompt.required_keys,
            model=self.model, api_key=self.api_key, max_retries=self.max_retries,
            extract=self._extract_writer_json,
        )
        if not parsed:
            print(f"Invalid annotation for text: {sentence}")
            return None
        if not isinstance(parsed, list):
            parsed = [parsed]
        comments = [
            c for c in parsed
            if isinstance(c, dict)
            and isinstance(c.get("comment"), str)
            and c["comment"].strip().lower() not in ("", "none")
        ]
        if not comments:
            return None
        annotation = {
            "summary": " ".join(c["comment"] for c in comments),
            "comments": comments,
        }
        return Annotation(text=sentence, annotation=annotation)


class GlossAnnotator(WordAnnotator):
    """
    Word-level gloss annotator backed by an LLM.

    Calls the model once per token and returns an Annotation whose dict
    includes all fields returned by the LLM plus a canonical "label" key
    (mapped from the LLM's "gloss" field) required by WordAnnotator.

    output_schema = None: each token's annotation is a plain gloss string.
    No additional metadata fields are declared or validated beyond "label".

    Uses NLTK word_tokenize by default; override tokenize() for different
    language-specific tokenisation.

    The model call goes through call_model_with_retry(), which retries
    (blind resubmission, up to max_retries times) on malformed JSON or a
    response missing a key declared on prompt.required_keys.
    """
    role = "gloss"
    output_schema = None

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
        prompt: str = "sanjaya.llm.prompts.gloss_prompt",
        max_retries: int = 5,
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key
        self.prompt = load_object(prompt)
        self.max_retries = max_retries

    def annotate(self, sentence: str) -> List[Annotation]:
        tokens = self.tokenize(sentence)
        annotations = []
        for i, token in enumerate(tokens):
            messages = self.prompt.create_messages(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=sentence,
                language_id=i,
                language_word=token,
            )
            annotation = call_model_with_retry(
                self.base_url, messages, self.prompt.required_keys,
                model=self.model, api_key=self.api_key, max_retries=self.max_retries,
            )
            if not annotation:
                print(f"Invalid annotation for token: {token}")
                continue
            annotation["label"] = annotation.get("gloss", "")
            annotations.append(Annotation(text=token, annotation=annotation))
        return annotations


class TranslationAnnotator(SentenceAnnotator):
    """
    Sentence-level translation annotator backed by an LLM.

    When the sentence comes from a line spoken within a TEI <sp> (drama),
    Generator resolves the speaker's name from that <sp>'s <speaker> child
    and passes it into annotate(). In that case play_prompt is used instead
    of prompt, with the speaker's name prefixed onto the sentence so the
    translation reflects who's speaking; for non-drama text (speaker is
    None) prompt is used exactly as before.

    Returns an Annotation whose dict includes all fields returned by the LLM
    plus a canonical "summary" key (mapped from the LLM's "translation" field)
    required by SentenceAnnotator.

    output_schema = None: the sentence annotation is a plain translation string.
    No additional metadata fields are declared or validated beyond "summary".

    The model call goes through call_model_with_retry(), which retries
    (blind resubmission, up to max_retries times) on malformed JSON or a
    response missing a key declared on whichever prompt's required_keys.
    """
    role = "translation"
    output_schema = None

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
        prompt: str = "sanjaya.llm.prompts.translation_prompt",
        play_prompt: str = "sanjaya.llm.prompts.translation_prompt_play",
        max_retries: int = 5,
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key
        self.prompt = load_object(prompt)
        self.play_prompt = load_object(play_prompt)
        self.max_retries = max_retries

    def annotate(self, sentence: str, speaker: Optional[str] = None) -> Optional[Annotation]:
        prompt = self.play_prompt if speaker else self.prompt
        if speaker:
            messages = prompt.create_messages(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=sentence,
                speaker=speaker,
            )
        else:
            messages = prompt.create_messages(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=sentence,
            )
        annotation = call_model_with_retry(
            self.base_url, messages, prompt.required_keys,
            model=self.model, api_key=self.api_key, max_retries=self.max_retries,
        )
        if not annotation:
            print(f"Invalid annotation for text: {sentence}")
            return None
        annotation["summary"] = annotation.get("translation", "")
        return Annotation(text=sentence, annotation=annotation)
