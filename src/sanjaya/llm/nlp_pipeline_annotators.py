import stanza

from .annotations import WordAnnotator, Annotation, call_model_with_retry
from sanjaya.utils import load_object
from nlp_pipeline.pipeline import LANG_CONFIGS, LANG_ID_CONFIG, NLPPipeline, TOKENIZER_LANG_CONFIGS
from typing import List, Optional

# Maps the natural-language names used in sanjaya configs (e.g. "language:
# Italian") to stanza's canonical short codes. Codes not already recognized
# by nlp_pipeline's own LANG_ID_CONFIG (see ensure_stanza_language_support
# below) are patched in at runtime rather than left unsupported — e.g.
# Japanese, which nlp_pipeline doesn't ship support for as of this writing.
STANZA_LANGUAGE_CODES = {
    "arabic": "ar",
    "german": "de",
    "english": "en",
    "spanish": "es",
    "persian": "fa",
    "farsi": "fa",
    "french": "fr",
    "ancient greek": "grc",
    "greek": "grc",
    "hebrew": "he",
    "italian": "it",
    "latin": "la",
    "portuguese": "pt",
    "japanese": "ja"
}

# Must match NLPPipeline.get_nlp()/get_tokenizer()'s own default model_dir —
# there's no constructor hook on NLPPipeline to pass this through, so the
# download below has to land in the same place those will later look.
_STANZA_MODEL_DIR = "./stanza_models"


def stanza_language_code(language: str) -> Optional[str]:
    """Map a natural-language name (as used in sanjaya configs) to stanza's
    canonical short code. Case-insensitive; returns None for names outside
    nlp_pipeline's supported language set."""
    return STANZA_LANGUAGE_CODES.get(language.strip().lower())


def ensure_stanza_model(language: str, model_dir: str = _STANZA_MODEL_DIR) -> None:
    """
    Proactively download the stanza model for the given natural-language
    name, so analysis doesn't depend on MultilingualPipeline's own per-text
    language identification picking the right language (it can and does
    misfire on short/ambiguous input).

    Uses whatever package nlp_pipeline's own LANG_CONFIGS declares for this
    language (e.g. "perseus" for grc/la) so the downloaded package matches
    what MultilingualPipeline will actually request at analysis time;
    otherwise stanza's own default package for that language.
    """
    code = stanza_language_code(language)
    if code is None:
        print(
            f"NLPGlossAnnotator: no known stanza language code for language={language!r}; "
            "skipping proactive model download (falling back to stanza's own language detection)."
        )
        return
    package = LANG_CONFIGS.get(code, {}).get("package", "default")
    stanza.download(code, model_dir=model_dir, package=package, verbose=False)


def ensure_stanza_language_support(language: str) -> None:
    """
    Runtime patch, not a real fix: nlp_pipeline hardcodes a fixed
    langid_lang_subset, and MultilingualPipeline will never consider a
    language outside it — no matter which model is downloaded — silently
    misclassifying the text as one of the listed languages instead. That's
    what produced literal "<UNK>" placeholder tokens for Japanese text:
    langid was forced to pick among ar/de/en/es/fa/fr/grc/he/it/la/pt, none
    of which fit.

    This mutates nlp_pipeline.pipeline's module-level LANG_ID_CONFIG /
    TOKENIZER_LANG_CONFIGS dicts in place — the same objects NLPPipeline
    passes into MultilingualPipeline — to add the configured language if
    it's missing, following the exact convention every other listed
    language already uses (tokenize-only for the tokenizer pass; no
    LANG_CONFIGS override, i.e. stanza's own default package/processors,
    for the full analysis pass — same as "it" already does). Idempotent:
    safe to call once per annotator construction.

    The real fix belongs upstream in nlp_pipeline itself; track that
    separately rather than relying on this patch long-term.
    """
    code = stanza_language_code(language)
    if code is None:
        return
    if code not in LANG_ID_CONFIG["langid_lang_subset"]:
        LANG_ID_CONFIG["langid_lang_subset"].append(code)
    if code not in TOKENIZER_LANG_CONFIGS:
        TOKENIZER_LANG_CONFIGS[code] = {"processors": "tokenize"}


NLP_CLASSES = {
    "id",
    "deprel",
    "deps",
    "feats",
    "head",
    "lemma",
    "misc",
    "text",
    "upos",
    "xpos"
}

# Map nlp_pipeline's raw UD attribute names to the field names the rest of
# sanjaya documents/expects (matching GlossAnnotator's LLM-returned fields).
NLP_FIELD_ALIASES = {
    "upos": "part_of_speech",
    "feats": "morphology",
}

class NLPGlossAnnotator(WordAnnotator):
    """
    Word-level gloss annotator that is compliant with nlp_pipeline.
    
    This annotator uses the nlp_pipeline for tokenization and linguistic analysis,
    then applies LLM-based glossing to the tokens identified by nlp_pipeline.

    Unless an nlp_pipeline instance is passed in explicitly, the constructor
    proactively downloads the stanza model matching `language` (see
    ensure_stanza_model / STANZA_LANGUAGE_CODES above) before building the
    default NLPPipeline. This is deliberate: NLPPipeline's MultilingualPipeline
    otherwise re-identifies the language on every single call from the text
    itself, which can and does misfire on short/ambiguous input.

    The output_schema is set to include only the gloss field, with lemma, part_of_speech,
    and morphology fields populated directly from nlp_pipeline's analysis.

    The gloss model call goes through call_model_with_retry(), which
    retries (blind resubmission, up to max_retries times) on malformed JSON
    or a response missing "gloss" (per prompt.required_keys) — checked on
    the model's raw response, before it gets merged with nlp_pipeline's own
    linguistic features below.
    """
    role = "nlp_gloss"
    
    # Define the output schema to include only the gloss field
    # The lemma, part_of_speech, and morphology are populated from nlp_pipeline
    output_schema = {
        "gloss": str
    }

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
        prompt: str = "sanjaya.llm.prompts.gloss_prompt_no_pos",
        nlp_pipeline: Optional[NLPPipeline] = None,
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
        if nlp_pipeline is None:
            # Only patch/download when we're about to build the default
            # NLPPipeline ourselves — a caller-supplied nlp_pipeline is left
            # untouched. Order matters: the language must be recognized
            # before NLPPipeline() builds (and permanently caches) its
            # MultilingualPipeline singletons.
            ensure_stanza_language_support(language)
            ensure_stanza_model(language)
        self.nlp_pipeline = nlp_pipeline or NLPPipeline()

    def annotate(self, sentence: str) -> List[Annotation]:
        # First, use nlp_pipeline to tokenize and analyze the sentence
        analyzed_chunk = self.nlp_pipeline.analyze_str(sentence)
        
        annotations = []
        for token in analyzed_chunk.tokens:
            word_list = token.words
            for word in word_list:
                # Prepare the prompt with minimal information for glossing
                messages = self.prompt.create_messages(
                    language=self.language,
                    author=self.author,
                    work=self.work,
                    sentence=sentence,
                    language_id=word.id,
                    language_word=word.text,
                )

                # Call the LLM with the prompt for glossing only. Retries
                # (and the required-key check) happen against the model's
                # own raw response here, before it's merged with
                # nlp_pipeline's linguistic features below.
                annotation = call_model_with_retry(
                    self.base_url, messages, self.prompt.required_keys,
                    model=self.model, api_key=self.api_key, max_retries=self.max_retries,
                )
                if not annotation:
                    print(f"Invalid annotation for token: {word.text}")
                    continue

                # Retrieve nlp_pipeline annotations
                nlp_annotations = {
                    NLP_FIELD_ALIASES.get(nlp_class, nlp_class): getattr(word, nlp_class)
                    for nlp_class in NLP_CLASSES
                }

                # Combine nlp_pipeline's linguistic features with LLM's gloss
                combined_annotation = nlp_annotations | annotation

                # Set label as required by WordAnnotator
                combined_annotation["label"] = combined_annotation.get("gloss", "")

                annotations.append(Annotation(text=word.text, annotation=combined_annotation))

        return annotations
