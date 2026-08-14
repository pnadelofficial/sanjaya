from .annotations import WordAnnotator, Annotation
from . import validator
from sanjaya.utils import load_object
from nlp_pipeline.pipeline import NLPPipeline
from typing import List, Optional

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
    
    The output_schema is set to include only the gloss field, with lemma, part_of_speech,
    and morphology fields populated directly from nlp_pipeline's analysis.
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
    ):
        self.base_url = base_url
        self.language = language
        self.author = author
        self.work = work
        self.model = model
        self.api_key = api_key
        self.prompt = load_object(prompt)
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

                # Call the LLM with the prompt for glossing only
                from .annotations import call_model
                response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
                raw = validator.extract_json_from_annotation(
                    response.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                annotation = validator.validate_annotation(raw)
                if not annotation:
                    print(f"Invalid annotation for token: {word.text}\nAnnotation: {raw}")
                    continue

                # Retrieve nlp_pipeline annotations
                nlp_annotations = {
                    NLP_FIELD_ALIASES.get(nlp_class, nlp_class): getattr(word, nlp_class)
                    for nlp_class in NLP_CLASSES
                }

                # Combine nlp_pipeline's linguistic features with LLM's gloss
                combined_annotation = nlp_annotations | annotation

                # Set label as required by WordAnnotator
                combined_annotation["label"] = combined_annotation["gloss"]

                annotations.append(Annotation(text=word.text, annotation=combined_annotation))

        return annotations
