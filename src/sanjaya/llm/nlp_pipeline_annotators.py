from .annotations import WordAnnotator, SentenceAnnotator, Annotation
from . import validator
from sanjaya.utils import load_object
from nlp_pipeline.pipeline import NLPPipeline
from nlp_pipeline.types import TokenizableChunk, TokenizedChunk
from typing import List, Optional, Dict, Any
import json

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
        tokenized_chunk = self.nlp_pipeline.tokenize_str(sentence)
        
        annotations = []
        for token in tokenized_chunk.tokens:
            # Get linguistic features directly from nlp_pipeline
            lemma = ""
            part_of_speech = ""
            morphology = ""
            
            if token.words:
                word = token.words[0]
                lemma = word.lemma or ""
                part_of_speech = word.upos or ""
                morphology = word.feats or ""
            
            # Prepare the prompt with minimal information for glossing
            messages = self.prompt.create_messages(
                language=self.language,
                author=self.author,
                work=self.work,
                sentence=sentence,
                language_id=token.identifier,
                language_word=token.text,
            )
            
            # Call the LLM with the prompt for glossing only
            from .annotations import call_model
            response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
            raw = validator.extract_json_from_annotation(
                response.get("choices", [{}])[0].get("message", {}).get("content", "")
            )
            annotation = validator.validate_annotation(raw)
            if not annotation:
                print(f"Invalid annotation for token: {token.text}\nAnnotation: {raw}")
                continue
                
            # Combine nlp_pipeline's linguistic features with LLM's gloss
            combined_annotation = {
                "lemma": lemma,
                "part_of_speech": part_of_speech,
                "morphology": morphology,
                "gloss": annotation.get("gloss", ""),
            }
            
            # Set label as required by WordAnnotator
            combined_annotation["label"] = combined_annotation["gloss"]
            
            annotations.append(Annotation(text=token.text, annotation=combined_annotation))
            
        return annotations


class NLPTranslationAnnotator(SentenceAnnotator):
    """
    Sentence-level translation annotator that is compliant with nlp_pipeline.
    
    This annotator uses nlp_pipeline for sentence analysis and then applies
    LLM-based translation, with enhanced context from nlp_pipeline's analysis.
    
    The output_schema is set to include the standard translation and additional
    linguistic context from nlp_pipeline.
    """
    role = "nlp_translation"
    
    # Define the output schema to match nlp_pipeline's sentence analysis
    output_schema = {
        "translation": str,
        "notes": str,
        "sentence_analysis": dict  # Include additional linguistic analysis
    }

    def __init__(
        self,
        base_url: str,
        language: str,
        author: str,
        work: str,
        model: str = "default",
        api_key: Optional[str] = None,
        prompt: str = "sanjaya.llm.prompts.translation_prompt",
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

    def annotate(self, sentence: str) -> Optional[Annotation]:
        # Use nlp_pipeline to analyze the sentence first
        analyzed_chunk = self.nlp_pipeline.analyze_str(sentence)
        
        # Prepare the prompt with nlp_pipeline's analysis information
        # Extract some key linguistic features for the LLM
        analysis_info = {
            "sentence_structure": self._extract_sentence_structure(analyzed_chunk),
            "key_words": self._extract_key_words(analyzed_chunk),
        }
        
        messages = self.prompt.create_messages(
            language=self.language,
            author=self.author,
            work=self.work,
            sentence=sentence,
            sentence_analysis=analysis_info,
        )
        
        # Call the LLM with enhanced context
        from .annotations import call_model
        response = call_model(self.base_url, messages, model=self.model, api_key=self.api_key)
        raw = validator.extract_json_from_annotation(
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        annotation = validator.validate_annotation(raw)
        if not annotation:
            print(f"Invalid annotation for text: {sentence}\nAnnotation: {raw}")
            return None
            
        # Map the LLM response to match nlp_pipeline's expected format
        annotation["summary"] = annotation.get("translation", "")
        return Annotation(text=sentence, annotation=annotation)
    
    def _extract_sentence_structure(self, chunk: TokenizedChunk) -> Dict[str, Any]:
        """Extract basic sentence structure information from nlp_pipeline analysis."""
        structure = {
            "tokens_count": len(chunk.tokens),
            "language": chunk.lang,
            "token_info": []
        }
        
        for token in chunk.tokens:
            if token.words:
                word = token.words[0]  # Take first word if multiple
                structure["token_info"].append({
                    "text": token.text,
                    "lemma": word.lemma,
                    "upos": word.upos,
                    "feats": word.feats,
                    "deps": word.deps,
                })
        
        return structure
    
    def _extract_key_words(self, chunk: TokenizedChunk) -> List[Dict[str, Any]]:
        """Extract key words from the analyzed sentence."""
        key_words = []
        for token in chunk.tokens:
            if token.words:
                word = token.words[0]
                # Consider nouns, verbs, adjectives, and adverbs as key words
                if word.upos in ['NOUN', 'VERB', 'ADJ', 'ADV']:
                    key_words.append({
                        "text": token.text,
                        "lemma": word.lemma,
                        "upos": word.upos,
                        "feats": word.feats,
                    })
        
        return key_words