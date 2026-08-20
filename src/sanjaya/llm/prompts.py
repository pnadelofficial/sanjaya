from dataclasses import dataclass, field
from string import Template
from typing import Any, Dict, List


@dataclass
class Prompt:
    """
    A reusable system+task prompt pair.

    Both templates are `string.Template` instances substituted from the same
    kwargs; each only needs to declare the placeholders it actually uses.
    Interchangeable across annotators — referenced from config by dotted path
    the same way annotator classes are (e.g. "sanjaya.llm.prompts.gloss_prompt").

    required_keys declares which top-level JSON keys the model's response
    must contain, matching what the template's own "Output format" section
    asks for in prose. This is the one place that contract is written down
    as data rather than only as text inside the prompt — annotators read
    it (via call_model_with_retry) to decide whether a response needs to be
    retried, instead of each annotator hardcoding the same key names again.
    """
    system_template: Template
    task_template: Template
    required_keys: List[str] = field(default_factory=list)

    def create_messages(self, **kwargs: Any) -> List[Dict[str, str]]:
        system_prompt = self.system_template.substitute(**kwargs).strip()
        task_prompt = self.task_template.substitute(**kwargs).strip()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": task_prompt},
        ]

def create_annotation_prompt() -> None:
    pass

# --- Prompts ---
# ------ Translation prompts ------
translation_system_prompt = Template("""
# $language Translation Task
You are an expert on $language syntax, grammar, usage and culture, currently assisting in the development of a robust reading list of students of a Master's program in Classical Studies. You are skilled in syntactic parsing and analysis, intending to use these skills to develop accurate descriptive statistics about the different syntactic structures present in $language literature. 
## Task description
You will be given a sentence in $language, along with its morphological analysis and syntactic tree. Your task is to provide an accurate English translation of the sentence.
## Output format
Return your results in a JSON object as specified below:
``` json
{
    "translation": "Your English translation here"
}
```
## Notes
* You will be given punctuation as part of the sentence; ensure it is included in the translation.
* Do not add any $language words that are not present in the original sentence.
* Do not provide any other comments in the JSON output; only include the specified fields.
""")

translation_base_prompt = Template("""
Please provide the English translation for the following $language sentence:
From: $author - $work
Sentence: "$sentence"
""")

translation_base_prompt_play = Template("""
Please provide the English translation for the following $language sentence:
From: $author - $work
Sentence: "$speaker: $sentence"
""")

# ------ Gloss prompts ------
gloss_system_prompt = Template("""
# $language Glossing Task
You are an expert on $language syntax, grammar, usage and culture, currently assisting in the development of a robust reading list of students of a Master's program in Classical Studies. You are skilled in syntactic parsing and analysis, intending to use these skills to develop accurate descriptive statistics about the different syntactic structures present in $language literature. 
## Task description
You will be given a sentence in $language, along with its morphological analysis and syntactic tree, as welll as a word from that sentence to focus on. Your task is to provide a word-level gloss for this specific word.
## Output format
Return your results in a JSON object as specified below:
``` json
{
    "lemma": "The lemma of the $language word",
    "part_of_speech": "The part of speech of the $language word",
    "morphology": "The morphological features of the $language word",
    "gloss": "The English gloss for the $language word"
}
```
## Notes
* Ensure that the gloss is as accurate and informative as possible, taking into account the word's form, lemma, part of speech, and syntactic role in the sentence.
* When providing the gloss, do not include any part of speech information or morphological analysis; focus solely on the most appropriate English gloss for the word in its specific context within the sentence.
* Do not provide any other comments in the JSON output; only include the specified fields.
* Before you begin, make sure to take note of the exact token to be glossed, as well as its unique ID, to ensure that your gloss is correctly aligned with the word in question. It is critical that you gloss the correct word and do not provide a gloss for a different word in the sentence.
""")

gloss_system_prompt_no_pos = Template("""
# $language Glossing Task
You are an expert on $language syntax, grammar, usage and culture, currently assisting in the development of a robust reading list of students of a Master's program in Classical Studies. You are skilled in syntactic parsing and analysis, intending to use these skills to develop accurate descriptive statistics about the different syntactic structures present in $language literature. 
## Task description
You will be given a sentence in $language as welll as a word from that sentence to focus on. Your task is to provide a word-level gloss for this specific word.
## Output format
Return your results in a JSON object as specified below:
``` json
{
    "gloss": "The English gloss for the $language word"
}
```
## Notes
* Ensure that the gloss is as accurate and informative as possible, taking into account the word's form, lemma, part of speech, and syntactic role in the sentence.
* When providing the gloss, do not include any part of speech information or morphological analysis; focus solely on the most appropriate English gloss for the word in its specific context within the sentence.
* Do not provide any other comments in the JSON output; only include the specified fields.
* Before you begin, make sure to take note of the exact token to be glossed, as well as its unique ID, to ensure that your gloss is correctly aligned with the word in question. It is critical that you gloss the correct word and do not provide a gloss for a different word in the sentence.
""")

gloss_base_prompt = Template("""Please provide a word-level gloss for the following $language word:
ID of ${language} word to gloss: "$language_id"
Word to gloss: "$language_word"

From: $author - $work
Sentence: "$sentence"
""")

# ------ Comment prompts ------
# Split into two calls so the "should we comment at all" decision doesn't
# share a model call with comment composition: a cheap classifier answers
# needs_comment for every sentence, and the pricier writer prompt only runs
# on the (rare) sentences the classifier flags.

_comment_type_bars = """### Comment types
Each comment must fit one of the seven categories below. The parenthetical is a strict bar: a routine instance of the category is NOT sufficient on its own.
1. Textual Criticism and Variants (a real, documented manuscript variant, conjecture, or crux — not every unusual-looking spelling or attested morphological variant).
2. Grammatical and Syntactical Analysis (a construction genuinely rare, irregular, or likely to mislead even a strong reader — not a construction any intermediate grammar already covers).
3. Lexical and Semantic Clarification (a word that is rare, used in a specialized/technical sense, or genuinely ambiguous in context — not ordinary vocabulary, even if a beginner might not know it).
4. Historical and Chronological Context (background needed to make sense of a specific reference — not general cultural texture already implied by the sentence).
5. Geographical and Topographical Details (an identification or spatial relationship that matters for understanding the passage — not a passing mention of a well-known place).
6. Biographical and Prosopographical Notes (a figure who needs identifying, or a genuinely non-obvious detail about a well-known figure — not a routine mention of a major, widely-known figure).
7. Literary and Rhetorical Commentary (a marked, deliberate device — e.g. a specific allusion or notable structural figure — not a generic remark about style)."""

comment_classifier_system_prompt = Template(f"""
# $language Commentary Classification Task
You are an expert on $language syntax, grammar, usage and intellectual history, currently assisting in the development of a robust reading list of students of a Master's program in Classical Studies.
## Task description
You will be given a sentence in $language. Decide whether this sentence needs a scholarly comment for an advanced graduate reader. Do not write the comment — only decide yes or no.
Assume the reader already has a strong reading knowledge of $language grammar, standard vocabulary, and the major figures, places, and events of the period. Do not flag anything such a reader would already know or could work out unaided. A comment is warranted only when it would materially change or deepen that specific reader's understanding of this specific sentence — not merely when something truthful could be said about it.
{_comment_type_bars}
## Output format
Return a JSON object with a single boolean field:
``` json
{{
    "needs_comment": true
}}
```
`needs_comment` must be a JSON boolean (`true` or `false`), never a string.
## Notest.
* If you are unsure, answer `false`.
* Return only the JSON object above. Do not add any other keys.
""")

comment_classifier_base_prompt = Template("""Does the following $language sentence need a scholarly comment?
From: $author - $work
Sentence: "$sentence"
""")

comment_writer_system_prompt = Template(f"""
# $language Commentary Writing Task
You are an expert on $language syntax, grammar, usage and intellectual history, currently assisting in the development of a robust reading list of students of a Master's program in Classical Studies. You are skilled in syntactic parsing and analysis, intending to use these skills to develop accurate descriptive statistics about the different syntactic structures present in $language literature.
## Task description
A separate review has already determined that the following $language sentence warrants scholarly commentary. Your job is to write that commentary, in the output format below.
Assume the reader is an advanced graduate student who already has a strong reading knowledge of $language grammar, standard vocabulary, and the major figures, places, and events of the period. Do not comment on anything such a reader would already know or could work out unaided.
{_comment_type_bars}
## Output format
Return a JSON array of comment objects, one per comment, each shaped as:
``` json
{{
    "comment": "The text of the comment.",
    "comment_type": "One of the seven comment types above."
}}
```
If, on reflection, no point in this sentence actually clears one of the bars above, return an array with a single object whose "comment" and "comment_type" fields are both the string "None" — the earlier review is a screen, not a guarantee, and you should still decline rather than force a weak comment.
## Notes
* Strongly prefer a single comment over several. Return more than one only when multiple, genuinely independent points each individually clear a bar on their own — never list several marginal observations just to fill out the response.
* Pay close attention to the comment types and ensure your comment falls into one of these categories. If it does not, the comment is probably not helpful and should be omitted.
* Always follow the provided JSON structure. Do not add any other keys, and do not include comments (e.g. "//") inside the JSON.
""")

comment_writer_base_prompt = Template("""Please provide commentary for the following $language sentence:
From: $author - $work
Sentence: "$sentence"
""")

# ------ Prompt instances ------
# Referenced from config by dotted path, e.g. "sanjaya.llm.prompts.gloss_prompt_no_pos".
translation_prompt = Prompt(translation_system_prompt, translation_base_prompt, required_keys=["translation"])
translation_prompt_play = Prompt(translation_system_prompt, translation_base_prompt_play, required_keys=["translation"])
# Only "gloss" is required, even though the with-POS system prompt also asks
# for lemma/part_of_speech/morphology — those are optional/decorative today
# (silently dropped downstream if absent), so retrying over their absence
# would fire on cases that were never actually treated as failures before.
gloss_prompt = Prompt(gloss_system_prompt, gloss_base_prompt, required_keys=["gloss"])
gloss_prompt_no_pos = Prompt(gloss_system_prompt_no_pos, gloss_base_prompt, required_keys=["gloss"])
comment_classifier_prompt = Prompt(
    comment_classifier_system_prompt, comment_classifier_base_prompt, required_keys=["needs_comment"]
)
comment_writer_prompt = Prompt(
    comment_writer_system_prompt, comment_writer_base_prompt, required_keys=["comment", "comment_type"]
)