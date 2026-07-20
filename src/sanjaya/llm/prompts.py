from dataclasses import dataclass
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
    """
    system_template: Template
    task_template: Template

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
You will be given a sentence in $language, along with its morphological analysis and syntactic tree. Your task is to provide:
1. An accurate English translation of the sentence.
2. Contextual notes on usage, cultural references, or any other relevant information that would aid a student in understanding the sentence. Do not focus on the work or the author, per se. For example, do not start your notes with the a phrase like "This sentence is from [work] by [author]...". This will be unhelpful for readers. Rather focus on the sentence itself and its linguistic and stylistic features. Be sure to include any literary devices or stylistic features present in the sentence, including but not limited to chiasmus, asyndeton and anaphora.
## Output format
Return your results in a JSON object as specified below:
``` json
{
    "translation": "Your English translation here",
    "notes": "Your contextual notes here"
}
```
## Notes
* You will be given punctuation as part of the sentence; ensure it is included in the translation.
* Do not add any $language words that are not present in the original sentence.
* Do not provide any other comments in the JSON output; only include the specified fields.
""")

translation_base_prompt = Template("""
Please provide the English translation, word-level glosses, and contextual notes for the following $language sentence:
From: $author - $work
Sentence: "$sentence"
""")


translation_base_prompt_play = Template("""
Please provide the English translation, word-level glosses, and contextual notes for the following $language sentence:
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
You will be given a sentence in $language, along with its morphological analysis and syntactic tree, as welll as a word from that sentence to focus on. Your task is to provide a word-level gloss for this specific word.
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

# ------ Prompt instances ------
# Referenced from config by dotted path, e.g. "sanjaya.llm.prompts.gloss_prompt_no_pos".
translation_prompt = Prompt(translation_system_prompt, translation_base_prompt)
translation_prompt_play = Prompt(translation_system_prompt, translation_base_prompt_play)
gloss_prompt = Prompt(gloss_system_prompt, gloss_base_prompt)
gloss_prompt_no_pos = Prompt(gloss_system_prompt_no_pos, gloss_base_prompt)