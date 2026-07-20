import importlib


def load_object(dotted_path: str):
    """Import and return the object at a dotted path, e.g. 'sanjaya.llm.prompts.gloss_prompt'."""
    module_path, attr_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, attr_name)
