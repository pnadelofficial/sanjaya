import re
import json_repair
import json
import jsonschema

def extract_json_from_annotation(annotation, regex_pattern=r"\{.*\}"):
    """
    Extracts JSON content from the annotation using a regular expression pattern.
    Returns the extracted JSON string if found, None otherwise.
    """
    match = re.search(regex_pattern, annotation, re.DOTALL)
    if match:
        return match.group(0)
    return None

def validate_annotation(annotation, json_schema=None):
    """
    Validates the annotation by checking if it can be repaired into valid JSON and optionally against a provided JSON schema.
    Returns True if the annotation is valid, False otherwise.
    """
    try:
        json.loads(annotation)  # Check if it can be parsed as JSON
        repaired_annotation = json_repair.repair_json(annotation)
        if json_schema:
            try:
                jsonschema.validate(instance=json.loads(repaired_annotation), schema=json_schema)  # Validate against schema if provided
                return repaired_annotation  # Return repaired annotation if valid against schema
            except jsonschema.ValidationError as ve:
                print(f"Annotation validation error: {ve}")
                return False
        return json.loads(repaired_annotation)  # Return repaired annotation if valid
    except Exception as e:
        print(f"Annotation validation error: {e}")
        return False