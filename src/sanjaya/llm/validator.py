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


def validate_output_schema(
    annotation: dict,
    schema: dict[str, type] | None,
    required_key: str,
) -> bool:
    """
    Validate a parsed annotation dict against an annotator's output_schema.

    Always checks that required_key ("label" or "summary") is present and a
    non-empty string. If schema is a dict, also checks that every declared
    field is present with the correct type. Extra fields are always permitted.

    Returns True if valid, False (with a diagnostic print) otherwise.
    """
    if not isinstance(annotation, dict):
        print(f"Annotation is not a dict: {annotation!r}")
        return False
    value = annotation.get(required_key)
    if not isinstance(value, str) or not value:
        print(f"Annotation missing required non-empty string '{required_key}': {annotation}")
        return False
    if schema is None:
        return True
    for field, expected_type in schema.items():
        field_value = annotation.get(field)
        if field_value is None:
            print(f"Annotation missing declared field '{field}': {annotation}")
            return False
        if not isinstance(field_value, expected_type):
            print(
                f"Annotation field '{field}' expected {expected_type.__name__}, "
                f"got {type(field_value).__name__}: {annotation}"
            )
            return False
    return True