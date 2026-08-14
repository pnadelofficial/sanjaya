import json_repair
import json
import jsonschema


def extract_json_from_annotation(annotation, regex_pattern=r"\{.*\}"):
    """
    Extracts a JSON value from the annotation text.

    Finds every balanced top-level bracket pair matching the type implied by
    regex_pattern (an array for a pattern containing '[', an object
    otherwise), tracking string-literal state while matching so brackets
    embedded in string content (e.g. "[interpolation]" inside philological
    commentary) don't get mistaken for structural JSON brackets. Among the
    candidate spans, prefers the first one that parses as strict JSON —
    this skips over incidental bracketed text preceding the real payload
    (e.g. a model preamble like "[see note]") — falling back to the very
    first candidate span if none parse, so a genuinely malformed-but-close
    response still reaches validate_annotation()'s repair step.

    Returns the extracted JSON string if any candidate span was found,
    None otherwise.
    """
    open_char, close_char = ("[", "]") if "[" in regex_pattern else ("{", "}")

    def balanced_span(start):
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(annotation)):
            ch = annotation[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == open_char:
                depth += 1
            elif ch == close_char:
                depth -= 1
                if depth == 0:
                    return annotation[start:i + 1]
        return None

    candidates = [
        span for i, ch in enumerate(annotation) if ch == open_char
        for span in [balanced_span(i)] if span is not None
    ]
    if not candidates:
        return None
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except ValueError:
            continue
    return candidates[0]


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