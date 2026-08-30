import ast


def hezar_to_text(output):
    if output is None:
        return ""
    if hasattr(output, "text") and not isinstance(output, dict):
        return str(output.text)
    if isinstance(output, dict):
        return str(output.get("text") or output.get("label") or "")
    if isinstance(output, (list, tuple)) and output:
        return hezar_to_text(output[0])
    value = str(output)
    if value.strip().startswith("{"):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return str(parsed.get("text") or "")
        except (SyntaxError, ValueError):
            pass
    return value
