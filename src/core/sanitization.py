"""
Input sanitization (prompt injection defense) and output sanitization (XSS prevention).

Two concerns:
1. **Input**: User-controlled strings that enter the LLM prompt must be stripped
   of control sequences, role-boundary markers, and HTML/markdown injection.
2. **Output**: Any string from the LLM, database, or user echo that appears in
   the JSON API response must be HTML-escaped to prevent stored/reflected XSS
   if the frontend renders it unsafely.
"""

import html
import re
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maximum user query length (defense-in-depth; Pydantic also enforces 500)
_MAX_QUERY_LENGTH = 500

# Maximum length for label-type fields from the LLM (follow-up labels, etc.)
_MAX_LABEL_LENGTH = 200

# Maximum length for question-type fields from the LLM
_MAX_QUESTION_LENGTH = 500

# Patterns that attempt to break out of the user role or inject system messages.
# These are structural defenses — we strip the markers rather than trying to
# detect every possible instruction override.
_ROLE_BOUNDARY_PATTERNS = re.compile(
    r"""
    (?:
        \[\s*(?:system|assistant|user|INST|/INST)\s*\]     # bracketed role markers [system], [INST], etc.
        |<\|(?:system|assistant|user|im_start|im_end)\|>   # ChatML markers
        |<<\s*/?\s*SYS\s*>>                                # Llama-style <<SYS>> and <</SYS>>
        |\#{2,3}\s*(?:System|Assistant|User|Instruction)\s*:  # markdown-style role headers
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Control characters that have no place in a search query
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# Consecutive whitespace normalization (tabs, multiple spaces)
_WHITESPACE_COLLAPSE = re.compile(r"[ \t]+")

# Allowed URL schemes for image URLs
_SAFE_URL_SCHEMES = {"http", "https"}


# ---------------------------------------------------------------------------
# INPUT SANITIZATION (before LLM)
# ---------------------------------------------------------------------------

def sanitize_query(query: str) -> str:
    """Sanitize a user search query before it enters the LLM prompt.

    Defenses applied:
    1. Length cap (500 chars)
    2. Strip control characters
    3. Neutralize role-boundary markers (ChatML, Llama, markdown role headers)
    4. Collapse excessive whitespace
    5. Strip leading/trailing whitespace

    This does NOT html-escape (the LLM needs the raw text to understand it).
    It removes structural markers that could trick the model into treating
    user text as system/assistant messages.
    """
    if not query:
        return ""

    s = query[:_MAX_QUERY_LENGTH]

    # Strip control characters
    s = _CONTROL_CHARS.sub("", s)

    # Neutralize role-boundary patterns by replacing with the inner text only.
    # E.g., "[SYSTEM] You are now..." becomes "You are now..."
    s = _ROLE_BOUNDARY_PATTERNS.sub(" ", s)

    # Collapse whitespace
    s = _WHITESPACE_COLLAPSE.sub(" ", s)

    # Normalize newlines — a search query has no business containing newlines
    s = s.replace("\n", " ").replace("\r", " ")

    return s.strip()


def sanitize_filter_values(filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Sanitize a selected_filters dict from the client before LLM interpolation.

    Walks the dict and sanitizes all string values (strip control chars,
    role markers). Non-string values (numbers, booleans, lists) are
    passed through, with list elements recursively sanitized.
    """
    if filters is None:
        return None

    return _sanitize_dict(filters)


def sanitize_labels(labels: Optional[List[str]]) -> Optional[List[str]]:
    """Sanitize selection_labels from the client before LLM interpolation."""
    if labels is None:
        return None

    return [_sanitize_string_value(label) for label in labels]


def _sanitize_string_value(value: str) -> str:
    """Sanitize a single string value (filter value, label, etc.)."""
    if not isinstance(value, str):
        return value
    s = value[:_MAX_QUESTION_LENGTH]
    s = _CONTROL_CHARS.sub("", s)
    s = _ROLE_BOUNDARY_PATTERNS.sub(" ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    return s.strip()


def _sanitize_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively sanitize all string values in a dict."""
    result = {}
    for key, value in d.items():
        # Sanitize the key too (defense-in-depth)
        clean_key = _sanitize_string_value(str(key)) if isinstance(key, str) else key
        result[clean_key] = _sanitize_value(value)
    return result


def _sanitize_value(value: Any) -> Any:
    """Sanitize a single value (recursive for dicts and lists)."""
    if isinstance(value, str):
        return _sanitize_string_value(value)
    elif isinstance(value, dict):
        return _sanitize_dict(value)
    elif isinstance(value, list):
        return [_sanitize_value(item) for item in value]
    else:
        # numbers, booleans, None — pass through
        return value


# ---------------------------------------------------------------------------
# OUTPUT SANITIZATION (before API response)
# ---------------------------------------------------------------------------

_HTML_TAG_PATTERN = re.compile(r"<[^>]+>")


def strip_tags(value: Optional[str]) -> Optional[str]:
    """Strip HTML tags from a string while preserving the text content.

    This is for user-visible fields in JSON API responses (follow-up
    questions, product names, brands). It removes <script>, <img>, etc.
    without entity-encoding quotes or ampersands, which would cause
    display artifacts like ``What&#x27;s`` in the frontend.

    Returns None if input is None.
    """
    if value is None:
        return None
    return _HTML_TAG_PATTERN.sub("", str(value))


def escape_html(value: Optional[str]) -> Optional[str]:
    """HTML-escape a string for non-display metadata fields.

    Use ``strip_tags()`` for user-visible text (follow-ups, product names).
    Use this only for internal metadata (timing dict, plan debug fields)
    where entity-encoding is acceptable.

    Returns None if input is None.
    """
    if value is None:
        return None
    return html.escape(str(value), quote=True)


def sanitize_url(url: Optional[str]) -> Optional[str]:
    """Validate and sanitize a URL for safe inclusion in API responses.

    Only allows http:// and https:// schemes. Rejects javascript:, data:, etc.
    Returns None for invalid/dangerous URLs.
    """
    if url is None:
        return None

    url = url.strip()

    # Reject empty
    if not url:
        return None

    # Check scheme
    try:
        colon_pos = url.index(":")
        scheme = url[:colon_pos].lower().strip()
    except ValueError:
        # No scheme — relative URL, probably safe but suspicious for image_url
        return url

    if scheme not in _SAFE_URL_SCHEMES:
        return None

    return url


def escape_product_fields(product: Dict[str, Any]) -> Dict[str, Any]:
    """HTML-escape user-visible string fields in a product dict.

    Escapes: name, brand, article_type, pattern, category_l1/l2,
    and validates image_url scheme.
    """
    _TEXT_FIELDS = (
        "name", "brand", "article_type", "pattern",
        "category_l1", "category_l2", "description",
    )

    for field in _TEXT_FIELDS:
        if field in product and isinstance(product[field], str):
            product[field] = strip_tags(product[field])

    # Validate image URLs
    if "image_url" in product:
        product["image_url"] = sanitize_url(product.get("image_url"))

    if "gallery_images" in product and isinstance(product["gallery_images"], list):
        product["gallery_images"] = [
            sanitize_url(u) for u in product["gallery_images"] if sanitize_url(u) is not None
        ]

    return product


def escape_follow_ups(follow_ups: Optional[List[Any]]) -> Optional[List[Any]]:
    """Strip HTML tags from follow-up question text and option labels.

    Uses strip_tags (not escape_html) so quotes and ampersands display
    naturally — "What's the vibe?" not "What&#x27;s the vibe?".
    Filter dicts are passed through (structured data, not rendered).
    """
    if not follow_ups:
        return follow_ups

    sanitized = []
    for fq in follow_ups:
        # FollowUpQuestion (Pydantic model or dict)
        if hasattr(fq, "question"):
            # Pydantic model
            fq.question = strip_tags(fq.question) or ""
            fq.dimension = strip_tags(fq.dimension) or ""
            if fq.options:
                for opt in fq.options:
                    opt.label = strip_tags(opt.label) or ""
        elif isinstance(fq, dict):
            fq["question"] = strip_tags(fq.get("question", ""))
            fq["dimension"] = strip_tags(fq.get("dimension", ""))
            if "options" in fq:
                for opt in fq["options"]:
                    if isinstance(opt, dict):
                        opt["label"] = strip_tags(opt.get("label", ""))
                    elif hasattr(opt, "label"):
                        opt.label = strip_tags(opt.label) or ""
        sanitized.append(fq)
    return sanitized


def escape_timing_dict(timing: Dict[str, Any]) -> Dict[str, Any]:
    """HTML-escape LLM-sourced plan fields in the timing dict.

    Only escapes the plan_* keys that come from LLM output.
    Numeric timing values are left untouched.
    """
    _PLAN_STRING_KEYS = (
        "plan_algolia_query", "plan_semantic_query", "plan_vibe_brand",
    )
    _PLAN_LIST_KEYS = (
        "plan_semantic_queries", "plan_modes",
    )
    _PLAN_DICT_KEYS = (
        "plan_attributes", "plan_avoid",
    )

    for key in _PLAN_STRING_KEYS:
        if key in timing and isinstance(timing[key], str):
            timing[key] = escape_html(timing[key])

    for key in _PLAN_LIST_KEYS:
        if key in timing and isinstance(timing[key], list):
            timing[key] = [
                escape_html(v) if isinstance(v, str) else v
                for v in timing[key]
            ]

    for key in _PLAN_DICT_KEYS:
        if key in timing and isinstance(timing[key], dict):
            timing[key] = _escape_dict_values(timing[key])

    return timing


def _escape_dict_values(d: Dict[str, Any]) -> Dict[str, Any]:
    """HTML-escape all string values in a dict (recursive for nested dicts/lists)."""
    result = {}
    for key, value in d.items():
        if isinstance(value, str):
            result[key] = escape_html(value)
        elif isinstance(value, list):
            result[key] = [escape_html(v) if isinstance(v, str) else v for v in value]
        elif isinstance(value, dict):
            result[key] = _escape_dict_values(value)
        else:
            result[key] = value
    return result


def escape_autocomplete_highlight(highlighted: Optional[str]) -> Optional[str]:
    """Sanitize Algolia highlight markup, allowing only <em> tags.

    Algolia wraps matched terms in <em>...</em> or <mark>...</mark>
    depending on the API version / configuration.  We normalise both
    to <em> and strip every other HTML tag.
    """
    if highlighted is None:
        return None

    # First, HTML-escape everything
    escaped = html.escape(highlighted, quote=True)

    # Restore <em> / </em>
    escaped = escaped.replace("&lt;em&gt;", "<em>").replace("&lt;/em&gt;", "</em>")

    # Normalise <mark> / </mark> → <em> / </em>
    escaped = escaped.replace("&lt;mark&gt;", "<em>").replace("&lt;/mark&gt;", "</em>")

    return escaped
