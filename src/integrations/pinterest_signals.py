"""Heuristics to detect shopping intent from Pinterest content."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse


_KEYWORD_SCORES = {
    "wishlist": 0.6,
    "wish list": 0.6,
    "to buy": 0.6,
    "shopping": 0.4,
    "shop": 0.4,
    "buy": 0.4,
    "purchase": 0.4,
    "cart": 0.4,
    "haul": 0.3,
    "sale": 0.3,
    "deal": 0.2,
    "gift": 0.2,
    "must have": 0.2,
    "must-have": 0.2,
}

_PRICE_RE = re.compile(r"(?:\$|usd|eur|gbp|£|€)\s*\d", re.IGNORECASE)


def score_text_intent(text: Optional[str]) -> Tuple[float, List[str]]:
    if not text:
        return 0.0, []

    text_lower = text.lower()
    score = 0.0
    signals: List[str] = []

    for keyword, weight in _KEYWORD_SCORES.items():
        if keyword in text_lower:
            score += weight
            signals.append(f"keyword:{keyword}")

    if _PRICE_RE.search(text_lower):
        score += 0.3
        signals.append("price_hint")

    return min(score, 1.0), signals


def extract_merchant_domain(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if not parsed.netloc:
            return None
        return parsed.netloc.lower()
    except Exception:
        return None


def score_pin_intent(pin: Dict[str, Any]) -> Tuple[float, List[str], Optional[str]]:
    title = pin.get("title") or ""
    description = pin.get("description") or ""
    link = pin.get("link") or ""

    score_title, signals_title = score_text_intent(title)
    score_desc, signals_desc = score_text_intent(description)
    score = max(score_title, score_desc)
    signals: List[str] = signals_title + signals_desc

    merchant_domain = extract_merchant_domain(link)
    if merchant_domain and not merchant_domain.endswith("pinterest.com"):
        score += 0.4
        signals.append("merchant_link")

    product_tags = pin.get("product_tags")
    if product_tags:
        score += 0.3
        signals.append("product_tags")

    if pin.get("price") or pin.get("price_value") or pin.get("price_currency"):
        score += 0.3
        signals.append("price_field")

    return min(score, 1.0), signals, merchant_domain


def score_collection_intent(name: Optional[str]) -> Tuple[float, List[str]]:
    return score_text_intent(name)
