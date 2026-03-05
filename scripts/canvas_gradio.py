"""
POSE Canvas — Gradio UI.

Full visual-inspiration experience: upload images, paste URLs, sync Pinterest
pins, view extracted style elements, and run complete-the-fit from any
inspiration image.

Tabs:
  1. Add Inspiration   -- upload an image file or paste a URL
  2. Pinterest Sync    -- connect Pinterest & bulk-import pins
  3. My Canvas         -- gallery of all inspirations, delete, see labels
  4. Style Elements    -- aggregated style profile mapped to feed params
  5. Complete the Fit  -- pick an inspiration → closest product → outfit

Uses the canvas service layer directly (no running API server required).

Usage:
    PYTHONPATH=src python scripts/canvas_gradio.py

    Gradio UI       -> http://localhost:7862
    OAuth callback  -> http://localhost:7861/callback   (shared with pinterest_style_gradio)
"""

from __future__ import annotations

import html as html_lib
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import gradio as gr
import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from config.database import get_supabase_client
from config.settings import get_settings
from canvas.service import CanvasService
from canvas.image_processor import (
    encode_from_bytes,
    encode_from_url,
    classify_style,
    _ATTR_TO_FEED_PARAM,
)
from integrations.pinterest_style import (
    extract_pin_image_urls,
    get_pinterest_style_extractor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CALLBACK_PORT = int(os.getenv("PINTEREST_CALLBACK_PORT", "7861"))
GRADIO_PORT = int(os.getenv("CANVAS_GRADIO_PORT", "7862"))
DEFAULT_USER_ID = "c9eefa63-e152-4515-aff7-4a528a5d9523"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_supabase = None
_service = CanvasService()

# Pinterest OAuth in-memory state (single user dev tool)
_access_token: Optional[str] = None
_token_source: Optional[str] = None
_oauth_result: Dict[str, Any] = {}


def _get_sb():
    global _supabase
    if _supabase is None:
        _supabase = get_supabase_client()
    return _supabase


def _esc(s: str) -> str:
    return html_lib.escape(str(s)) if s else ""


# ============================================================================
# CSS
# ============================================================================
CUSTOM_CSS = """
.status-ok { color: #059669; font-weight: 600; }
.status-err { color: #dc2626; font-weight: 600; }
.status-warn { color: #d97706; font-weight: 600; }

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px; padding: 10px 0;
}
.card {
    border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
    background: #fff; transition: box-shadow .15s;
}
.card:hover { box-shadow: 0 4px 14px rgba(0,0,0,.10); }
.card img {
    width: 100%; height: 240px; object-fit: cover; display: block;
}
.card-body { padding: 10px 12px; }
.card-label {
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 11px; font-weight: 600; margin: 2px 2px 0 0;
}
.label-style { background: #ede9fe; color: #6d28d9; }
.label-source { background: #dbeafe; color: #1d4ed8; }
.label-conf { background: #fef3c7; color: #92400e; }
.card-title {
    font-size: 13px; margin: 6px 0 2px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.card-delete {
    font-size: 11px; color: #dc2626; cursor: pointer;
    text-decoration: underline; margin-top: 4px; display: inline-block;
}

.product-card {
    border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
    background: #fff;
}
.product-card img {
    width: 100%; height: 260px; object-fit: cover; display: block;
}
.product-body { padding: 10px 12px; }
.product-brand { font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: .5px; }
.product-name { font-size: 13px; font-weight: 600; margin: 4px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.product-price { font-size: 14px; font-weight: 700; color: #111827; }
.product-sim { font-size: 11px; display: inline-block; padding: 2px 8px; border-radius: 99px; margin-top: 4px; }
.sim-high { background: #d1fae5; color: #065f46; }
.sim-med  { background: #fef3c7; color: #92400e; }
.sim-low  { background: #fee2e2; color: #991b1b; }
.product-tag { display: inline-block; font-size: 10px; padding: 1px 6px; border-radius: 99px; background: #f3f4f6; color: #374151; margin: 2px 2px 0 0; }

.filter-chip {
    display: inline-block; padding: 4px 12px; border-radius: 99px;
    font-size: 12px; font-weight: 500; margin: 3px 3px 0 0;
    background: #ede9fe; color: #5b21b6;
}
.filter-section { margin: 12px 0; }
.filter-section-title { font-size: 14px; font-weight: 600; color: #374151; margin-bottom: 6px; }

.outfit-section { margin: 16px 0; }
.outfit-section-title { font-size: 16px; font-weight: 700; margin-bottom: 8px; color: #1f2937; }

.stats-panel {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px;
    padding: 12px 16px; margin: 8px 0; font-size: 13px;
}
"""


# ============================================================================
# Tab 1 — Add Inspiration (Upload / URL)
# ============================================================================

def add_from_upload(user_id: str, file) -> Tuple[str, str]:
    """Upload an image file and add it as an inspiration."""
    if file is None:
        return '<span class="status-err">No file selected</span>', ""

    try:
        if hasattr(file, "read"):
            image_bytes = file.read()
            filename = getattr(file, "name", "upload.jpg")
        else:
            # Gradio passes a filepath string
            with open(file, "rb") as f:
                image_bytes = f.read()
            filename = os.path.basename(file)

        sb = _get_sb()
        result = _service.add_inspiration_upload(
            user_id=user_id,
            image_bytes=image_bytes,
            filename=filename,
            supabase=sb,
        )

        html = _render_single_inspiration(result)
        stats = json.dumps({
            "id": result.id,
            "style_label": result.style_label,
            "style_confidence": result.style_confidence,
            "style_attributes": result.style_attributes,
            "source": result.source.value,
        }, indent=2)
        return html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


def add_from_url(user_id: str, url: str, title: str) -> Tuple[str, str]:
    """Fetch an image from URL and add it as an inspiration."""
    if not url or not url.strip():
        return '<span class="status-err">No URL provided</span>', ""

    try:
        sb = _get_sb()
        result = _service.add_inspiration_url(
            user_id=user_id,
            url=url.strip(),
            title=title.strip() or None,
            supabase=sb,
        )

        html = _render_single_inspiration(result)
        stats = json.dumps({
            "id": result.id,
            "style_label": result.style_label,
            "style_confidence": result.style_confidence,
            "style_attributes": result.style_attributes,
            "source": result.source.value,
        }, indent=2)
        return html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


def _render_single_inspiration(insp) -> str:
    """Render a single inspiration result card."""
    img = f'<img src="{_esc(insp.image_url)}" style="max-height:300px;border-radius:10px;">'
    label = insp.style_label or "unknown"
    conf = f"{insp.style_confidence:.0%}" if insp.style_confidence else "n/a"

    attrs_html = ""
    if insp.style_attributes:
        for key, dist in insp.style_attributes.items():
            if isinstance(dist, dict):
                top_items = sorted(dist.items(), key=lambda x: -x[1])[:5]
                chips = " ".join(
                    f'<span class="filter-chip">{_esc(v)} ({s:.0%})</span>'
                    for v, s in top_items
                )
                feed_param = _ATTR_TO_FEED_PARAM.get(key, key)
                attrs_html += f'<div class="filter-section"><div class="filter-section-title">{_esc(feed_param)}</div>{chips}</div>'

    return f"""
    <div style="display:flex;gap:20px;flex-wrap:wrap;">
        <div>{img}</div>
        <div style="flex:1;min-width:250px;">
            <h3>Style: <span class="label-style card-label">{_esc(label)}</span>
                <span class="label-conf card-label">{conf}</span></h3>
            {attrs_html}
        </div>
    </div>
    """


# ============================================================================
# Tab 2 — Pinterest Sync
# ============================================================================

def _pinterest_get(path: str, access_token: str, params=None) -> Dict:
    import requests as http_requests
    settings = get_settings()
    url = f"{settings.pinterest_api_base_url}{path}"
    resp = http_requests.get(
        url,
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
        timeout=settings.pinterest_request_timeout_seconds,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Pinterest API error ({resp.status_code}): {resp.text}")
    return resp.json()


def _pinterest_exchange_code(code: str) -> Dict[str, Any]:
    import requests as http_requests
    settings = get_settings()
    url = f"{settings.pinterest_api_base_url}/oauth/token"
    data: Dict[str, Any] = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": settings.pinterest_redirect_uri,
    }
    if settings.pinterest_continuous_refresh:
        data["continuous_refresh"] = "true"
    resp = http_requests.post(
        url,
        data=data,
        auth=(settings.pinterest_app_id, settings.pinterest_app_secret),
        timeout=settings.pinterest_request_timeout_seconds,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"Token exchange failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _get_access_token() -> str:
    global _access_token, _token_source
    if _access_token:
        return _access_token
    env_token = get_settings().pinterest_access_token
    if env_token:
        _access_token = env_token
        _token_source = "env"
        return _access_token
    raise RuntimeError("Pinterest not connected. Use the Connect button first.")


def check_pinterest() -> str:
    global _access_token, _token_source, _oauth_result
    # Check if OAuth callback arrived
    if _oauth_result and not _access_token:
        if "code" in _oauth_result:
            try:
                token_data = _pinterest_exchange_code(_oauth_result["code"])
                _access_token = token_data.get("access_token")
                _token_source = "oauth"
            except Exception as exc:
                return f'<span class="status-err">Token exchange failed: {_esc(str(exc))}</span>'
    try:
        token = _get_access_token()
        user = _pinterest_get("/user_account", token)
        username = user.get("username", "unknown")
        return f'<span class="status-ok">Connected as @{_esc(username)} (via {_token_source})</span>'
    except Exception as exc:
        return f'<span class="status-err">Not connected: {_esc(str(exc))}</span>'


def start_oauth() -> str:
    settings = get_settings()
    if not settings.pinterest_app_id:
        return '<span class="status-err">Pinterest not configured (missing PINTEREST_APP_ID)</span>'

    state = f"canvas-{int(time.time())}"
    params = {
        "response_type": "code",
        "client_id": settings.pinterest_app_id,
        "redirect_uri": settings.pinterest_redirect_uri,
        "scope": ",".join(settings.pinterest_scopes),
        "state": state,
    }
    auth_url = f"{settings.pinterest_auth_url}?{urlencode(params)}"
    return (
        f'<a href="{_esc(auth_url)}" target="_blank" '
        f'style="font-size:15px;font-weight:600;">Click here to authorize Pinterest</a>'
        f'<br><small>After authorizing, click "Refresh Status" above.</small>'
    )


def disconnect_pinterest() -> str:
    global _access_token, _token_source, _oauth_result
    _access_token = None
    _token_source = None
    _oauth_result = {}
    return '<span class="status-warn">Disconnected</span>'


def sync_pins(user_id: str, max_pins: int, progress=gr.Progress()) -> Tuple[str, str]:
    """Fetch pins from Pinterest and add them as inspirations."""
    try:
        token = _get_access_token()
    except Exception as exc:
        return f'<span class="status-err">{_esc(str(exc))}</span>', ""

    progress(0.1, desc="Fetching pins from Pinterest...")

    try:
        # Fetch all pins (no board selection for simplicity)
        items: List[Dict] = []
        bookmark = None
        while len(items) < int(max_pins):
            params: Dict[str, Any] = {"page_size": min(100, int(max_pins))}
            if bookmark:
                params["bookmark"] = bookmark
            resp = _pinterest_get("/pins", token, params)
            page = resp.get("items") or []
            items.extend(page)
            bookmark = resp.get("bookmark")
            if not bookmark or not page:
                break

        items = items[:int(max_pins)]
        if not items:
            return '<span class="status-warn">No pins found</span>', ""

        progress(0.3, desc=f"Fetched {len(items)} pins. Encoding images...")

        # Use the service to add them
        sb = _get_sb()
        existing = _service._existing_pinterest_pin_ids(user_id, sb)

        added_count = 0
        skipped = 0
        failed = 0
        extractor = get_pinterest_style_extractor()

        for i, pin in enumerate(items):
            pid = pin.get("id")
            if not pid or pid in existing:
                skipped += 1
                continue

            image_urls = extract_pin_image_urls([pin])
            if not image_urls:
                skipped += 1
                continue

            progress(0.3 + 0.65 * (i / len(items)),
                     desc=f"Encoding pin {i+1}/{len(items)}...")

            try:
                embedding = encode_from_url(image_urls[0])
                style_label, style_confidence, style_attrs = classify_style(
                    embedding, sb, nearest_k=get_settings().canvas_style_nearest_k,
                )

                _service._insert_inspiration(
                    supabase=sb,
                    user_id=user_id,
                    source=_InspirationSource_pinterest(),
                    image_url=image_urls[0],
                    embedding=embedding,
                    style_label=style_label,
                    style_confidence=style_confidence,
                    style_attributes=style_attrs,
                    original_url=pin.get("link"),
                    title=pin.get("title"),
                    pinterest_pin_id=pid,
                )
                added_count += 1
                existing.add(pid)
            except Exception:
                failed += 1
                continue

        # Recompute taste vector once
        if added_count > 0:
            _service.recompute_taste_vector(user_id, sb)

        progress(1.0, desc="Done!")

        stats = json.dumps({
            "pins_fetched": len(items),
            "added": added_count,
            "skipped_duplicate": skipped,
            "failed": failed,
        }, indent=2)

        return (
            f'<span class="status-ok">Added {added_count} new inspirations '
            f'(skipped {skipped}, failed {failed})</span>',
            stats,
        )

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


def _InspirationSource_pinterest():
    from canvas.models import InspirationSource
    return InspirationSource.pinterest


# ============================================================================
# Tab 3 — My Canvas
# ============================================================================

def load_canvas(user_id: str) -> Tuple[str, str]:
    """Load and display all inspirations."""
    try:
        sb = _get_sb()
        items = _service.list_inspirations(user_id, sb)
        if not items:
            return '<p style="color:#6b7280;">No inspirations yet. Add some in Tab 1 or 2!</p>', ""

        html = f'<p><b>{len(items)} inspirations</b></p>'
        html += '<div class="card-grid">'
        for insp in items:
            label = insp.style_label or "?"
            conf = f"{insp.style_confidence:.0%}" if insp.style_confidence else ""
            title = insp.title or ""

            html += f"""
            <div class="card">
                <img src="{_esc(insp.image_url)}" loading="lazy" alt="">
                <div class="card-body">
                    <div class="card-title" title="{_esc(title)}">{_esc(title or 'Untitled')}</div>
                    <span class="card-label label-style">{_esc(label)}</span>
                    {f'<span class="card-label label-conf">{conf}</span>' if conf else ''}
                    <span class="card-label label-source">{insp.source.value}</span>
                    <br><small style="color:#9ca3af;">{insp.id[:8]}...</small>
                </div>
            </div>"""
        html += "</div>"

        stats = json.dumps({
            "total": len(items),
            "by_source": _count_by(items, lambda x: x.source.value),
            "by_style": _count_by(items, lambda x: x.style_label or "unknown"),
        }, indent=2)

        return html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


def delete_inspiration(user_id: str, inspiration_id: str) -> Tuple[str, str]:
    """Delete an inspiration by ID."""
    if not inspiration_id or not inspiration_id.strip():
        return '<span class="status-err">No ID provided</span>', ""
    try:
        sb = _get_sb()
        result = _service.remove_inspiration(user_id, inspiration_id.strip(), sb)
        if result.deleted:
            msg = f'<span class="status-ok">Deleted. {result.remaining_count} remaining. Taste vector updated: {result.taste_vector_updated}</span>'
        else:
            msg = '<span class="status-warn">Not found (wrong ID or not yours)</span>'
        return msg, ""
    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', ""


def _count_by(items, key_fn) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        k = key_fn(item)
        counts[k] = counts.get(k, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def _get_inspiration_choices(user_id: str) -> List[str]:
    """Return inspiration IDs for the dropdown in Tab 5."""
    try:
        sb = _get_sb()
        items = _service.list_inspirations(user_id, sb)
        choices = []
        for insp in items:
            label = insp.style_label or "?"
            title = insp.title or "Untitled"
            short_title = title[:30] + "..." if len(title) > 30 else title
            choices.append(f"{insp.id}  |  {label}  |  {short_title}")
        return choices
    except Exception:
        return []


# ============================================================================
# Tab 4 — Style Elements
# ============================================================================

def load_style_elements(user_id: str) -> Tuple[str, str]:
    """Load aggregated style elements."""
    try:
        sb = _get_sb()
        result = _service.get_style_elements(user_id, sb)

        if result.inspiration_count == 0:
            return '<p style="color:#6b7280;">No inspirations yet.</p>', ""

        html = f'<p><b>Style profile from {result.inspiration_count} inspirations</b></p>'

        # Suggested filters
        if result.suggested_filters:
            html += '<div class="outfit-section"><div class="outfit-section-title">Suggested Feed Filters</div>'
            html += '<div class="stats-panel"><code>GET /api/recs/v2/feed?'
            params = []
            for param_name, values in result.suggested_filters.items():
                params.append(f"{_esc(param_name)}={_esc(','.join(values))}")
            html += '&amp;'.join(params)
            html += '</code></div></div>'

        # Raw attributes
        if result.raw_attributes:
            html += '<div class="outfit-section"><div class="outfit-section-title">Style Breakdown</div>'
            for key, scores in result.raw_attributes.items():
                feed_param = _ATTR_TO_FEED_PARAM.get(key, key)
                html += f'<div class="filter-section"><div class="filter-section-title">{_esc(feed_param)}</div>'
                for s in scores[:8]:
                    opacity = max(0.3, s.confidence)
                    html += (
                        f'<span class="filter-chip" style="opacity:{opacity:.2f}">'
                        f'{_esc(s.value)} ({s.confidence:.0%})</span>'
                    )
                html += '</div>'
            html += '</div>'

        stats = json.dumps({
            "inspiration_count": result.inspiration_count,
            "suggested_filters": result.suggested_filters,
            "attribute_categories": list(result.raw_attributes.keys()),
        }, indent=2)

        return html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


# ============================================================================
# Tab 5 — Similar Items + Complete the Fit
# ============================================================================

def _parse_inspiration_id(raw: str) -> str:
    """Extract UUID from dropdown format ``uuid  |  style  |  title``."""
    return raw.strip().split("|")[0].strip() if "|" in raw else raw.strip()


def find_similar(user_id: str, inspiration_id: str, count: int, progress=gr.Progress()) -> Tuple[str, str]:
    """Find products visually similar to an inspiration image."""
    if not inspiration_id or not inspiration_id.strip():
        return '<span class="status-err">Select an inspiration first</span>', ""

    inspiration_id = _parse_inspiration_id(inspiration_id)
    sb = _get_sb()

    try:
        progress(0.2, desc="Searching similar products...")
        products = _service.find_similar_products(inspiration_id, user_id, sb, count=int(count))

        if not products:
            return '<span class="status-warn">No similar products found</span>', ""

        progress(1.0, desc="Done!")

        html = f'<p><b>{len(products)} similar products</b></p>'
        html += '<div class="card-grid">'
        for i, prod in enumerate(products):
            html += _render_product_card(prod, rank=i + 1, label=f"#{i + 1}")
        html += '</div>'

        stats = json.dumps({
            "count": len(products),
            "products": [
                {
                    "product_id": str(p.get("product_id", "")),
                    "name": p.get("name"),
                    "brand": p.get("brand"),
                    "similarity": p.get("similarity"),
                }
                for p in products
            ],
        }, indent=2)

        return html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', traceback.format_exc()


def complete_fit(user_id: str, inspiration_id: str, progress=gr.Progress()) -> Tuple[str, str, str]:
    """Find closest product and build outfit from an inspiration."""
    if not inspiration_id or not inspiration_id.strip():
        return '<span class="status-err">Select an inspiration first</span>', "", ""

    inspiration_id = _parse_inspiration_id(inspiration_id)
    sb = _get_sb()

    try:
        progress(0.1, desc="Finding closest product...")

        matched = _service.find_closest_product(inspiration_id, user_id, sb)
        if not matched:
            return '<span class="status-err">Inspiration not found or no match</span>', "", ""

        product_id = str(matched["product_id"])
        matched_html = _render_product_card(matched, rank=0, label="Closest Match")

        progress(0.3, desc="Building outfit...")

        try:
            from services.outfit_engine import get_outfit_engine
            engine = get_outfit_engine()

            outfit_result = engine.build_outfit(
                product_id=product_id,
                items_per_category=6,
                user_id=user_id,
            )

            outfit_html = _render_outfit(outfit_result)
        except Exception as exc:
            outfit_html = f'<p class="status-warn">Outfit engine not available: {_esc(str(exc))}</p>'
            outfit_result = {"error": str(exc)}

        progress(1.0, desc="Done!")

        stats = json.dumps({
            "matched_product_id": product_id,
            "matched_product_name": matched.get("name"),
            "matched_similarity": matched.get("similarity"),
            "outfit_status": outfit_result.get("status", "n/a"),
        }, indent=2)

        return matched_html, outfit_html, stats

    except Exception as exc:
        return f'<span class="status-err">Error: {_esc(str(exc))}</span>', "", traceback.format_exc()


def _render_product_card(row: Dict[str, Any], rank: int = 0, label: str = "") -> str:
    name = row.get("name") or "Unknown"
    brand = row.get("brand") or ""
    price = row.get("price")
    image_url = row.get("hero_image_url") or row.get("primary_image_url")
    similarity = row.get("similarity", 0)
    style_tags = row.get("style_tags") or []

    sim_pct = f"{similarity:.0%}" if similarity else ""
    sim_cls = "sim-high" if similarity and similarity > 0.7 else ("sim-med" if similarity and similarity > 0.5 else "sim-low")
    price_str = f"${price:.2f}" if price else ""

    img_html = f'<img src="{_esc(image_url)}" loading="lazy" alt="">' if image_url else '<div style="height:260px;background:#f3f4f6;display:flex;align-items:center;justify-content:center;">No image</div>'

    tags = " ".join(f'<span class="product-tag">{_esc(t)}</span>' for t in style_tags[:5])
    label_html = f'<div style="font-size:12px;font-weight:700;color:#4f46e5;margin-bottom:4px;">{_esc(label)}</div>' if label else ""

    return f"""
    <div class="product-card" style="max-width:280px;">
        {img_html}
        <div class="product-body">
            {label_html}
            <div class="product-brand">{_esc(brand)}</div>
            <div class="product-name">{_esc(name)}</div>
            <div class="product-price">{price_str}</div>
            <span class="product-sim {sim_cls}">{sim_pct} match</span>
            <div style="margin-top:4px;">{tags}</div>
            <div style="font-size:10px;color:#9ca3af;margin-top:4px;">ID: {_esc(str(row.get('product_id','')))}</div>
        </div>
    </div>
    """


def _render_outfit(result: Dict[str, Any]) -> str:
    """Render the outfit builder result."""
    if result.get("error") and not result.get("source_product"):
        return f'<p class="status-err">{_esc(result["error"])}</p>'

    status = result.get("status", "unknown")
    recs = result.get("recommendations", {})

    if not recs:
        return f'<p>Status: {_esc(status)} — No complementary items found.</p>'

    html = f'<div class="outfit-section-title">Outfit Complements (status: {_esc(status)})</div>'

    for category, cat_data in recs.items():
        items = cat_data.get("items", [])
        if not items:
            continue

        html += f'<div class="outfit-section"><div class="filter-section-title">{_esc(category.title())} ({len(items)} items)</div>'
        html += '<div class="card-grid">'
        for item in items[:6]:
            name = item.get("name") or "Unknown"
            brand = item.get("brand") or ""
            price = item.get("price")
            img = item.get("hero_image_url") or item.get("primary_image_url") or item.get("image_url")
            score = item.get("tattoo_score") or item.get("final_score") or item.get("compatibility_score", 0)

            img_html = f'<img src="{_esc(img)}" loading="lazy">' if img else '<div style="height:180px;background:#f3f4f6;"></div>'
            price_str = f"${price:.2f}" if price else ""
            score_str = f"{score:.2f}" if score else ""

            html += f"""
            <div class="card">
                {img_html}
                <div class="card-body">
                    <div class="product-brand">{_esc(brand)}</div>
                    <div class="card-title">{_esc(name)}</div>
                    <div class="product-price">{price_str}</div>
                    <span class="product-sim sim-med">{score_str}</span>
                </div>
            </div>"""
        html += '</div></div>'

    # Complete outfit summary
    outfit_info = result.get("complete_outfit", {})
    if outfit_info:
        total = outfit_info.get("total_price", 0)
        count = outfit_info.get("item_count", 0)
        html += f'<div class="stats-panel">Complete outfit: {count} items, total ${total:.2f}</div>'

    return html


# ============================================================================
# OAuth Callback Server
# ============================================================================

class _CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global _oauth_result
        parsed = urlparse(self.path)
        if parsed.path == "/callback":
            qs = parse_qs(parsed.query)
            code = qs.get("code", [None])[0]
            state = qs.get("state", [None])[0]
            error = qs.get("error", [None])[0]
            _oauth_result = {"code": code, "state": state, "error": error}

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            if code:
                self.wfile.write(b"<h2>Pinterest connected! Return to the Gradio app.</h2>")
            else:
                msg = f"Error: {error}" if error else "No code received"
                self.wfile.write(f"<h2>{msg}</h2>".encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # suppress noisy logs


def _start_callback_server():
    server = HTTPServer(("0.0.0.0", CALLBACK_PORT), _CallbackHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()


# ============================================================================
# Gradio App
# ============================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="POSE Canvas") as app:
        gr.Markdown("# POSE Canvas\nUpload images, sync Pinterest pins, explore your style, and complete the fit.")

        user_id = gr.Textbox(
            value=DEFAULT_USER_ID, label="User ID",
            info="Simulated user ID for this dev session",
        )

        # ==================================================================
        # Tab 1 — Add Inspiration
        # ==================================================================
        with gr.Tab("1. Add Inspiration"):
            gr.Markdown("### Upload an image or paste a URL")

            with gr.Tabs():
                with gr.Tab("Upload"):
                    upload_file = gr.File(
                        label="Image File",
                        file_types=["image"],
                        type="filepath",
                    )
                    upload_btn = gr.Button("Add from Upload", variant="primary")
                    upload_result = gr.HTML()
                    upload_stats = gr.Code(label="Details", language="json")

                with gr.Tab("URL"):
                    url_input = gr.Textbox(
                        label="Image URL",
                        placeholder="https://example.com/outfit.jpg",
                    )
                    url_title = gr.Textbox(label="Title (optional)", placeholder="My favorite look")
                    url_btn = gr.Button("Add from URL", variant="primary")
                    url_result = gr.HTML()
                    url_stats = gr.Code(label="Details", language="json")

        # ==================================================================
        # Tab 2 — Pinterest Sync
        # ==================================================================
        with gr.Tab("2. Pinterest Sync"):
            gr.Markdown("### Connect Pinterest and import pins as inspirations")

            with gr.Row():
                with gr.Column(scale=2):
                    pin_status = gr.HTML(value='<span class="status-warn">Checking...</span>')
                    pin_check_btn = gr.Button("Refresh Status", variant="secondary", size="sm")
                    pin_check_btn.click(fn=check_pinterest, outputs=[pin_status])

                with gr.Column(scale=2):
                    pin_oauth_btn = gr.Button("Authorize Pinterest", variant="primary")
                    pin_oauth_html = gr.HTML()
                    pin_oauth_btn.click(fn=start_oauth, outputs=[pin_oauth_html])

                with gr.Column(scale=1):
                    pin_disconnect_btn = gr.Button("Disconnect", variant="stop", size="sm")
                    pin_disconnect_html = gr.HTML()
                    pin_disconnect_btn.click(fn=disconnect_pinterest, outputs=[pin_disconnect_html])

            gr.Markdown("---")

            with gr.Row():
                max_pins_slider = gr.Slider(
                    minimum=5, maximum=200, value=60, step=5,
                    label="Max Pins to Import",
                )
                sync_btn = gr.Button("Sync Pins as Inspirations", variant="primary", size="lg")

            sync_result = gr.HTML()
            sync_stats = gr.Code(label="Sync Stats", language="json")

            _sync_event = sync_btn.click(
                fn=sync_pins,
                inputs=[user_id, max_pins_slider],
                outputs=[sync_result, sync_stats],
            )

            app.load(fn=check_pinterest, outputs=[pin_status])

        # ==================================================================
        # Tab 3 — My Canvas
        # ==================================================================
        with gr.Tab("3. My Canvas") as tab_canvas:
            gr.Markdown("### Your inspiration collection")

            refresh_btn = gr.Button("Refresh Canvas", variant="primary")
            canvas_html = gr.HTML()
            canvas_stats = gr.Code(label="Canvas Stats", language="json")
            refresh_btn.click(
                fn=load_canvas,
                inputs=[user_id],
                outputs=[canvas_html, canvas_stats],
            )

            gr.Markdown("---")
            gr.Markdown("#### Delete an Inspiration")
            with gr.Row():
                del_id = gr.Textbox(label="Inspiration ID", placeholder="paste the ID here")
                del_btn = gr.Button("Delete", variant="stop")
            del_result = gr.HTML()
            del_stats = gr.Textbox(label="", visible=False)
            del_btn.click(
                fn=delete_inspiration,
                inputs=[user_id, del_id],
                outputs=[del_result, del_stats],
            )

        # Auto-load canvas when tab is selected
        tab_canvas.select(
            fn=load_canvas,
            inputs=[user_id],
            outputs=[canvas_html, canvas_stats],
        )

        # ==================================================================
        # Tab 4 — Style Elements
        # ==================================================================
        with gr.Tab("4. Style Elements") as tab_style:
            gr.Markdown(
                "### Your aggregated style profile\n"
                "Shows extracted attributes mapped to the feed endpoint's "
                "`include_*` query parameters. Copy the suggested query string "
                "to test with the feed API."
            )

            style_btn = gr.Button("Load Style Profile", variant="primary")
            style_html = gr.HTML()
            style_stats = gr.Code(label="Details", language="json")
            style_btn.click(
                fn=load_style_elements,
                inputs=[user_id],
                outputs=[style_html, style_stats],
            )

        # Auto-load style elements when tab is selected
        tab_style.select(
            fn=load_style_elements,
            inputs=[user_id],
            outputs=[style_html, style_stats],
        )

        # ==================================================================
        # Tab 5 — Similar Items
        # ==================================================================
        with gr.Tab("5. Similar Items") as tab_similar:
            gr.Markdown(
                "### Find products similar to your inspiration\n"
                "Select an inspiration and find visually similar products "
                "from the catalog."
            )

            with gr.Row():
                sim_id = gr.Dropdown(
                    label="Inspiration",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                sim_count = gr.Slider(
                    minimum=4, maximum=30, value=12, step=2,
                    label="Number of results",
                )
                sim_btn = gr.Button("Find Similar", variant="primary", size="lg")

            sim_html = gr.HTML()
            sim_stats = gr.Code(label="Details", language="json")

            sim_btn.click(
                fn=find_similar,
                inputs=[user_id, sim_id, sim_count],
                outputs=[sim_html, sim_stats],
            )

        # Auto-load inspiration choices when tab is selected
        tab_similar.select(
            fn=lambda uid: gr.update(choices=_get_inspiration_choices(uid)),
            inputs=[user_id],
            outputs=[sim_id],
        )

        # ==================================================================
        # Tab 6 — Complete the Fit
        # ==================================================================
        with gr.Tab("6. Complete the Fit") as tab_fit:
            gr.Markdown(
                "### Complete the fit from an inspiration image\n"
                "Select an inspiration from the dropdown. The system finds the "
                "closest real product and builds an outfit around it."
            )

            with gr.Row():
                fit_id = gr.Dropdown(
                    label="Inspiration",
                    choices=[],
                    interactive=True,
                    allow_custom_value=True,
                )
                fit_btn = gr.Button("Complete the Fit", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Closest Product")
                    fit_product_html = gr.HTML()
                with gr.Column(scale=2):
                    gr.Markdown("#### Outfit Recommendations")
                    fit_outfit_html = gr.HTML()

            fit_stats = gr.Code(label="Pipeline Stats", language="json")

            fit_btn.click(
                fn=complete_fit,
                inputs=[user_id, fit_id],
                outputs=[fit_product_html, fit_outfit_html, fit_stats],
            )

        # Auto-load inspiration choices when tab is selected
        tab_fit.select(
            fn=lambda uid: gr.update(choices=_get_inspiration_choices(uid)),
            inputs=[user_id],
            outputs=[fit_id],
        )

        # ==================================================================
        # Chains — after sync / upload / url, auto-refresh canvas
        # ==================================================================
        _sync_event.then(
            fn=load_canvas,
            inputs=[user_id],
            outputs=[canvas_html, canvas_stats],
        )
        upload_btn.click(
            fn=add_from_upload,
            inputs=[user_id, upload_file],
            outputs=[upload_result, upload_stats],
        ).then(
            fn=load_canvas,
            inputs=[user_id],
            outputs=[canvas_html, canvas_stats],
        )
        url_btn.click(
            fn=add_from_url,
            inputs=[user_id, url_input, url_title],
            outputs=[url_result, url_stats],
        ).then(
            fn=load_canvas,
            inputs=[user_id],
            outputs=[canvas_html, canvas_stats],
        )

    return app


# ============================================================================
# Main
# ============================================================================

def main():
    print(f"Starting OAuth callback server on port {CALLBACK_PORT}...")
    _start_callback_server()
    print(f"OAuth callback ready at http://localhost:{CALLBACK_PORT}/callback")
    print()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=GRADIO_PORT,
        share=False,
        show_error=True,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
