"""
Pinterest Style Discovery -- Gradio UI.

Connects to Pinterest via OAuth, extracts visual taste from pin images
using FashionCLIP, then finds similar products in the database via
pgvector cosine similarity.

All Pinterest state (token, selection) is kept **in-memory** so no
``user_oauth_tokens`` table is required.  Supabase is only used for the
final pgvector product-similarity query.

Tabs:
  1. Pinterest Connection  -- OAuth login + manual token fallback
  2. Board & Section Picker -- choose which boards to analyse
  3. Pin Preview            -- gallery of fetched pins with shopping-intent scores
  4. Style Match            -- extract taste vector, show matching DB products

Usage:
    PYTHONPATH=src python scripts/pinterest_style_gradio.py

    Gradio UI  -> http://localhost:7860
    OAuth callback -> http://localhost:7861/callback

    Make sure PINTEREST_REDIRECT_URI is set to http://localhost:7861/callback
    in your .env (and in the Pinterest developer console).
"""

from __future__ import annotations

import html as html_lib
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse

import gradio as gr
import numpy as np
import requests as http_requests

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
from integrations.pinterest_signals import score_collection_intent
from integrations.pinterest_style import (
    extract_pin_image_urls,
    extract_pin_preview,
    get_pinterest_style_extractor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CALLBACK_PORT = int(os.getenv("PINTEREST_CALLBACK_PORT", "7861"))
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
DEFAULT_USER_ID = "gradio-pinterest-user"
DEFAULT_MAX_PINS = 120
DEFAULT_MAX_IMAGES = 60
DEFAULT_MATCH_COUNT = 80

# ---------------------------------------------------------------------------
# In-memory state (single-user dev tool -- no DB needed for tokens)
# ---------------------------------------------------------------------------
_access_token: Optional[str] = None  # Pinterest access token
_token_source: Optional[str] = None  # "oauth" | "manual" | "env"
_token_obtained_at: Optional[float] = None
_oauth_result: Dict[str, Any] = {}  # filled by callback server

_saved_selection: Dict[str, Any] = {}  # board/section selection
_cached_boards: List[Dict[str, Any]] = []
_cached_pins: List[Dict[str, Any]] = []
_cached_taste_vector: Optional[List[float]] = None


# ============================================================================
# Lightweight Pinterest API helpers (no DB, token passed explicitly)
# ============================================================================

def _pinterest_get(path: str, access_token: str, params: Optional[Dict] = None) -> Dict:
    """GET request to Pinterest v5 API."""
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
    """Exchange OAuth authorization code for access token."""
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
        raise RuntimeError(f"Pinterest token exchange failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _build_auth_url(state: str) -> str:
    settings = get_settings()
    params = {
        "response_type": "code",
        "client_id": settings.pinterest_app_id,
        "redirect_uri": settings.pinterest_redirect_uri,
        "scope": ",".join(settings.pinterest_scopes),
        "state": state,
    }
    return f"{settings.pinterest_auth_url}?{urlencode(params)}"


def _pagination_params(page_size: int, bookmark: Optional[str] = None) -> Dict:
    p: Dict[str, Any] = {"page_size": page_size}
    if bookmark:
        p["bookmark"] = bookmark
    return p


def _fetch_all_pages(path: str, access_token: str, max_items: int, page_size: int = 100) -> List[Dict]:
    """Paginate through a Pinterest list endpoint."""
    items: List[Dict] = []
    bookmark: Optional[str] = None
    while len(items) < max_items:
        resp = _pinterest_get(path, access_token, _pagination_params(min(page_size, max_items), bookmark))
        page = resp.get("items") or []
        items.extend(page)
        bookmark = resp.get("bookmark")
        if not bookmark or not page:
            break
    return items[:max_items]


def _get_access_token() -> str:
    """Return the current in-memory token or the env fallback."""
    global _access_token, _token_source, _token_obtained_at
    if _access_token:
        return _access_token
    # Try env-var fallback
    env_token = get_settings().pinterest_access_token
    if env_token:
        _access_token = env_token
        _token_source = "env"
        _token_obtained_at = time.time()
        return _access_token
    raise RuntimeError("Pinterest not connected. Go to Tab 1 to connect.")


def _is_configured() -> bool:
    s = get_settings()
    return bool(s.pinterest_app_id and s.pinterest_app_secret and s.pinterest_redirect_uri)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
/* ---------- general ---------- */
.status-connected { color: #059669; font-weight: 600; }
.status-disconnected { color: #dc2626; font-weight: 600; }
.status-expired { color: #d97706; font-weight: 600; }
.section-header { font-size: 15px; font-weight: 600; margin: 12px 0 6px; }
.stats-bar {
    display: flex; gap: 18px; padding: 10px 14px;
    background: #f0f4ff; border-radius: 10px; margin: 8px 0;
    font-size: 13px; flex-wrap: wrap;
}
.stats-bar b { color: #4338ca; }

/* ---------- board cards ---------- */
.board-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
    gap: 12px; margin: 10px 0;
}
.board-card {
    border: 1px solid #e0e0e0; border-radius: 12px;
    padding: 14px; background: #fafafa;
    transition: box-shadow .2s, border-color .2s;
    cursor: default;
}
.board-card:hover { box-shadow: 0 2px 12px rgba(0,0,0,.08); }
.board-card.selected { border-color: #6366f1; background: #eef2ff; }
.board-name { font-weight: 600; font-size: 14px; margin-bottom: 4px; }
.board-meta { color: #666; font-size: 12px; }
.board-intent { font-size: 11px; margin-top: 4px; }
.intent-high { color: #059669; }
.intent-low { color: #888; }

/* ---------- pin gallery ---------- */
.pin-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(155px, 1fr));
    gap: 10px; margin: 10px 0;
}
.pin-card {
    border: 1px solid #e5e7eb; border-radius: 10px;
    overflow: hidden; background: #fff;
    transition: box-shadow .2s;
}
.pin-card:hover { box-shadow: 0 2px 10px rgba(0,0,0,.1); }
.pin-img {
    width: 100%; height: 200px; object-fit: cover;
    display: block;
}
.pin-img-placeholder {
    width: 100%; height: 200px; background: #f3f4f6;
    display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 12px;
}
.pin-body { padding: 8px 10px; }
.pin-title {
    font-size: 12px; font-weight: 500;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.pin-intent {
    font-size: 11px; margin-top: 3px; color: #666;
}
.pin-merchant {
    font-size: 10px; color: #999; margin-top: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ---------- product result cards ---------- */
.product-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 14px; margin: 10px 0;
}
.product-card {
    border: 1px solid #e0e0e0; border-radius: 12px;
    overflow: hidden; background: #fff;
    transition: box-shadow .2s;
}
.product-card:hover { box-shadow: 0 3px 14px rgba(0,0,0,.1); }
.product-img {
    width: 100%; height: 260px; object-fit: cover; display: block;
}
.product-img-placeholder {
    width: 100%; height: 260px; background: #f3f4f6;
    display: flex; align-items: center; justify-content: center;
    color: #aaa; font-size: 13px;
}
.product-body { padding: 10px 12px; }
.product-name {
    font-weight: 600; font-size: 13px; line-height: 1.3;
    margin-bottom: 3px;
    display: -webkit-box; -webkit-line-clamp: 2;
    -webkit-box-orient: vertical; overflow: hidden;
}
.product-brand { color: #6366f1; font-size: 12px; font-weight: 500; }
.product-price { font-size: 13px; margin-top: 3px; }
.product-category { font-size: 11px; color: #888; margin-top: 2px; }
.product-sim {
    font-size: 11px; font-weight: 600; margin-top: 5px;
    padding: 2px 8px; border-radius: 8px; display: inline-block;
}
.sim-high { background: #d1fae5; color: #065f46; }
.sim-mid  { background: #fef3c7; color: #92400e; }
.sim-low  { background: #fee2e2; color: #991b1b; }
.product-tags {
    margin-top: 5px; display: flex; flex-wrap: wrap; gap: 4px;
}
.product-tag {
    display: inline-block; padding: 1px 7px; border-radius: 10px;
    font-size: 10px; background: #eef2ff; color: #4338ca;
}

/* ---------- extraction panel ---------- */
.panel-header {
    font-size: 16px; font-weight: 700; margin-bottom: 8px;
    padding-bottom: 6px; border-bottom: 2px solid #e0e0e0;
}
"""

# ============================================================================
# OAuth Callback Server
# ============================================================================

class _OAuthCallbackHandler(BaseHTTPRequestHandler):
    """Tiny HTTP handler that catches the Pinterest OAuth redirect."""

    def do_GET(self):  # noqa: N802
        global _oauth_result, _access_token, _token_source, _token_obtained_at
        parsed = urlparse(self.path)
        if parsed.path != "/callback":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return

        params = parse_qs(parsed.query)
        code = (params.get("code") or [None])[0]
        state = (params.get("state") or [None])[0]
        error = (params.get("error") or [None])[0]

        if error:
            _oauth_result = {"error": error}
            self._respond(f"Pinterest authorization failed: {error}")
            return

        if not code or not state:
            _oauth_result = {"error": "Missing code or state"}
            self._respond("Missing OAuth code or state in callback URL.")
            return

        # Exchange code for token -- store in memory, no DB
        try:
            settings = get_settings()
            secret = settings.pinterest_oauth_state_secret or settings.supabase_jwt_secret

            from integrations.oauth_state import verify_state
            verify_state(state, secret, settings.pinterest_oauth_state_ttl_seconds)

            token_data = _pinterest_exchange_code(code)
            _access_token = token_data.get("access_token")
            _token_source = "oauth"
            _token_obtained_at = time.time()
            _oauth_result = {"status": "connected"}
            self._respond(
                "Pinterest connected successfully! You can close this tab and return to Gradio."
            )
        except Exception as exc:
            _oauth_result = {"error": str(exc)}
            self._respond(f"Token exchange failed: {exc}")

    def _respond(self, message: str):
        body = f"""<!DOCTYPE html>
<html><head><title>Pinterest OAuth</title>
<style>body{{font-family:system-ui;display:flex;align-items:center;
justify-content:center;height:100vh;margin:0;background:#f9fafb}}
.box{{background:#fff;padding:40px;border-radius:16px;
box-shadow:0 4px 24px rgba(0,0,0,.08);text-align:center;max-width:400px}}
</style></head><body><div class="box"><h2>Pinterest OAuth</h2>
<p>{html_lib.escape(message)}</p></div></body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, format, *args):  # noqa: A002
        pass  # silence request logs


def _start_callback_server():
    """Start the OAuth callback HTTP server in a daemon thread."""
    server = HTTPServer(("0.0.0.0", CALLBACK_PORT), _OAuthCallbackHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


# ============================================================================
# Helper utilities
# ============================================================================

def _esc(text: Any) -> str:
    return html_lib.escape(str(text)) if text else ""


def _sim_class(sim: float) -> str:
    if sim >= 0.45:
        return "sim-high"
    if sim >= 0.30:
        return "sim-mid"
    return "sim-low"


def _intent_class(score: float) -> str:
    return "intent-high" if score >= 0.3 else "intent-low"


# ============================================================================
# Tab 1 -- Pinterest Connection
# ============================================================================

def check_connection(_user_id: str) -> str:
    """Check if Pinterest is connected (in-memory token)."""
    global _oauth_result

    # Consume a pending oauth callback result
    if _oauth_result.get("status") == "connected":
        _oauth_result = {}

    if not _access_token:
        # Try env fallback
        env_token = get_settings().pinterest_access_token
        if env_token:
            # Will be picked up by _get_access_token()
            return '<span class="status-connected">Connected (env token)</span>'
        return '<span class="status-disconnected">Not connected</span>'

    age_min = int((time.time() - (_token_obtained_at or 0)) / 60)
    return (
        f'<span class="status-connected">Connected ({_token_source})</span><br>'
        f'<span style="font-size:12px;color:#666">'
        f"Token age: {age_min} min</span>"
    )


def start_oauth(_user_id: str) -> str:
    """Build the Pinterest OAuth authorize URL and open it in the browser."""
    if not _is_configured():
        return (
            "Pinterest integration is not configured. "
            "Set PINTEREST_APP_ID, PINTEREST_APP_SECRET, and PINTEREST_REDIRECT_URI in .env"
        )

    from integrations.oauth_state import sign_state
    import secrets as _secrets

    settings = get_settings()
    secret = settings.pinterest_oauth_state_secret or settings.supabase_jwt_secret
    uid = _user_id.strip() or DEFAULT_USER_ID
    payload = {"uid": uid, "ts": time.time(), "nonce": _secrets.token_urlsafe(12)}
    state = sign_state(payload, secret)
    auth_url = _build_auth_url(state)

    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    return (
        f'<a href="{_esc(auth_url)}" target="_blank" '
        f'style="word-break:break-all">Click here if your browser did not open</a>'
    )


def connect_manual_token(_user_id: str, access_token_input: str) -> str:
    """Connect Pinterest using a manually pasted access token."""
    global _access_token, _token_source, _token_obtained_at
    token = access_token_input.strip()
    if len(token) < 10:
        return "Token too short -- paste the full Pinterest access token."

    _access_token = token
    _token_source = "manual"
    _token_obtained_at = time.time()
    return '<span class="status-connected">Connected via manual token</span>'


def disconnect_pinterest(_user_id: str) -> str:
    global _access_token, _token_source, _token_obtained_at
    _access_token = None
    _token_source = None
    _token_obtained_at = None
    return '<span class="status-disconnected">Disconnected</span>'


# ============================================================================
# Tab 2 -- Board & Section Picker
# ============================================================================

def fetch_boards(_user_id: str):
    """Fetch boards from Pinterest and return rendered HTML + choice update."""
    global _cached_boards
    try:
        token = _get_access_token()
    except RuntimeError as exc:
        return f"<p style='color:red'>{_esc(str(exc))}</p>", gr.update(choices=[], value=[])

    try:
        boards = _fetch_all_pages("/boards", token, max_items=200)
    except RuntimeError as exc:
        return f"<p style='color:red'>{_esc(str(exc))}</p>", gr.update(choices=[], value=[])

    _cached_boards = boards
    if not boards:
        return "<p>No boards found.</p>", gr.update(choices=[], value=[])

    cards_html = '<div class="board-grid">'
    choices: List[str] = []
    for b in boards:
        bid = b.get("id", "")
        name = b.get("name", "Untitled")
        pin_count = b.get("pin_count", 0)
        privacy = b.get("privacy", "")
        score, signals = score_collection_intent(name)
        intent_cls = _intent_class(score)
        label = f"{name} ({pin_count} pins)"
        choices.append(label)

        cards_html += f"""
        <div class="board-card" data-board-id="{_esc(bid)}">
            <div class="board-name">{_esc(name)}</div>
            <div class="board-meta">{pin_count} pins | {_esc(privacy)}</div>
            <div class="board-intent {intent_cls}">
                Shopping intent: {score:.0%}
                {(' -- ' + ', '.join(signals)) if signals else ''}
            </div>
        </div>"""

    cards_html += "</div>"

    stats = (
        f'<div class="stats-bar">'
        f"<span><b>{len(boards)}</b> boards found</span>"
        f"<span><b>{sum(b.get('pin_count', 0) for b in boards)}</b> total pins</span>"
        f"</div>"
    )
    return stats + cards_html, gr.update(choices=choices, value=[])


def fetch_sections(_user_id: str, selected_boards: List[str]):
    """Fetch sections for the selected boards."""
    if not selected_boards:
        return "<p>Select at least one board first.</p>", gr.update(choices=[], value=[])

    try:
        token = _get_access_token()
    except RuntimeError as exc:
        return f"<p style='color:red'>{_esc(str(exc))}</p>", gr.update(choices=[], value=[])

    board_id_map = {
        f"{b.get('name', 'Untitled')} ({b.get('pin_count', 0)} pins)": b.get("id")
        for b in _cached_boards
    }

    all_sections: List[Dict[str, Any]] = []
    section_choices: List[str] = []

    for label in selected_boards:
        board_id = board_id_map.get(label)
        if not board_id:
            continue
        board_name = label.split(" (")[0]
        try:
            resp = _pinterest_get(f"/boards/{board_id}/sections", token, {"page_size": 100})
            items = resp.get("items") or []
            for s in items:
                s["_board_id"] = board_id
                s["_board_name"] = board_name
                all_sections.append(s)
                sec_label = f"{board_name} / {s.get('name', 'Untitled')}"
                section_choices.append(sec_label)
        except RuntimeError:
            continue

    if not all_sections:
        return (
            "<p>No sections found in the selected boards (or boards have no sections).</p>",
            gr.update(choices=[], value=[]),
        )

    html = '<div class="board-grid">'
    for s in all_sections:
        name = s.get("name", "Untitled")
        board_name = s.get("_board_name", "")
        score, signals = score_collection_intent(name)
        intent_cls = _intent_class(score)
        html += f"""
        <div class="board-card">
            <div class="board-name">{_esc(name)}</div>
            <div class="board-meta">Board: {_esc(board_name)}</div>
            <div class="board-intent {intent_cls}">
                Shopping intent: {score:.0%}
                {(' -- ' + ', '.join(signals)) if signals else ''}
            </div>
        </div>"""
    html += "</div>"

    return html, gr.update(choices=section_choices, value=[])


def save_selection(_user_id: str, selected_boards: List[str], selected_sections: List[str]) -> str:
    """Save the board/section selection in memory."""
    global _saved_selection

    board_id_map = {
        f"{b.get('name', 'Untitled')} ({b.get('pin_count', 0)} pins)": b.get("id")
        for b in _cached_boards
    }
    board_ids = [board_id_map[l] for l in (selected_boards or []) if l in board_id_map]

    # Resolve section labels
    sections: List[Dict[str, str]] = []
    if selected_sections:
        try:
            token = _get_access_token()
        except RuntimeError:
            token = None

        if token:
            boards_needing_sections: Dict[str, str] = {}
            for sec_label in selected_sections:
                if " / " in sec_label:
                    board_name = sec_label.rsplit(" / ", 1)[0]
                    for b in _cached_boards:
                        if b.get("name", "Untitled") == board_name and b.get("id"):
                            boards_needing_sections[board_name] = b["id"]
                            break

            section_map: Dict[str, Dict[str, str]] = {}
            for board_name, board_id in boards_needing_sections.items():
                try:
                    resp = _pinterest_get(f"/boards/{board_id}/sections", token, {"page_size": 100})
                    for s in (resp.get("items") or []):
                        key = f"{board_name} / {s.get('name', 'Untitled')}"
                        section_map[key] = {"board_id": board_id, "section_id": s.get("id")}
                except Exception:
                    continue
            sections = [section_map[l] for l in selected_sections if l in section_map]

    _saved_selection = {"boards": board_ids, "sections": sections}

    n_boards = len(board_ids)
    n_sections = len(sections)
    return (
        f'<span class="status-connected">Selection saved: '
        f"{n_boards} board(s), {n_sections} section(s)</span>"
    )


# ============================================================================
# Tab 3 -- Pin Preview
# ============================================================================

def _fetch_pins_from_selection(token: str, selection: Dict, max_pins: int) -> List[Dict]:
    """Fetch pins using in-memory selection (boards/sections)."""
    sections = selection.get("sections") or []
    board_ids = selection.get("boards") or []

    if sections:
        pins: List[Dict] = []
        for sec in sections:
            bid = sec.get("board_id")
            sid = sec.get("section_id")
            if not bid or not sid:
                continue
            pins.extend(_fetch_all_pages(f"/boards/{bid}/sections/{sid}/pins", token, max_pins - len(pins)))
            if len(pins) >= max_pins:
                break
        return pins[:max_pins]

    if board_ids:
        pins = []
        for bid in board_ids:
            pins.extend(_fetch_all_pages(f"/boards/{bid}/pins", token, max_pins - len(pins)))
            if len(pins) >= max_pins:
                break
        return pins[:max_pins]

    # No selection -- fetch user's recent pins
    return _fetch_all_pages("/pins", token, max_pins)


def fetch_pins_preview(_user_id: str, max_pins: int) -> Tuple[str, str]:
    """Fetch pins using the saved selection and render a preview gallery."""
    global _cached_pins

    max_pins = int(max_pins) if max_pins else DEFAULT_MAX_PINS

    try:
        token = _get_access_token()
        pins = _fetch_pins_from_selection(token, _saved_selection, max_pins)
    except RuntimeError as exc:
        return f"<p style='color:red'>{_esc(str(exc))}</p>", ""

    _cached_pins = pins
    if not pins:
        return "<p>No pins fetched. Check your selection or connection.</p>", ""

    previews = [extract_pin_preview(p) for p in pins]
    image_urls = [pv.get("image_url") for pv in previews if pv.get("image_url")]

    avg_intent = (
        sum(pv.get("shopping_intent_score", 0) for pv in previews) / len(previews)
        if previews
        else 0
    )

    stats = (
        f'<div class="stats-bar">'
        f"<span><b>{len(pins)}</b> pins fetched</span>"
        f"<span><b>{len(image_urls)}</b> with images</span>"
        f"<span>Avg shopping intent: <b>{avg_intent:.0%}</b></span>"
        f"</div>"
    )

    gallery = '<div class="pin-grid">'
    for pv in previews[:200]:
        img_url = pv.get("image_url")
        title = pv.get("title") or ""
        intent = pv.get("shopping_intent_score", 0)
        merchant = pv.get("merchant_domain") or ""
        intent_cls = _intent_class(intent)

        if img_url:
            img_html = f'<img class="pin-img" src="{_esc(img_url)}" loading="lazy" alt="">'
        else:
            img_html = '<div class="pin-img-placeholder">No image</div>'

        gallery += f"""
        <div class="pin-card">
            {img_html}
            <div class="pin-body">
                <div class="pin-title" title="{_esc(title)}">{_esc(title) or '&mdash;'}</div>
                <div class="pin-intent {intent_cls}">Intent: {intent:.0%}</div>
                {'<div class="pin-merchant">' + _esc(merchant) + '</div>' if merchant else ''}
            </div>
        </div>"""
    gallery += "</div>"

    return stats + gallery, json.dumps({"total_pins": len(pins), "with_images": len(image_urls)}, indent=2)


# ============================================================================
# Tab 4 -- Style Extraction & Product Matches
# ============================================================================

def extract_style_and_match(
    _user_id: str,
    max_images: int,
    match_count: int,
    filter_gender: str,
    filter_categories: List[str],
    min_price: float,
    max_price: float,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[str, str, str]:
    """
    Full pipeline:
      1. Use cached pins (or fetch fresh ones)
      2. Encode images with FashionCLIP
      3. Compute mean taste vector
      4. Query pgvector for similar products
    Returns (pins_html, products_html, stats_json).
    """
    global _cached_taste_vector
    max_images = int(max_images) if max_images else DEFAULT_MAX_IMAGES
    match_count = int(match_count) if match_count else DEFAULT_MATCH_COUNT

    # --- Step 1: get pins ---
    progress(0, desc="Fetching pins...")
    if _cached_pins:
        pins = _cached_pins
    else:
        try:
            token = _get_access_token()
            pins = _fetch_pins_from_selection(token, _saved_selection, DEFAULT_MAX_PINS)
        except RuntimeError as exc:
            err = f"<p style='color:red'>Failed to fetch pins: {_esc(str(exc))}</p>"
            return err, "", "{}"

    image_urls = extract_pin_image_urls(pins)
    if not image_urls:
        return "<p>No images found in pins.</p>", "", "{}"

    # --- Step 2: compute taste vector ---
    progress(0.1, desc=f"Encoding {min(max_images, len(image_urls))} images with FashionCLIP...")
    settings = get_settings()
    extractor = get_pinterest_style_extractor()
    taste_vector, embed_stats = extractor.compute_taste_vector(
        image_urls=image_urls,
        max_images=max_images,
        timeout_seconds=settings.pinterest_request_timeout_seconds,
    )

    if taste_vector is None:
        return "<p>Failed to encode any images.</p>", "", json.dumps(embed_stats, indent=2)

    _cached_taste_vector = taste_vector

    # --- Step 3: render sampled pins panel ---
    progress(0.7, desc="Querying database for similar products...")
    pin_sample_html = _render_pin_sample(image_urls, max_images)

    # --- Step 4: pgvector similarity search ---
    products_html, search_stats = _query_similar_products(
        taste_vector,
        match_count=match_count,
        gender=filter_gender.strip() or None,
        categories=[c for c in (filter_categories or []) if c] or None,
        min_price=min_price if min_price and min_price > 0 else None,
        max_price=max_price if max_price and max_price > 0 else None,
    )

    progress(1.0, desc="Done!")

    all_stats = {
        "embedding": embed_stats,
        "taste_vector_dim": len(taste_vector),
        "taste_vector_norm": float(np.linalg.norm(taste_vector)),
        "search": search_stats,
    }

    return pin_sample_html, products_html, json.dumps(all_stats, indent=2)


def _render_pin_sample(image_urls: List[str], max_images: int) -> str:
    """Render a small grid of the pin images used for extraction."""
    urls = image_urls[:max_images]
    html = f'<div class="panel-header">Your Pinterest Pins ({len(urls)} used for style extraction)</div>'
    html += '<div class="pin-grid">'
    for url in urls[:60]:
        html += f"""
        <div class="pin-card">
            <img class="pin-img" src="{_esc(url)}" loading="lazy" alt="">
        </div>"""
    if len(urls) > 60:
        html += f'<div class="pin-card"><div class="pin-img-placeholder">+{len(urls) - 60} more</div></div>'
    html += "</div>"
    return html


def _query_similar_products(
    taste_vector: List[float],
    match_count: int,
    gender: Optional[str] = None,
    categories: Optional[List[str]] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Query Supabase pgvector for products similar to the taste vector."""
    supabase = get_supabase_client()
    vector_str = f"[{','.join(map(str, taste_vector))}]"

    params: Dict[str, Any] = {
        "query_embedding": vector_str,
        "match_count": match_count,
    }
    if gender:
        params["filter_gender"] = gender
    if categories:
        params["filter_categories"] = categories
    if min_price is not None:
        params["min_price"] = float(min_price)
    if max_price is not None:
        params["max_price"] = float(max_price)

    t0 = time.time()
    try:
        result = supabase.rpc("match_products_with_hard_filters", params).execute()
        rows = result.data or []
    except Exception as exc:
        return f"<p style='color:red'>DB query failed: {_esc(str(exc))}</p>", {"error": str(exc)}
    elapsed = time.time() - t0

    search_stats = {
        "query_time_ms": round(elapsed * 1000),
        "results_returned": len(rows),
        "match_count_requested": match_count,
    }

    if not rows:
        return "<p>No matching products found.</p>", search_stats

    html = f'<div class="panel-header">Matching Products ({len(rows)} results in {elapsed:.1f}s)</div>'
    html += '<div class="product-grid">'
    for i, row in enumerate(rows):
        html += _render_product_card(row, rank=i + 1)
    html += "</div>"

    return html, search_stats


def _render_product_card(row: Dict[str, Any], rank: int) -> str:
    """Render a single product card."""
    name = row.get("name") or "Unknown"
    brand = row.get("brand") or ""
    price = row.get("price")
    category = row.get("broad_category") or row.get("category") or ""
    image_url = row.get("hero_image_url") or row.get("primary_image_url")
    similarity = row.get("similarity", 0)
    style_tags = row.get("style_tags") or []
    neckline = row.get("neckline") or ""
    fit = row.get("fit") or ""
    sleeve = row.get("sleeve") or ""

    sim_cls = _sim_class(similarity)

    if image_url:
        img_html = f'<img class="product-img" src="{_esc(image_url)}" loading="lazy" alt="">'
    else:
        img_html = '<div class="product-img-placeholder">No image</div>'

    price_str = f"${price:.2f}" if price else ""

    tags_html = ""
    if style_tags:
        tags_html = '<div class="product-tags">'
        for tag in style_tags[:5]:
            tags_html += f'<span class="product-tag">{_esc(tag)}</span>'
        tags_html += "</div>"

    details = []
    if fit:
        details.append(fit)
    if neckline:
        details.append(neckline)
    if sleeve:
        details.append(sleeve)
    details_str = " | ".join(details)

    return f"""
    <div class="product-card">
        {img_html}
        <div class="product-body">
            <div class="product-brand">{_esc(brand)}</div>
            <div class="product-name" title="{_esc(name)}">{_esc(name)}</div>
            <div class="product-price">{price_str}</div>
            <div class="product-category">{_esc(category)}
                {(' -- ' + _esc(details_str)) if details_str else ''}</div>
            <span class="product-sim {sim_cls}">
                #{rank} &middot; {similarity:.1%} match
            </span>
            {tags_html}
        </div>
    </div>"""


# ============================================================================
# Gradio App
# ============================================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="Pinterest Style Discovery") as app:
        gr.Markdown("# Pinterest Style Discovery\nConnect your Pinterest, extract your visual taste, and find matching products.")

        # Shared user-id across tabs
        user_id = gr.Textbox(
            value=DEFAULT_USER_ID,
            label="User ID",
            info="Identifier used for OAuth state signing (dev/testing)",
            scale=1,
        )

        # ==================================================================
        # Tab 1 -- Connection
        # ==================================================================
        with gr.Tab("1. Connect Pinterest"):
            gr.Markdown("### Connect your Pinterest account")
            with gr.Row():
                with gr.Column(scale=2):
                    connection_status = gr.HTML(value='<span class="status-disconnected">Checking...</span>')
                    check_btn = gr.Button("Refresh Status", variant="secondary", size="sm")
                    check_btn.click(fn=check_connection, inputs=[user_id], outputs=[connection_status])

                with gr.Column(scale=3):
                    oauth_btn = gr.Button("Authorize with Pinterest", variant="primary")
                    oauth_output = gr.HTML()
                    oauth_btn.click(fn=start_oauth, inputs=[user_id], outputs=[oauth_output])

            with gr.Row():
                disconnect_btn = gr.Button("Disconnect Pinterest", variant="stop", size="sm")
                disconnect_output = gr.HTML()
                disconnect_btn.click(fn=disconnect_pinterest, inputs=[user_id], outputs=[disconnect_output])

            # Auto-check connection on page load
            app.load(fn=check_connection, inputs=[user_id], outputs=[connection_status])

        # ==================================================================
        # Tab 2 -- Board Picker
        # ==================================================================
        with gr.Tab("2. Board & Section Picker"):
            gr.Markdown("### Select boards and sections to analyse")
            fetch_boards_btn = gr.Button("Fetch My Boards", variant="primary")

            boards_html = gr.HTML()
            board_choices = gr.CheckboxGroup(label="Select Boards", choices=[])

            fetch_boards_btn.click(
                fn=fetch_boards,
                inputs=[user_id],
                outputs=[boards_html, board_choices],
            )

            with gr.Accordion("Sections (optional -- expand after selecting boards)", open=False):
                fetch_sections_btn = gr.Button("Load Sections for Selected Boards", size="sm")
                sections_html = gr.HTML()
                section_choices = gr.CheckboxGroup(label="Select Sections", choices=[])
                fetch_sections_btn.click(
                    fn=fetch_sections,
                    inputs=[user_id, board_choices],
                    outputs=[sections_html, section_choices],
                )

            save_btn = gr.Button("Save Selection", variant="primary")
            save_output = gr.HTML()
            save_btn.click(
                fn=save_selection,
                inputs=[user_id, board_choices, section_choices],
                outputs=[save_output],
            )

        # ==================================================================
        # Tab 3 -- Pin Preview
        # ==================================================================
        with gr.Tab("3. Pin Preview"):
            gr.Markdown("### Preview pins from your selected boards")
            with gr.Row():
                max_pins_slider = gr.Slider(
                    minimum=10, maximum=300, value=DEFAULT_MAX_PINS, step=10,
                    label="Max Pins to Fetch",
                )
                fetch_pins_btn = gr.Button("Fetch Pins", variant="primary")

            pins_gallery = gr.HTML()
            pins_stats = gr.Code(label="Stats", language="json")

            fetch_pins_btn.click(
                fn=fetch_pins_preview,
                inputs=[user_id, max_pins_slider],
                outputs=[pins_gallery, pins_stats],
            )

        # ==================================================================
        # Tab 4 -- Style Extraction & Product Matches
        # ==================================================================
        with gr.Tab("4. Style Match"):
            gr.Markdown(
                "### Extract your visual taste and find matching products\n"
                "This encodes your Pinterest images with FashionCLIP, computes a taste vector, "
                "and queries the product database for the closest matches."
            )

            with gr.Row():
                max_images_slider = gr.Slider(
                    minimum=5, maximum=150, value=DEFAULT_MAX_IMAGES, step=5,
                    label="Max Images to Encode",
                )
                match_count_slider = gr.Slider(
                    minimum=10, maximum=200, value=DEFAULT_MATCH_COUNT, step=10,
                    label="Max Products to Return",
                )

            with gr.Accordion("Filters (optional)", open=False):
                with gr.Row():
                    filter_gender = gr.Textbox(label="Gender", placeholder="e.g. female", value="")
                    filter_categories = gr.CheckboxGroup(
                        label="Broad Categories",
                        choices=["tops", "bottoms", "dresses", "outerwear"],
                    )
                with gr.Row():
                    filter_min_price = gr.Number(label="Min Price ($)", value=0, minimum=0)
                    filter_max_price = gr.Number(label="Max Price ($)", value=0, minimum=0)

            extract_btn = gr.Button("Extract My Style & Find Matches", variant="primary", size="lg")

            with gr.Row():
                with gr.Column(scale=1):
                    pins_panel = gr.HTML(label="Your Pins")
                with gr.Column(scale=1):
                    products_panel = gr.HTML(label="Matching Products")

            match_stats = gr.Code(label="Pipeline Stats", language="json")

            extract_btn.click(
                fn=extract_style_and_match,
                inputs=[
                    user_id,
                    max_images_slider,
                    match_count_slider,
                    filter_gender,
                    filter_categories,
                    filter_min_price,
                    filter_max_price,
                ],
                outputs=[pins_panel, products_panel, match_stats],
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
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
