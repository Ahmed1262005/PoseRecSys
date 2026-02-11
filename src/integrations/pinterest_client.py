"""Pinterest OAuth + API client helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlencode

import requests

from config.database import get_supabase_client
from config.settings import get_settings


class PinterestApiError(RuntimeError):
    """Raised for Pinterest API failures."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PinterestTokenRecord:
    access_token: str
    refresh_token: Optional[str]
    token_type: Optional[str]
    scope: Optional[str]
    expires_at: Optional[datetime]
    refresh_expires_at: Optional[datetime]


class PinterestClient:
    """Pinterest OAuth client with Supabase-backed token storage."""

    def __init__(self) -> None:
        self._settings = get_settings()
        self._supabase = get_supabase_client()

    def is_configured(self) -> bool:
        return bool(
            self._settings.pinterest_app_id
            and self._settings.pinterest_app_secret
            and self._settings.pinterest_redirect_uri
        )

    # ---------------------------------------------------------------------
    # OAuth helpers
    # ---------------------------------------------------------------------

    def build_authorization_url(self, state: str, scopes: Optional[List[str]] = None) -> str:
        scopes = scopes or self._settings.pinterest_scopes
        params = {
            "response_type": "code",
            "client_id": self._settings.pinterest_app_id,
            "redirect_uri": self._settings.pinterest_redirect_uri,
            "scope": ",".join(scopes),
            "state": state,
        }
        return f"{self._settings.pinterest_auth_url}?{urlencode(params)}"

    def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self._settings.pinterest_redirect_uri,
        }
        if self._settings.pinterest_continuous_refresh:
            data["continuous_refresh"] = "true"

        return self._token_request(data)

    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        if self._settings.pinterest_continuous_refresh:
            data["continuous_refresh"] = "true"

        return self._token_request(data)

    def _token_request(self, data: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self._settings.pinterest_api_base_url}/oauth/token"
        resp = requests.post(
            url,
            data=data,
            auth=(self._settings.pinterest_app_id, self._settings.pinterest_app_secret),
            timeout=self._settings.pinterest_request_timeout_seconds,
        )
        if resp.status_code >= 400:
            raise PinterestApiError(
                f"Pinterest token request failed ({resp.status_code}): {resp.text}",
                status_code=resp.status_code,
            )
        return resp.json()

    # ---------------------------------------------------------------------
    # Token storage
    # ---------------------------------------------------------------------

    def get_token_row(self, user_id: str) -> Optional[Dict[str, Any]]:
        result = (
            self._supabase
            .table("user_oauth_tokens")
            .select("*")
            .eq("user_id", user_id)
            .eq("provider", "pinterest")
            .limit(1)
            .execute()
        )
        if result.data:
            return result.data[0]
        return None

    def get_access_token(self, user_id: str) -> str:
        token_row = self.get_token_row(user_id)
        if not token_row:
            self.ensure_env_token(user_id)
            token_row = self.get_token_row(user_id)
        if not token_row:
            raise PinterestApiError("Pinterest account not connected")
        return self._ensure_access_token(user_id, token_row)

    def upsert_tokens(self, user_id: str, token_data: Dict[str, Any]) -> None:
        record = self._normalize_token_record(token_data)
        payload = {
            "user_id": user_id,
            "provider": "pinterest",
            "access_token": record.access_token,
            "refresh_token": record.refresh_token,
            "token_type": record.token_type,
            "scope": record.scope,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "refresh_expires_at": record.refresh_expires_at.isoformat() if record.refresh_expires_at else None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        existing = self.get_token_row(user_id)
        if existing and not payload.get("refresh_token"):
            payload["refresh_token"] = existing.get("refresh_token")
        if existing and not payload.get("scope"):
            payload["scope"] = existing.get("scope")
        if existing:
            self._supabase.table("user_oauth_tokens").update(payload).eq("id", existing["id"]).execute()
        else:
            payload["created_at"] = datetime.now(timezone.utc).isoformat()
            self._supabase.table("user_oauth_tokens").insert(payload).execute()

    def delete_tokens(self, user_id: str) -> None:
        self._supabase.table("user_oauth_tokens").delete().eq("user_id", user_id).eq("provider", "pinterest").execute()

    def update_sync_status(self, user_id: str, count: int, error: Optional[str] = None) -> None:
        payload: Dict[str, Any] = {
            "last_sync_at": datetime.now(timezone.utc).isoformat(),
            "last_sync_count": count,
            "last_sync_error": error,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._supabase.table("user_oauth_tokens").update(payload).eq("user_id", user_id).eq("provider", "pinterest").execute()

    # ---------------------------------------------------------------------
    # Pinterest API
    # ---------------------------------------------------------------------

    def list_pins(
        self,
        access_token: str,
        page_size: int = 100,
        bookmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._get(
            access_token=access_token,
            path="/pins",
            params=_build_pagination_params(page_size, bookmark),
            error_prefix="Pinterest list pins failed",
        )

    def list_boards(
        self,
        access_token: str,
        page_size: int = 100,
        bookmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._get(
            access_token=access_token,
            path="/boards",
            params=_build_pagination_params(page_size, bookmark),
            error_prefix="Pinterest list boards failed",
        )

    def list_board_sections(
        self,
        access_token: str,
        board_id: str,
        page_size: int = 100,
        bookmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._get(
            access_token=access_token,
            path=f"/boards/{board_id}/sections",
            params=_build_pagination_params(page_size, bookmark),
            error_prefix="Pinterest list board sections failed",
        )

    def list_board_pins(
        self,
        access_token: str,
        board_id: str,
        page_size: int = 100,
        bookmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._get(
            access_token=access_token,
            path=f"/boards/{board_id}/pins",
            params=_build_pagination_params(page_size, bookmark),
            error_prefix="Pinterest list board pins failed",
        )

    def list_section_pins(
        self,
        access_token: str,
        board_id: str,
        section_id: str,
        page_size: int = 100,
        bookmark: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._get(
            access_token=access_token,
            path=f"/boards/{board_id}/sections/{section_id}/pins",
            params=_build_pagination_params(page_size, bookmark),
            error_prefix="Pinterest list section pins failed",
        )

    def fetch_pins(
        self,
        user_id: str,
        max_pins: int,
        selection: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        token_row = self.get_token_row(user_id)
        if not token_row:
            raise PinterestApiError("Pinterest account not connected")

        access_token = self._ensure_access_token(user_id, token_row)
        selection = selection or {}

        board_ids = selection.get("boards") or []
        sections = selection.get("sections") or []

        if sections:
            return self._fetch_from_sections(access_token, sections, max_pins)
        if board_ids:
            return self._fetch_from_boards(access_token, board_ids, max_pins)

        return self._fetch_all_pins(user_id, access_token, token_row, max_pins)

    # ---------------------------------------------------------------------
    # Selection helpers
    # ---------------------------------------------------------------------

    def get_selection(self, user_id: str) -> Dict[str, Any]:
        metadata = self._get_metadata(user_id)
        selection = metadata.get("pinterest_selection")
        if isinstance(selection, dict):
            return selection
        return {}

    def get_metadata(self, user_id: str) -> Dict[str, Any]:
        return self._get_metadata(user_id)

    def save_selection(self, user_id: str, selection: Dict[str, Any]) -> None:
        metadata = self._get_metadata(user_id)
        metadata["pinterest_selection"] = selection
        self._update_metadata(user_id, metadata)

    def set_token_source(self, user_id: str, source: str) -> None:
        metadata = self._get_metadata(user_id)
        metadata["token_source"] = source
        self._update_metadata(user_id, metadata)

    def ensure_env_token(self, user_id: str) -> None:
        if self.get_token_row(user_id):
            return
        access_token = self._settings.pinterest_access_token
        if not access_token:
            return
        token_data = {
            "access_token": access_token,
            "expires_in": self._settings.pinterest_access_token_expires_in,
            "scope": self._settings.pinterest_access_token_scope or None,
            "token_type": self._settings.pinterest_access_token_token_type or "bearer",
        }
        self.upsert_tokens(user_id, token_data)
        self.set_token_source(user_id, "env")

    def get_token_health(self, user_id: str, grace_seconds: int) -> Dict[str, Any]:
        row = self.get_token_row(user_id)
        now = datetime.now(timezone.utc)

        if not row:
            return {
                "status": "missing",
                "expires_at": None,
                "seconds_until_expiry": None,
                "has_refresh_token": False,
                "refresh_available": False,
                "should_reconnect": True,
                "checked_at": now.isoformat(),
                "token_source": None,
            }

        expires_at = _parse_datetime(row.get("expires_at"))
        refresh_expires_at = _parse_datetime(row.get("refresh_expires_at"))
        has_refresh = bool(row.get("refresh_token"))
        refresh_available = has_refresh and (refresh_expires_at is None or refresh_expires_at > now)

        seconds_until = None
        status = "unknown"
        if expires_at:
            seconds_until = int((expires_at - now).total_seconds())
            if seconds_until <= 0:
                status = "expired"
            elif seconds_until <= grace_seconds:
                status = "expiring"
            else:
                status = "ok"

        should_reconnect = status in {"missing", "expired"} or (status == "expiring" and not refresh_available)

        metadata = self._get_metadata(user_id)
        return {
            "status": status,
            "expires_at": expires_at.isoformat() if expires_at else None,
            "seconds_until_expiry": seconds_until,
            "has_refresh_token": has_refresh,
            "refresh_available": refresh_available,
            "should_reconnect": should_reconnect,
            "checked_at": now.isoformat(),
            "token_source": metadata.get("token_source"),
        }

    def sync_selection(self, user_id: str) -> Dict[str, Any]:
        token_row = self.get_token_row(user_id)
        if not token_row:
            raise PinterestApiError("Pinterest account not connected")

        access_token = self._ensure_access_token(user_id, token_row)
        selection = self.get_selection(user_id)
        auto_include = selection.get("auto_include_new_boards", True)

        try:
            boards = self._fetch_all_boards(access_token)
        except PinterestApiError as exc:
            if exc.status_code == 401 and token_row.get("refresh_token"):
                token_data = self.refresh_access_token(token_row["refresh_token"])
                self.upsert_tokens(user_id, token_data)
                token_row = self.get_token_row(user_id) or token_row
                access_token = token_row.get("access_token")
                boards = self._fetch_all_boards(access_token)
            else:
                raise
        board_ids = [b.get("id") for b in boards if b.get("id")]
        board_id_set = set(board_ids)

        selected_boards = selection.get("boards") or []
        selected_sections = selection.get("sections") or []

        # Prune removed boards
        selected_boards = [bid for bid in selected_boards if bid in board_id_set]

        # Prune removed sections
        sections_by_board: Dict[str, List[str]] = {}
        for section in selected_sections:
            board_id = section.get("board_id")
            section_id = section.get("section_id")
            if not board_id or not section_id:
                continue
            sections_by_board.setdefault(board_id, []).append(section_id)

        valid_sections: List[Dict[str, Any]] = []
        for board_id, section_ids in sections_by_board.items():
            if board_id not in board_id_set:
                continue
            try:
                existing_sections = self._fetch_board_sections(access_token, board_id)
            except PinterestApiError as exc:
                if exc.status_code == 401 and token_row.get("refresh_token"):
                    token_data = self.refresh_access_token(token_row["refresh_token"])
                    self.upsert_tokens(user_id, token_data)
                    token_row = self.get_token_row(user_id) or token_row
                    access_token = token_row.get("access_token")
                    existing_sections = self._fetch_board_sections(access_token, board_id)
                else:
                    raise
            existing_ids = {s.get("id") for s in existing_sections if s.get("id")}
            for section_id in section_ids:
                if section_id in existing_ids:
                    valid_sections.append({"board_id": board_id, "section_id": section_id})

        # Auto-include new boards if enabled
        if auto_include:
            existing = set(selected_boards)
            for board_id in board_ids:
                if board_id not in existing:
                    selected_boards.append(board_id)

        synced_selection = {
            "boards": selected_boards,
            "sections": valid_sections,
            "auto_include_new_boards": auto_include,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        }

        metadata = self._get_metadata(user_id)
        metadata["pinterest_selection"] = synced_selection
        metadata["boards_snapshot"] = [
            {
                "id": board.get("id"),
                "name": board.get("name"),
                "pin_count": board.get("pin_count"),
                "privacy": board.get("privacy"),
            }
            for board in boards
            if board.get("id")
        ]
        metadata["boards_snapshot_at"] = datetime.now(timezone.utc).isoformat()
        self._update_metadata(user_id, metadata)

        return synced_selection

    # ---------------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------------

    def _get_metadata(self, user_id: str) -> Dict[str, Any]:
        row = self.get_token_row(user_id)
        if not row:
            return {}
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}
        if isinstance(metadata, dict):
            return metadata
        return {}

    def _update_metadata(self, user_id: str, metadata: Dict[str, Any]) -> None:
        payload = {
            "metadata": metadata,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self._supabase.table("user_oauth_tokens").update(payload).eq("user_id", user_id).eq("provider", "pinterest").execute()

    def _get(
        self,
        access_token: str,
        path: str,
        params: Optional[Dict[str, Any]],
        error_prefix: str,
    ) -> Dict[str, Any]:
        url = f"{self._settings.pinterest_api_base_url}{path}"
        resp = requests.get(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
            timeout=self._settings.pinterest_request_timeout_seconds,
        )
        if resp.status_code >= 400:
            raise PinterestApiError(
                f"{error_prefix} ({resp.status_code}): {resp.text}",
                status_code=resp.status_code,
            )
        return resp.json()

    def _fetch_all_pins(
        self,
        user_id: str,
        access_token: str,
        token_row: Dict[str, Any],
        max_pins: int,
    ) -> List[Dict[str, Any]]:
        pins: List[Dict[str, Any]] = []
        bookmark: Optional[str] = None
        page_size = min(100, max(1, max_pins))

        while len(pins) < max_pins:
            try:
                response = self.list_pins(access_token, page_size=page_size, bookmark=bookmark)
            except PinterestApiError as exc:
                if exc.status_code == 401 and token_row.get("refresh_token"):
                    token_data = self.refresh_access_token(token_row["refresh_token"])
                    self.upsert_tokens(user_id, token_data)
                    token_row = self.get_token_row(user_id) or token_row
                    access_token = token_row.get("access_token")
                    response = self.list_pins(access_token, page_size=page_size, bookmark=bookmark)
                else:
                    raise

            items = response.get("items") or []
            pins.extend(items)
            bookmark = response.get("bookmark")
            if not bookmark or not items:
                break

        return pins[:max_pins]

    def _fetch_from_boards(
        self,
        access_token: str,
        board_ids: Iterable[str],
        max_pins: int,
    ) -> List[Dict[str, Any]]:
        pins: List[Dict[str, Any]] = []
        for board_id in board_ids:
            pins.extend(self._fetch_paginated(
                lambda bookmark: self.list_board_pins(
                    access_token=access_token,
                    board_id=board_id,
                    page_size=min(100, max_pins),
                    bookmark=bookmark,
                ),
                max_pins=max_pins - len(pins),
            ))
            if len(pins) >= max_pins:
                break
        return pins[:max_pins]

    def _fetch_from_sections(
        self,
        access_token: str,
        sections: Iterable[Dict[str, Any]],
        max_pins: int,
    ) -> List[Dict[str, Any]]:
        pins: List[Dict[str, Any]] = []
        for section in sections:
            board_id = section.get("board_id")
            section_id = section.get("section_id")
            if not board_id or not section_id:
                continue
            pins.extend(self._fetch_paginated(
                lambda bookmark: self.list_section_pins(
                    access_token=access_token,
                    board_id=board_id,
                    section_id=section_id,
                    page_size=min(100, max_pins),
                    bookmark=bookmark,
                ),
                max_pins=max_pins - len(pins),
            ))
            if len(pins) >= max_pins:
                break
        return pins[:max_pins]

    def _fetch_paginated(
        self,
        fetch_page,
        max_pins: int,
    ) -> List[Dict[str, Any]]:
        pins: List[Dict[str, Any]] = []
        bookmark: Optional[str] = None
        while len(pins) < max_pins:
            response = fetch_page(bookmark)
            items = response.get("items") or []
            pins.extend(items)
            bookmark = response.get("bookmark")
            if not bookmark or not items:
                break
        return pins[:max_pins]

    def _fetch_all_boards(self, access_token: str) -> List[Dict[str, Any]]:
        boards: List[Dict[str, Any]] = []
        bookmark: Optional[str] = None
        while True:
            response = self.list_boards(access_token, page_size=100, bookmark=bookmark)
            items = response.get("items") or []
            boards.extend(items)
            bookmark = response.get("bookmark")
            if not bookmark or not items:
                break
        return boards

    def _fetch_board_sections(self, access_token: str, board_id: str) -> List[Dict[str, Any]]:
        sections: List[Dict[str, Any]] = []
        bookmark: Optional[str] = None
        while True:
            response = self.list_board_sections(access_token, board_id=board_id, page_size=100, bookmark=bookmark)
            items = response.get("items") or []
            sections.extend(items)
            bookmark = response.get("bookmark")
            if not bookmark or not items:
                break
        return sections

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _ensure_access_token(self, user_id: str, token_row: Dict[str, Any]) -> str:
        expires_at = _parse_datetime(token_row.get("expires_at"))
        if expires_at and expires_at <= datetime.now(timezone.utc) + timedelta(seconds=60):
            refresh_token = token_row.get("refresh_token")
            if not refresh_token:
                raise PinterestApiError("Access token expired and no refresh token available")
            token_data = self.refresh_access_token(refresh_token)
            self.upsert_tokens(user_id, token_data)
            token_row = self.get_token_row(user_id) or token_row

        access_token = token_row.get("access_token")
        if not access_token:
            raise PinterestApiError("Missing Pinterest access token")
        return access_token

    def _normalize_token_record(self, token_data: Dict[str, Any]) -> PinterestTokenRecord:
        if not token_data.get("access_token"):
            raise PinterestApiError("Pinterest token response missing access_token")
        expires_at = _compute_expiry(token_data.get("expires_in"))
        refresh_expires_at = _compute_expiry(token_data.get("refresh_token_expires_in"))
        scope = token_data.get("scope")
        if isinstance(scope, list):
            scope = " ".join(str(s) for s in scope)

        return PinterestTokenRecord(
            access_token=token_data.get("access_token"),
            refresh_token=token_data.get("refresh_token"),
            token_type=token_data.get("token_type"),
            scope=scope,
            expires_at=expires_at,
            refresh_expires_at=refresh_expires_at,
        )


def _compute_expiry(expires_in: Optional[Any]) -> Optional[datetime]:
    if expires_in is None:
        return None
    try:
        seconds = int(expires_in)
    except (TypeError, ValueError):
        return None
    return datetime.now(timezone.utc) + timedelta(seconds=seconds)


def _parse_datetime(value: Optional[Any]) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            if value.endswith("Z"):
                value = value[:-1] + "+00:00"
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _build_pagination_params(page_size: int, bookmark: Optional[str]) -> Dict[str, Any]:
    params: Dict[str, Any] = {"page_size": page_size}
    if bookmark:
        params["bookmark"] = bookmark
    return params
