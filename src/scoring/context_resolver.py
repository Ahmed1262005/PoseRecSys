"""
Context Resolver — builds UserContext from available data sources.

Resolution order per signal:

**Age**
    1. ``OnboardingProfile.birthdate`` -> compute age -> map to AgeGroup

**Address**
    1. JWT ``user_metadata`` (free, already decoded on every request)
    2. ``supabase.auth.admin.get_user()`` (fallback, cached 24 h)

**Weather**
    1. In-memory cache (TTL 30 min per city)
    2. OpenWeatherMap API call
    3. Fallback: derive season from date + hemisphere (no API needed)
"""

import logging
import time
import threading
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

from scoring.context import AgeGroup, Season, UserContext, WeatherContext

logger = logging.getLogger(__name__)

# ── Southern-hemisphere countries (ISO-2 and full names) ──────────
_SOUTHERN_COUNTRIES = frozenset({
    "au", "australia", "nz", "new zealand", "za", "south africa",
    "ar", "argentina", "br", "brazil", "cl", "chile", "py", "paraguay",
    "uy", "uruguay", "pe", "peru", "bw", "botswana", "mz", "mozambique",
    "mg", "madagascar", "na", "namibia", "zw", "zimbabwe",
    "id", "indonesia", "fj", "fiji",
})


class ContextResolver:
    """
    Builds :class:`UserContext` from available data sources.

    Thread-safe.  Designed to be instantiated once per application
    lifetime and reused across requests.
    """

    def __init__(self, weather_api_key: str = "") -> None:
        self._weather_api_key = weather_api_key
        # {cache_key: (timestamp, WeatherContext)}
        self._weather_cache: Dict[str, Tuple[float, WeatherContext]] = {}
        self._WEATHER_TTL = 86400  # 24 hours
        # {user_id: (timestamp, address_dict)}
        self._address_cache: Dict[str, Tuple[float, Optional[dict]]] = {}
        self._ADDRESS_TTL = 86400  # 24 hours
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────

    def resolve(
        self,
        user_id: str,
        jwt_user_metadata: Optional[dict] = None,
        birthdate: Optional[str] = None,
        onboarding_profile: Optional[dict] = None,
    ) -> UserContext:
        """
        Build complete :class:`UserContext`.

        Best-effort — missing data results in ``None`` fields and the
        corresponding scorer is simply skipped.
        """
        ctx = UserContext(user_id=user_id)

        # 1. Age
        if birthdate:
            try:
                ctx.age_years = _compute_age(birthdate)
                ctx.age_group = _age_to_group(ctx.age_years)
            except (ValueError, TypeError):
                logger.debug("Invalid birthdate %r for user %s", birthdate, user_id)

        # 2. Address
        address = self._resolve_address(user_id, jwt_user_metadata)
        if address:
            ctx.city = address.get("city")
            ctx.state = address.get("state")
            ctx.country = address.get("country")
            ctx.zip_code = address.get("zip") or address.get("postal_code")

        # 3. Weather
        if ctx.city and ctx.country:
            ctx.weather = self._resolve_weather(ctx.city, ctx.country)
        elif ctx.country:
            # No city but have country — still derive season
            season = _season_from_date_and_country(ctx.country)
            ctx.weather = WeatherContext(
                temperature_c=20.0, feels_like_c=20.0,
                condition="unknown", humidity=50, wind_speed_mps=0,
                season=season, is_mild=True,
            )

        # 4. Coverage preferences from profile
        if onboarding_profile:
            ctx.coverage_prefs = _extract_coverage_prefs(onboarding_profile)
            ctx.modesty_level = (
                onboarding_profile.get("modesty")
                or onboarding_profile.get("modesty_level")
            )

        return ctx

    # ── Address resolution ────────────────────────────────────────

    def _resolve_address(
        self, user_id: str, jwt_metadata: Optional[dict],
    ) -> Optional[dict]:
        """Try JWT user_metadata first, fall back to Supabase admin API."""
        # Attempt 1: JWT metadata
        if jwt_metadata:
            addr = _extract_address_from_metadata(jwt_metadata)
            if addr and addr.get("city"):
                return addr

        # Attempt 2: Cached address
        with self._lock:
            cached = self._address_cache.get(user_id)
        if cached and (time.time() - cached[0]) < self._ADDRESS_TTL:
            return cached[1]

        # Attempt 3: Supabase auth admin API
        try:
            from config.database import get_supabase_client
            client = get_supabase_client()
            user_response = client.auth.admin.get_user(user_id)
            metadata = (
                getattr(user_response, "user", None)
                and getattr(user_response.user, "user_metadata", None)
            ) or {}
            addr = _extract_address_from_metadata(metadata)
            with self._lock:
                self._address_cache[user_id] = (time.time(), addr)
            return addr
        except Exception as exc:
            logger.debug("Supabase auth admin lookup failed: %s", exc)
            with self._lock:
                self._address_cache[user_id] = (time.time(), None)
            return None

    # ── Weather resolution ────────────────────────────────────────

    def _resolve_weather(self, city: str, country: str) -> WeatherContext:
        """
        Fetch weather from OpenWeatherMap API (cached 30 min).

        Falls back to season-from-date if the API call fails or no API
        key is configured.
        """
        cache_key = f"{city}:{country}".lower()

        with self._lock:
            cached = self._weather_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._WEATHER_TTL:
            return cached[1]

        # Try OpenWeatherMap
        if self._weather_api_key:
            try:
                weather = _fetch_openweathermap(
                    city, country, self._weather_api_key,
                )
                with self._lock:
                    self._weather_cache[cache_key] = (time.time(), weather)
                return weather
            except Exception as exc:
                logger.warning("Weather API failed for %s,%s: %s", city, country, exc)

        # Fallback: derive season from date + hemisphere
        season = _season_from_date_and_country(country)
        weather = WeatherContext(
            temperature_c=20.0, feels_like_c=20.0,
            condition="unknown", humidity=50, wind_speed_mps=0,
            season=season, is_mild=True,
        )
        with self._lock:
            self._weather_cache[cache_key] = (time.time(), weather)
        return weather


# ── Pure helpers (no I/O, easily testable) ────────────────────────

def _extract_address_from_metadata(metadata: dict) -> Optional[dict]:
    """
    Extract address from user_metadata.

    Handles both nested and flat structures::

        {"address": {"city": "NYC", ...}}          # nested
        {"city": "NYC", "country": "US", ...}      # flat
    """
    if not metadata:
        return None

    if "address" in metadata and isinstance(metadata["address"], dict):
        return metadata["address"]

    # Flat fields
    if any(k in metadata for k in ("city", "country", "state", "zip", "postal_code")):
        return {
            "city": metadata.get("city"),
            "state": metadata.get("state"),
            "country": metadata.get("country"),
            "zip": metadata.get("zip") or metadata.get("postal_code"),
            "street": metadata.get("street"),
        }
    return None


def _compute_age(birthdate: str) -> int:
    """``YYYY-MM-DD`` string -> age in whole years."""
    birth = date.fromisoformat(birthdate)
    today = date.today()
    return today.year - birth.year - (
        (today.month, today.day) < (birth.month, birth.day)
    )


def _age_to_group(age: int) -> AgeGroup:
    """Map integer age to the closest :class:`AgeGroup`."""
    if age <= 24:
        return AgeGroup.GEN_Z
    elif age <= 34:
        return AgeGroup.YOUNG_ADULT
    elif age <= 44:
        return AgeGroup.MID_CAREER
    elif age <= 64:
        return AgeGroup.ESTABLISHED
    else:
        return AgeGroup.SENIOR


def _season_from_date_and_country(country: str) -> Season:
    """Derive current season from today's date and hemisphere."""
    month = date.today().month
    southern = country.lower().strip() in _SOUTHERN_COUNTRIES

    if southern:
        if month in (12, 1, 2):
            return Season.SUMMER
        elif month in (3, 4, 5):
            return Season.FALL
        elif month in (6, 7, 8):
            return Season.WINTER
        else:
            return Season.SPRING
    else:
        if month in (12, 1, 2):
            return Season.WINTER
        elif month in (3, 4, 5):
            return Season.SPRING
        elif month in (6, 7, 8):
            return Season.SUMMER
        else:
            return Season.FALL


def _extract_coverage_prefs(profile: dict) -> List[str]:
    """
    Extract explicit coverage preferences from an onboarding profile dict.

    Returns a list like ``["no_revealing", "no_crop"]``.
    """
    prefs: List[str] = []
    flag_names = [
        "no_sleeveless", "no_tanks", "no_crop",
        "no_athletic", "no_revealing",
    ]
    for flag in flag_names:
        if profile.get(flag):
            prefs.append(flag)

    # Also include styles_to_avoid as coverage signals
    styles_avoid = profile.get("styles_to_avoid") or []
    if isinstance(styles_avoid, list):
        prefs.extend(styles_avoid)

    return prefs


def _fetch_openweathermap(
    city: str, country: str, api_key: str,
) -> WeatherContext:
    """Call OpenWeatherMap Current Weather API."""
    import requests

    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={
            "q": f"{city},{country}",
            "appid": api_key,
            "units": "metric",
        },
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()

    temp = data["main"]["temp"]
    feels = data["main"]["feels_like"]
    condition = data["weather"][0]["main"].lower() if data.get("weather") else "unknown"

    season = _season_from_date_and_country(country)

    return WeatherContext(
        temperature_c=temp,
        feels_like_c=feels,
        condition=condition,
        humidity=data["main"].get("humidity", 50),
        wind_speed_mps=data.get("wind", {}).get("speed", 0),
        season=season,
        is_hot=feels > 25,
        is_cold=feels < 10,
        is_mild=10 <= feels <= 25,
        is_rainy=condition in ("rain", "drizzle", "thunderstorm"),
    )
