# Shared Scoring Module — Complete Implementation Plan

## Goal

Build `src/scoring/` — a shared module that both the **feed pipeline** (`recs/`) and **search reranker** (`search/`) import from. Contains context-aware scoring: **age affinity**, **season/weather**, and the **session scoring integration** from the previous plan.

---

## Architecture

```
src/scoring/
├── __init__.py                    # Public API exports
├── context.py                     # UserContext dataclass (age, location, weather, season)
├── context_resolver.py            # Builds UserContext from JWT + Supabase + Weather API
├── age_scorer.py                  # Age-affinity scoring engine
├── weather_scorer.py              # Season/weather/material scoring
├── scorer.py                      # ContextScorer — orchestrates all shared scorers
└── constants/
    ├── __init__.py
    ├── age_item_frequency.py      # Item frequency tables by age group
    ├── age_style_affinity.py      # Style popularity ranks by age group
    ├── age_occasion_affinity.py   # Occasion popularity ranks by age group
    ├── age_coverage_tolerance.py  # Coverage tolerance by age group
    ├── age_fit_preferences.py     # Fit preferences by age × category
    ├── age_color_pattern.py       # Color/pattern loudness by age group
    └── weather_materials.py       # Material-season mapping, weather rules
```

### Integration Points

```
                    ┌─────────────────────┐
                    │   src/scoring/       │
                    │   ContextScorer      │
                    │   (age + weather)    │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼                             ▼
    ┌─────────────────┐          ┌──────────────────┐
    │ recs/pipeline.py │          │ search/reranker.py│
    │ feed_reranker.py │          │ hybrid_search.py  │
    │ session_scoring  │          │ search routes     │
    └─────────────────┘          └──────────────────┘
```

Both pipelines:
1. Build `UserContext` once per request (from JWT + cached weather)
2. Call `ContextScorer.score_item(item, user_context)` → returns adjustment float
3. Add adjustment to existing scores (feed score or rrf_score)

---

## File-by-File Implementation

### 1. `src/scoring/context.py` — UserContext Dataclass

```python
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum

class AgeGroup(Enum):
    """Age brackets matching the affinity tables."""
    GEN_Z = "18-24"
    YOUNG_ADULT = "25-34"
    MID_CAREER = "35-44"
    ESTABLISHED = "45-64"
    SENIOR = "65+"

class Season(Enum):
    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    WINTER = "winter"

@dataclass
class WeatherContext:
    """Current weather at user's location."""
    temperature_c: float              # Current temp in Celsius
    feels_like_c: float               # Feels-like temp
    condition: str                     # "clear", "rain", "snow", "clouds", etc.
    humidity: int                      # 0-100
    wind_speed_mps: float             # m/s
    season: Season                     # Derived from hemisphere + date
    is_hot: bool = False              # feels_like > 25°C
    is_cold: bool = False             # feels_like < 10°C
    is_mild: bool = False             # 10-25°C
    is_rainy: bool = False            # rain/drizzle/thunderstorm

@dataclass
class UserContext:
    """Complete user context for scoring. Built once per request."""
    user_id: str
    age_group: Optional[AgeGroup] = None
    age_years: Optional[int] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    weather: Optional[WeatherContext] = None
    # Convenience flags derived from profile
    coverage_prefs: List[str] = field(default_factory=list)  # ["no_revealing", "no_crop", etc.]
    modesty_level: Optional[str] = None  # "modest", "balanced", "revealing"
```

### 2. `src/scoring/context_resolver.py` — Build UserContext

```python
class ContextResolver:
    """
    Builds UserContext from available data sources.
    
    Resolution order for address:
    1. JWT user_metadata (free, already available)
    2. Supabase auth.admin.get_user() (fallback, cached)
    
    Resolution order for weather:
    1. In-memory cache (TTL 30 min per city)
    2. OpenWeatherMap API call
    
    Resolution for age:
    1. OnboardingProfile.birthdate → compute age → map to AgeGroup
    """
    
    def __init__(self, weather_api_key: str):
        self._weather_cache: Dict[str, Tuple[float, WeatherContext]] = {}
        self._WEATHER_TTL = 1800  # 30 minutes
        self._address_cache: Dict[str, Tuple[float, dict]] = {}
        self._ADDRESS_TTL = 86400  # 24 hours (address rarely changes)
        self._weather_api_key = weather_api_key
    
    def resolve(
        self,
        user_id: str,
        jwt_user_metadata: Optional[dict] = None,
        birthdate: Optional[str] = None,
        onboarding_profile: Optional[dict] = None,
    ) -> UserContext:
        """Build complete UserContext. Best-effort — missing data = None fields."""
        
        ctx = UserContext(user_id=user_id)
        
        # 1. Resolve age
        if birthdate:
            ctx.age_years = _compute_age(birthdate)
            ctx.age_group = _age_to_group(ctx.age_years)
        
        # 2. Resolve address (JWT first, then Supabase fallback)
        address = self._resolve_address(user_id, jwt_user_metadata)
        if address:
            ctx.city = address.get("city")
            ctx.state = address.get("state")
            ctx.country = address.get("country")
            ctx.zip_code = address.get("zip") or address.get("postal_code")
        
        # 3. Resolve weather (if we have a city)
        if ctx.city and ctx.country:
            ctx.weather = self._resolve_weather(ctx.city, ctx.country)
        
        # 4. Resolve coverage preferences from profile
        if onboarding_profile:
            ctx.coverage_prefs = _extract_coverage_prefs(onboarding_profile)
            ctx.modesty_level = onboarding_profile.get("modesty")
        
        return ctx
    
    def _resolve_address(self, user_id, jwt_metadata):
        """Try JWT user_metadata first, fall back to Supabase admin API."""
        # Attempt 1: JWT metadata
        if jwt_metadata:
            addr = _extract_address_from_metadata(jwt_metadata)
            if addr and addr.get("city"):
                return addr
        
        # Attempt 2: Cached address
        cached = self._address_cache.get(user_id)
        if cached and (time.time() - cached[0]) < self._ADDRESS_TTL:
            return cached[1]
        
        # Attempt 3: Supabase auth admin API
        try:
            from config.database import get_supabase_client
            client = get_supabase_client()
            user_response = client.auth.admin.get_user(user_id)
            metadata = user_response.user.user_metadata or {}
            addr = _extract_address_from_metadata(metadata)
            self._address_cache[user_id] = (time.time(), addr)
            return addr
        except Exception:
            return None
    
    def _resolve_weather(self, city, country):
        """Fetch weather from OpenWeatherMap API (cached 30 min)."""
        cache_key = f"{city}:{country}".lower()
        cached = self._weather_cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._WEATHER_TTL:
            return cached[1]
        
        try:
            weather = _fetch_openweathermap(city, country, self._weather_api_key)
            self._weather_cache[cache_key] = (time.time(), weather)
            return weather
        except Exception:
            # Fallback: derive season from date + hemisphere
            season = _season_from_date_and_country(country)
            weather = WeatherContext(
                temperature_c=20.0,  # mild default
                feels_like_c=20.0,
                condition="unknown",
                humidity=50,
                wind_speed_mps=0,
                season=season,
                is_mild=True,
            )
            self._weather_cache[cache_key] = (time.time(), weather)
            return weather


def _extract_address_from_metadata(metadata: dict) -> Optional[dict]:
    """
    Extract address from user_metadata. 
    Handles both flat and nested structures:
    - Flat: {"city": "NYC", "state": "NY", "country": "US", "zip": "10001"}
    - Nested: {"address": {"city": "NYC", ...}}
    """
    if "address" in metadata and isinstance(metadata["address"], dict):
        return metadata["address"]
    # Check for flat fields
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
    """YYYY-MM-DD → age in years."""
    from datetime import date
    birth = date.fromisoformat(birthdate)
    today = date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def _age_to_group(age: int) -> AgeGroup:
    if age < 18:
        return AgeGroup.GEN_Z      # Treat under-18 as Gen Z
    elif age <= 24:
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
    """Derive current season from date + hemisphere."""
    from datetime import date
    month = date.today().month
    southern = country.lower() in (
        "au", "australia", "nz", "new zealand", "za", "south africa",
        "ar", "argentina", "br", "brazil", "cl", "chile",
    )
    if southern:
        # Flip seasons
        if month in (12, 1, 2): return Season.SUMMER
        elif month in (3, 4, 5): return Season.FALL
        elif month in (6, 7, 8): return Season.WINTER
        else: return Season.SPRING
    else:
        if month in (12, 1, 2): return Season.WINTER
        elif month in (3, 4, 5): return Season.SPRING
        elif month in (6, 7, 8): return Season.SUMMER
        else: return Season.FALL


def _fetch_openweathermap(city, country, api_key) -> WeatherContext:
    """Call OpenWeatherMap Current Weather API."""
    import requests
    resp = requests.get(
        "https://api.openweathermap.org/data/2.5/weather",
        params={"q": f"{city},{country}", "appid": api_key, "units": "metric"},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    
    temp = data["main"]["temp"]
    feels = data["main"]["feels_like"]
    condition = data["weather"][0]["main"].lower()
    
    season = _season_from_date_and_country(country)
    
    return WeatherContext(
        temperature_c=temp,
        feels_like_c=feels,
        condition=condition,
        humidity=data["main"]["humidity"],
        wind_speed_mps=data["wind"]["speed"],
        season=season,
        is_hot=feels > 25,
        is_cold=feels < 10,
        is_mild=10 <= feels <= 25,
        is_rainy=condition in ("rain", "drizzle", "thunderstorm"),
    )
```

### 3. `src/scoring/constants/age_item_frequency.py`

Encodes the user's item frequency tables as numeric weights.

Mapping: `Very common` = 1.0, `Common` = 0.6, `Uncommon` = 0.2, transitional (e.g., `Uncommon → Common`) = 0.4

```python
"""
Item frequency weights by age group.

Source: Domain expert fashion data.
Scale: 0.0 (never) → 1.0 (very common)
Keys: canonical article types from feasibility_filter.py
"""

from scoring.context import AgeGroup

# {AgeGroup: {article_type: frequency_weight}}
ITEM_FREQUENCY: dict = {
    AgeGroup.GEN_Z: {  # 18-24
        # Tops
        "tank_top": 1.0, "cami": 0.6, "tshirt": 1.0, "blouse": 0.6,
        "tube_top": 0.8, "sweater": 1.0, "cardigan": 0.6, "bodysuit": 0.6,
        # Bottoms
        "pants": 0.6, "jeans": 1.0, "shorts": 0.6, "leggings": 0.6, "skirt": 0.6,
        # One-piece
        "dress": 0.6, "romper": 0.2, "jumpsuit": 0.4,
        # Outerwear
        "coat": 0.6, "jacket": 1.0, "vest": 0.6, "blazer": 0.6,
    },
    AgeGroup.YOUNG_ADULT: {  # 25-34
        "tank_top": 1.0, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.4, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.6,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.6, "leggings": 1.0, "skirt": 0.6,
        "dress": 1.0, "romper": 0.4, "jumpsuit": 0.6,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
    },
    AgeGroup.MID_CAREER: {  # 35-44
        "tank_top": 0.8, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.4,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.4, "leggings": 0.6, "skirt": 0.6,
        "dress": 1.0, "romper": 0.2, "jumpsuit": 0.6,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
    },
    AgeGroup.ESTABLISHED: {  # 45-64
        "tank_top": 0.6, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.2,
        "pants": 1.0, "jeans": 1.0, "shorts": 0.4, "leggings": 0.6, "skirt": 0.6,
        "dress": 1.0, "romper": 0.2, "jumpsuit": 0.4,
        "coat": 1.0, "jacket": 1.0, "vest": 0.6, "blazer": 1.0,
    },
    AgeGroup.SENIOR: {  # 65+
        "tank_top": 0.6, "cami": 0.6, "tshirt": 1.0, "blouse": 1.0,
        "tube_top": 0.2, "sweater": 1.0, "cardigan": 1.0, "bodysuit": 0.2,
        "pants": 1.0, "jeans": 0.6, "shorts": 0.4, "leggings": 0.6, "skirt": 0.6,
        "dress": 0.8, "romper": 0.2, "jumpsuit": 0.2,
        "coat": 1.0, "jacket": 1.0, "vest": 1.0, "blazer": 0.6,
    },
}
```

### 4. `src/scoring/constants/age_style_affinity.py`

Direct encoding of style rank tables (1-5 scale, normalized to 0.0-1.0).

```python
"""
Style popularity by age group.

Source rank 1-5 normalized: rank/5 → 0.2, 0.4, 0.6, 0.8, 1.0
Maps to system style_tags and style_persona values.
"""

from scoring.context import AgeGroup

# {AgeGroup: {style_tag: affinity_weight}}
STYLE_AFFINITY: dict = {
    AgeGroup.GEN_Z: {
        "trendy": 1.0, "streetwear": 1.0, "sporty": 0.8,
        "romantic": 0.8, "boho": 0.6, "minimal": 0.6,
        "classic": 0.4, "elegant": 0.4,
    },
    AgeGroup.YOUNG_ADULT: {
        "minimal": 1.0, "sporty": 1.0, "trendy": 0.8,
        "classic": 0.8, "streetwear": 0.8, "elegant": 0.8,
        "boho": 0.6, "romantic": 0.6,
    },
    AgeGroup.MID_CAREER: {
        "classic": 1.0, "elegant": 1.0, "minimal": 0.8,
        "sporty": 0.8, "trendy": 0.6, "boho": 0.6,
        "streetwear": 0.6, "romantic": 0.6,
    },
    AgeGroup.ESTABLISHED: {
        "classic": 1.0, "elegant": 1.0, "minimal": 0.8,
        "sporty": 0.8, "boho": 0.6, "romantic": 0.6,
        "trendy": 0.4, "streetwear": 0.4,
    },
    AgeGroup.SENIOR: {
        "sporty": 0.8, "classic": 0.8, "elegant": 0.8,
        "minimal": 0.6, "boho": 0.4, "romantic": 0.4,
        "trendy": 0.2, "streetwear": 0.2,
    },
}

# Mapping from system style_tags/persona values → canonical style keys
STYLE_TAG_MAP: dict = {
    # Direct matches
    "trendy": "trendy", "classic": "classic", "elegant": "elegant",
    "minimal": "minimal", "minimalist": "minimal", "clean": "minimal",
    "streetwear": "streetwear", "sporty": "sporty", "athletic": "sporty",
    "romantic": "romantic", "feminine": "romantic", "boho": "boho",
    "bohemian": "boho",
    # Extended mappings
    "casual": "trendy",  # Weighted toward trendy in Gen Z, classic in older
    "statement": "trendy",
    "preppy": "classic",
    "coquette": "romantic",
    "cottagecore": "boho",
    "dark academia": "classic",
    "quiet luxury": "classic",
    "old money": "classic",
    "clean girl": "minimal",
    "coastal": "boho",
    "y2k": "trendy",
    # Brand cluster style_tags
    "tailored": "classic", "knitwear": "classic", "blazers": "classic",
    "denim": "streetwear", "basics": "minimal", "logo": "streetwear",
    "leggings": "sporty", "performance": "sporty",
    "micro-trends": "trendy", "party": "trendy", "crop": "trendy",
    "cutouts": "trendy", "bodycon": "trendy",
    "boho": "boho", "western": "boho", "prints": "boho",
    "resort": "boho", "linen": "boho",
    "investment-pieces": "classic", "tailoring": "classic",
}
```

### 5. `src/scoring/constants/age_occasion_affinity.py`

```python
"""
Occasion popularity by age group.

Source rank 1-5 normalized to 0.0-1.0.
"""

from scoring.context import AgeGroup

OCCASION_AFFINITY: dict = {
    AgeGroup.GEN_Z: {
        "casual": 1.0, "evenings": 1.0, "events": 0.8,
        "sporty": 0.8, "smart_casual": 0.6, "office": 0.5,
    },
    AgeGroup.YOUNG_ADULT: {
        "casual": 1.0, "smart_casual": 1.0, "sporty": 1.0,
        "office": 0.8, "events": 0.8, "evenings": 0.8,
    },
    AgeGroup.MID_CAREER: {
        "casual": 1.0, "office": 1.0, "smart_casual": 1.0,
        "sporty": 0.8, "events": 0.8, "evenings": 0.6,
    },
    AgeGroup.ESTABLISHED: {
        "casual": 1.0, "smart_casual": 1.0,
        "office": 0.8, "sporty": 0.8, "events": 0.8,
        "evenings": 0.6,
    },
    AgeGroup.SENIOR: {
        "casual": 1.0, "sporty": 0.8,
        "smart_casual": 0.8, "events": 0.6,
        "office": 0.4, "evenings": 0.4,
    },
}

# Map system occasion values → canonical keys
OCCASION_MAP: dict = {
    "casual": "casual", "everyday": "casual", "weekend": "casual",
    "lounging": "casual", "errands": "casual",
    "office": "office", "work": "office",
    "smart-casual": "smart_casual", "smart casual": "smart_casual",
    "brunch": "smart_casual",
    "evening": "evenings", "date night": "evenings", "date_night": "evenings",
    "party": "evenings", "nights-out": "evenings", "going-out": "evenings",
    "events": "events", "wedding": "events", "formal": "events",
    "galas": "events", "weddings-guest": "events",
    "workout": "sporty", "active": "sporty", "gym": "sporty",
    "athleisure": "sporty", "sportswear": "sporty",
    "beach": "casual", "vacation": "casual", "resort": "casual",
}
```

### 6. `src/scoring/constants/age_coverage_tolerance.py`

```python
"""
Coverage tolerance by age group.

Scale: 0.0 (never) to 1.0 (very common/accepted).
Maps to styles_to_avoid values and product construction attributes.
"""

from scoring.context import AgeGroup

# {AgeGroup: {coverage_dimension: tolerance_score}}
COVERAGE_TOLERANCE: dict = {
    AgeGroup.GEN_Z: {
        "deep_necklines": 0.9, "open_back": 0.9, "sheer": 0.7,
        "cutouts": 0.9, "high_slit": 0.7, "crop": 0.9,
        "mini": 0.9, "bodycon": 0.9, "strapless": 0.8,
    },
    AgeGroup.YOUNG_ADULT: {
        "deep_necklines": 0.7, "open_back": 0.5, "sheer": 0.5,
        "cutouts": 0.5, "high_slit": 0.8, "crop": 0.6,
        "mini": 0.6, "bodycon": 0.5, "strapless": 0.6,
    },
    AgeGroup.MID_CAREER: {
        "deep_necklines": 0.5, "open_back": 0.3, "sheer": 0.3,
        "cutouts": 0.3, "high_slit": 0.6, "crop": 0.3,
        "mini": 0.3, "bodycon": 0.3, "strapless": 0.4,
    },
    AgeGroup.ESTABLISHED: {
        "deep_necklines": 0.3, "open_back": 0.2, "sheer": 0.2,
        "cutouts": 0.2, "high_slit": 0.5, "crop": 0.1,
        "mini": 0.2, "bodycon": 0.2, "strapless": 0.2,
    },
    AgeGroup.SENIOR: {
        "deep_necklines": 0.2, "open_back": 0.1, "sheer": 0.1,
        "cutouts": 0.1, "high_slit": 0.3, "crop": 0.05,
        "mini": 0.1, "bodycon": 0.1, "strapless": 0.1,
    },
}

# Map product attributes → coverage dimensions
# Used to detect which coverage dimension an item triggers
ARTICLE_TYPE_COVERAGE: dict = {
    # Article types that ARE a coverage dimension
    "crop_top": "crop",
    "tube_top": "strapless",
    "bandeau": "strapless",
    "bodycon_dress": "bodycon",
    "mini_dress": "mini",
    "mini_skirt": "mini",
    "bralette": "crop",
}

# Neckline values → coverage dimension
NECKLINE_COVERAGE: dict = {
    "deep v": "deep_necklines", "plunging": "deep_necklines",
    "sweetheart": "deep_necklines", "halter": "strapless",
    "off shoulder": "strapless", "off_shoulder": "strapless",
    "strapless": "strapless",
}

# Style tags → coverage dimension
STYLE_TAG_COVERAGE: dict = {
    "cutouts": "cutouts", "sheer": "sheer",
    "backless": "open_back", "open back": "open_back",
}
```

### 7. `src/scoring/constants/age_fit_preferences.py`

```python
"""
Fit preferences by age group, per broad category.

Scale: 0.0 (avoided) to 1.0 (dominant fit for this age).
"""

from scoring.context import AgeGroup

# {AgeGroup: {broad_category: {fit_value: affinity}}}
FIT_PREFERENCES: dict = {
    AgeGroup.GEN_Z: {
        "tops": {
            "cropped": 0.9, "oversized": 0.9, "fitted": 0.7,
            "relaxed": 0.6, "regular": 0.5, "slim": 0.4,
        },
        "bottoms": {
            "wide-leg": 0.9, "baggy": 0.9, "relaxed": 0.8,
            "straight": 0.7, "skinny": 0.3, "regular": 0.5,
        },
        "dresses": {
            "bodycon": 0.8, "mini": 0.8, "slip": 0.7,
            "fitted": 0.7, "relaxed": 0.5, "regular": 0.5,
        },
        "outerwear": {
            "oversized": 0.9, "regular": 0.5, "fitted": 0.4,
        },
    },
    AgeGroup.YOUNG_ADULT: {
        "tops": {
            "fitted": 0.8, "regular": 0.8, "relaxed": 0.6,
            "oversized": 0.4, "cropped": 0.4, "slim": 0.6,
        },
        "bottoms": {
            "straight": 0.8, "wide-leg": 0.8, "high-rise": 0.8,
            "regular": 0.7, "slim": 0.5, "skinny": 0.4,
        },
        "dresses": {
            "midi": 0.9, "wrap": 0.8, "fitted": 0.7,
            "regular": 0.7, "relaxed": 0.5, "mini": 0.4,
        },
        "outerwear": {
            "regular": 0.8, "fitted": 0.7, "oversized": 0.4,
        },
    },
    AgeGroup.MID_CAREER: {
        "tops": {
            "regular": 0.9, "fitted": 0.7, "relaxed": 0.6,
            "slim": 0.5, "oversized": 0.3, "cropped": 0.2,
        },
        "bottoms": {
            "straight": 0.9, "wide-leg": 0.7, "regular": 0.8,
            "high-rise": 0.8, "slim": 0.5, "skinny": 0.3,
        },
        "dresses": {
            "wrap": 0.9, "midi": 0.9, "fitted": 0.7,
            "regular": 0.7, "relaxed": 0.5, "mini": 0.2,
        },
        "outerwear": {
            "regular": 0.9, "fitted": 0.7, "oversized": 0.3,
        },
    },
    AgeGroup.ESTABLISHED: {
        "tops": {
            "regular": 0.9, "relaxed": 0.8, "fitted": 0.5,
            "slim": 0.4, "oversized": 0.2, "cropped": 0.1,
        },
        "bottoms": {
            "straight": 0.9, "regular": 0.8, "wide-leg": 0.6,
            "high-rise": 0.7, "slim": 0.4, "skinny": 0.2,
        },
        "dresses": {
            "midi": 0.9, "maxi": 0.7, "wrap": 0.8,
            "regular": 0.8, "relaxed": 0.6, "fitted": 0.5,
        },
        "outerwear": {
            "regular": 0.9, "fitted": 0.6, "oversized": 0.2,
        },
    },
    AgeGroup.SENIOR: {
        "tops": {
            "relaxed": 0.9, "regular": 0.9, "fitted": 0.3,
            "slim": 0.2, "oversized": 0.2, "cropped": 0.05,
        },
        "bottoms": {
            "straight": 0.9, "relaxed": 0.9, "regular": 0.8,
            "wide-leg": 0.5, "slim": 0.3, "skinny": 0.1,
        },
        "dresses": {
            "maxi": 0.8, "midi": 0.9, "relaxed": 0.9,
            "regular": 0.8, "wrap": 0.6, "mini": 0.1,
        },
        "outerwear": {
            "regular": 0.9, "relaxed": 0.7, "fitted": 0.3,
        },
    },
}
```

### 8. `src/scoring/constants/age_color_pattern.py`

```python
"""
Color/pattern preferences by age group.

Pattern loudness: how bold/experimental the patterns are.
Color boldness: how bright/varied the color palette is.
"""

from scoring.context import AgeGroup

# Pattern loudness tolerance (higher = more accepting of bold patterns)
PATTERN_LOUDNESS: dict = {
    AgeGroup.GEN_Z: {
        "bold": 0.9,        # checker, neon, graphic, animal print, tie dye
        "playful": 0.9,     # stripes, polka dots, novelty
        "classic": 0.7,     # florals, subtle stripes
        "solid": 0.7,       # still major (black/white)
    },
    AgeGroup.YOUNG_ADULT: {
        "bold": 0.5,
        "playful": 0.6,
        "classic": 0.8,     # refined prints
        "solid": 0.9,       # neutrals dominate
    },
    AgeGroup.MID_CAREER: {
        "bold": 0.3,
        "playful": 0.4,
        "classic": 0.9,     # stripes, dots, tasteful florals
        "solid": 0.9,
    },
    AgeGroup.ESTABLISHED: {
        "bold": 0.2,
        "playful": 0.3,
        "classic": 0.9,     # larger florals, geometric
        "solid": 0.9,
    },
    AgeGroup.SENIOR: {
        "bold": 0.2,
        "playful": 0.3,
        "classic": 0.8,
        "solid": 0.9,
    },
}

# Map pattern values to loudness categories
PATTERN_TO_LOUDNESS: dict = {
    # Bold
    "animal_print": "bold", "leopard": "bold", "neon": "bold",
    "tie_dye": "bold", "tie dye": "bold", "camo": "bold",
    "abstract": "bold", "graphic": "bold",
    # Playful
    "polka_dots": "playful", "polka dot": "playful",
    "checkered": "playful", "checker": "playful",
    "stripes": "playful", "striped": "playful",
    "plaid": "playful",
    # Classic
    "floral": "classic", "geometric": "classic",
    "paisley": "classic", "houndstooth": "classic",
    "herringbone": "classic",
    # Solid
    "solid": "solid",
}

# Color boldness by age (higher = more open to bright/varied colors)
COLOR_BOLDNESS: dict = {
    AgeGroup.GEN_Z: 0.9,       # brights, pastels, neon, high variance
    AgeGroup.YOUNG_ADULT: 0.6,  # neutrals + intentional pops
    AgeGroup.MID_CAREER: 0.5,   # jewel tones, muted palettes
    AgeGroup.ESTABLISHED: 0.4,  # neutrals + confident accents
    AgeGroup.SENIOR: 0.5,       # uplifting accents welcomed
}

# Color family classifications
BOLD_COLORS = {"neon", "bright", "hot pink", "electric blue", "lime"}
NEUTRAL_COLORS = {"neutrals", "black", "white", "cream", "beige", "camel", "grey", "browns"}
JEWEL_COLORS = {"emerald", "ruby", "sapphire", "burgundy", "navy", "deep purple"}
PASTEL_COLORS = {"pastels", "blush", "lavender", "mint", "baby blue", "soft pink"}
```

### 9. `src/scoring/constants/weather_materials.py`

```python
"""
Material-season/weather mapping.

Maps fabric types to weather appropriateness.
"""

from scoring.context import Season

# Materials well-suited for each season
SEASON_MATERIALS: dict = {
    Season.SUMMER: {
        "good": {"linen", "cotton", "silk", "chiffon", "rayon", "chambray",
                 "seersucker", "mesh", "jersey", "bamboo"},
        "bad": {"wool", "cashmere", "fleece", "velvet", "corduroy",
                "sherpa", "down", "heavy knit"},
    },
    Season.WINTER: {
        "good": {"wool", "cashmere", "fleece", "velvet", "corduroy",
                 "sherpa", "down", "heavy knit", "leather", "suede",
                 "faux fur", "thermal"},
        "bad": {"linen", "chiffon", "seersucker", "mesh"},
    },
    Season.SPRING: {
        "good": {"cotton", "linen", "denim", "jersey", "silk",
                 "rayon", "chambray", "light knit"},
        "bad": {"heavy knit", "sherpa", "down", "faux fur", "thermal"},
    },
    Season.FALL: {
        "good": {"wool", "cashmere", "denim", "corduroy", "suede",
                 "leather", "flannel", "knit", "jersey"},
        "bad": {"linen", "seersucker", "mesh", "chiffon"},
    },
}

# Temperature-based item type appropriateness
TEMP_ITEM_AFFINITY: dict = {
    "hot": {  # > 25°C
        "boost": {"tank_top", "cami", "shorts", "sundress", "sandals",
                  "tube_top", "crop_top", "mini_dress", "mini_skirt"},
        "penalize": {"coat", "puffer", "sweater", "turtleneck",
                     "hoodie", "vest", "heavy knit"},
    },
    "cold": {  # < 10°C
        "boost": {"coat", "puffer", "sweater", "cardigan", "turtleneck",
                  "hoodie", "vest", "jacket", "blazer", "pants", "jeans"},
        "penalize": {"tank_top", "tube_top", "crop_top", "shorts",
                     "mini_dress", "mini_skirt", "sundress", "cami"},
    },
    "mild": {  # 10-25°C
        "boost": {"jacket", "blazer", "cardigan", "jeans", "pants",
                  "dress", "blouse", "tshirt"},
        "penalize": {"puffer", "heavy coat"},
    },
    "rainy": {
        "boost": {"jacket", "coat", "pants", "jeans", "boots"},
        "penalize": {"suede", "open_toe", "sandals", "linen"},
    },
}

# Season-to-product seasons mapping
SEASON_PRODUCT_MAP: dict = {
    Season.SPRING: "Spring",
    Season.SUMMER: "Summer",
    Season.FALL: "Fall",
    Season.WINTER: "Winter",
}
```

### 10. `src/scoring/age_scorer.py` — Age Affinity Engine

```python
"""
Age-Affinity Scoring Engine.

Scores items based on how well they match the user's age group preferences.
Uses expert-curated affinity tables for:
- Item type frequency
- Style affinity
- Occasion affinity  
- Coverage tolerance
- Fit preferences
- Color/pattern loudness

All scores are additive adjustments (positive = boost, negative = penalize).
Total age adjustment is capped at MAX_AGE_ADJUSTMENT.
"""

MAX_AGE_ADJUSTMENT = 0.15  # Cap total age-based adjustment
COVERAGE_PENALTY_WEIGHT = 0.08
ITEM_FREQ_WEIGHT = 0.05
STYLE_AFFINITY_WEIGHT = 0.04
OCCASION_AFFINITY_WEIGHT = 0.03
FIT_AFFINITY_WEIGHT = 0.03
PATTERN_WEIGHT = 0.02


class AgeScorer:
    """Score items based on age-group affinity."""
    
    def score(self, item: dict, age_group: AgeGroup,
              coverage_prefs: list = None) -> float:
        """
        Compute age-affinity adjustment for a single item.
        
        Args:
            item: Product dict (must have canonical article_type, style_tags, etc.)
            age_group: User's age bracket
            coverage_prefs: User's explicit coverage preferences (override age defaults)
        
        Returns:
            Float adjustment to add to item's score. Range: [-MAX, +MAX].
        """
        adjustment = 0.0
        
        # 1. Item type frequency
        adjustment += self._score_item_frequency(item, age_group)
        
        # 2. Style affinity
        adjustment += self._score_style_affinity(item, age_group)
        
        # 3. Occasion affinity
        adjustment += self._score_occasion_affinity(item, age_group)
        
        # 4. Coverage tolerance (only penalize, never boost)
        #    User's explicit prefs override age defaults
        adjustment += self._score_coverage(item, age_group, coverage_prefs)
        
        # 5. Fit preference
        adjustment += self._score_fit(item, age_group)
        
        # 6. Pattern loudness
        adjustment += self._score_pattern(item, age_group)
        
        # Cap
        return max(-MAX_AGE_ADJUSTMENT, min(MAX_AGE_ADJUSTMENT, adjustment))
    
    def _score_item_frequency(self, item, age_group):
        """Boost items common for this age, penalize uncommon ones."""
        article_type = _get_canonical_type(item)
        if not article_type:
            return 0.0
        freq_table = ITEM_FREQUENCY.get(age_group, {})
        freq = freq_table.get(article_type, 0.5)  # 0.5 = neutral default
        # Convert 0-1 frequency to -1 to +1 range, then apply weight
        return (freq - 0.5) * 2 * ITEM_FREQ_WEIGHT
    
    def _score_style_affinity(self, item, age_group):
        """Boost styles popular for this age group."""
        style_tags = item.get("style_tags") or []
        if isinstance(style_tags, str):
            style_tags = [style_tags]
        
        affinity_table = STYLE_AFFINITY.get(age_group, {})
        if not style_tags or not affinity_table:
            return 0.0
        
        # Average affinity across item's style tags
        scores = []
        for tag in style_tags:
            canonical = STYLE_TAG_MAP.get(tag.lower())
            if canonical and canonical in affinity_table:
                scores.append(affinity_table[canonical])
        
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return (avg - 0.5) * 2 * STYLE_AFFINITY_WEIGHT
    
    def _score_occasion_affinity(self, item, age_group):
        """Boost occasions popular for this age group."""
        occasions = item.get("occasions") or []
        if isinstance(occasions, str):
            occasions = [occasions]
        
        affinity_table = OCCASION_AFFINITY.get(age_group, {})
        if not occasions or not affinity_table:
            return 0.0
        
        scores = []
        for occ in occasions:
            canonical = OCCASION_MAP.get(occ.lower())
            if canonical and canonical in affinity_table:
                scores.append(affinity_table[canonical])
        
        if not scores:
            return 0.0
        avg = sum(scores) / len(scores)
        return (avg - 0.5) * 2 * OCCASION_AFFINITY_WEIGHT
    
    def _score_coverage(self, item, age_group, user_prefs):
        """
        Penalize revealing items for older age groups.
        
        IMPORTANT: User's explicit preferences OVERRIDE age defaults.
        If user says "no_revealing=False" (i.e., they're OK with it),
        don't penalize based on age alone.
        """
        tolerance_table = COVERAGE_TOLERANCE.get(age_group, {})
        if not tolerance_table:
            return 0.0
        
        # Detect which coverage dimensions this item triggers
        triggered = _detect_coverage_dimensions(item)
        if not triggered:
            return 0.0
        
        # If user has explicit coverage prefs, use those instead
        if user_prefs:
            # User explicitly said OK → no penalty
            # User explicitly said avoid → full penalty regardless of age
            # (handled by feasibility filter, not here)
            pass
        
        # Apply age-based penalty for each triggered dimension
        penalty = 0.0
        for dim in triggered:
            tolerance = tolerance_table.get(dim, 0.5)
            # Low tolerance → bigger penalty
            # tolerance 0.9 → penalty ≈ 0, tolerance 0.1 → penalty ≈ -0.08
            dim_penalty = (1.0 - tolerance) * COVERAGE_PENALTY_WEIGHT * -1
            penalty += dim_penalty
        
        return penalty  # Always <= 0
    
    def _score_fit(self, item, age_group):
        """Boost fits common for this age × category."""
        broad_cat = _get_broad_category(item)
        fit = (item.get("fit_type") or item.get("fit") or "").lower()
        if not broad_cat or not fit:
            return 0.0
        
        fit_table = FIT_PREFERENCES.get(age_group, {}).get(broad_cat, {})
        if not fit_table:
            return 0.0
        
        affinity = fit_table.get(fit, 0.5)
        return (affinity - 0.5) * 2 * FIT_AFFINITY_WEIGHT
    
    def _score_pattern(self, item, age_group):
        """Score pattern based on age-appropriate loudness."""
        pattern = (item.get("pattern") or "").lower()
        if not pattern:
            return 0.0
        
        loudness_cat = PATTERN_TO_LOUDNESS.get(pattern)
        if not loudness_cat:
            return 0.0
        
        loudness_table = PATTERN_LOUDNESS.get(age_group, {})
        tolerance = loudness_table.get(loudness_cat, 0.5)
        return (tolerance - 0.5) * 2 * PATTERN_WEIGHT
```

### 11. `src/scoring/weather_scorer.py` — Weather/Season Engine

```python
"""
Weather & Season Scoring Engine.

Scores items based on:
1. Product season tags vs current season
2. Material-weather appropriateness
3. Temperature-based item type affinity
"""

MAX_WEATHER_ADJUSTMENT = 0.12

SEASON_MATCH_WEIGHT = 0.06
MATERIAL_WEIGHT = 0.04
TEMP_ITEM_WEIGHT = 0.05


class WeatherScorer:
    """Score items based on weather/season context."""
    
    def score(self, item: dict, weather: WeatherContext) -> float:
        """
        Compute weather-based adjustment for a single item.
        
        Returns float adjustment. Range: [-MAX, +MAX].
        """
        adjustment = 0.0
        
        # 1. Season tag match
        adjustment += self._score_season(item, weather.season)
        
        # 2. Material appropriateness
        adjustment += self._score_materials(item, weather)
        
        # 3. Temperature-based item type
        adjustment += self._score_temperature(item, weather)
        
        return max(-MAX_WEATHER_ADJUSTMENT, min(MAX_WEATHER_ADJUSTMENT, adjustment))
    
    def _score_season(self, item, season):
        """Boost items tagged for current season."""
        item_seasons = item.get("seasons") or []
        if isinstance(item_seasons, str):
            item_seasons = [item_seasons]
        if not item_seasons:
            return 0.0  # No season data = neutral
        
        current = SEASON_PRODUCT_MAP.get(season, "")
        if current in item_seasons:
            return SEASON_MATCH_WEIGHT  # Boost: in-season
        elif len(item_seasons) == 4:
            return 0.0  # All-season item = neutral
        else:
            return -SEASON_MATCH_WEIGHT * 0.5  # Mild penalty: out-of-season
    
    def _score_materials(self, item, weather):
        """Boost weather-appropriate materials, penalize inappropriate ones."""
        materials = item.get("materials") or item.get("apparent_fabric") or []
        if isinstance(materials, str):
            materials = [materials.lower()]
        else:
            materials = [m.lower() for m in materials if m]
        
        if not materials:
            return 0.0
        
        season_mats = SEASON_MATERIALS.get(weather.season, {})
        good_mats = season_mats.get("good", set())
        bad_mats = season_mats.get("bad", set())
        
        score = 0.0
        for mat in materials:
            if mat in good_mats:
                score += MATERIAL_WEIGHT
            elif mat in bad_mats:
                score -= MATERIAL_WEIGHT
        
        return score / max(len(materials), 1)  # Average per material
    
    def _score_temperature(self, item, weather):
        """Boost/penalize items based on current temperature."""
        article_type = _get_canonical_type(item)
        if not article_type:
            return 0.0
        
        if weather.is_hot:
            temp_rules = TEMP_ITEM_AFFINITY["hot"]
        elif weather.is_cold:
            temp_rules = TEMP_ITEM_AFFINITY["cold"]
        elif weather.is_rainy:
            temp_rules = TEMP_ITEM_AFFINITY["rainy"]
        else:
            temp_rules = TEMP_ITEM_AFFINITY["mild"]
        
        if article_type in temp_rules.get("boost", set()):
            return TEMP_ITEM_WEIGHT
        elif article_type in temp_rules.get("penalize", set()):
            return -TEMP_ITEM_WEIGHT
        return 0.0
```

### 12. `src/scoring/scorer.py` — Main Orchestrator

```python
"""
ContextScorer — the shared scoring orchestrator.

Both the feed pipeline and search reranker call this.
Combines age + weather scoring into a single adjustment.
"""

MAX_CONTEXT_ADJUSTMENT = 0.20  # Total cap across all context signals


class ContextScorer:
    """
    Orchestrates all context-aware scoring signals.
    
    Usage:
        scorer = ContextScorer()
        ctx = context_resolver.resolve(user_id, jwt_metadata, birthdate, profile)
        adjustment = scorer.score_item(item_dict, ctx)
        item["score"] += adjustment
    
    Or batch:
        items = scorer.score_items(items, ctx)  # Modifies in-place
    """
    
    def __init__(self):
        self._age_scorer = AgeScorer()
        self._weather_scorer = WeatherScorer()
    
    def score_item(self, item: dict, ctx: UserContext) -> float:
        """
        Compute total context-aware adjustment for one item.
        
        Returns float in range [-MAX_CONTEXT_ADJUSTMENT, +MAX_CONTEXT_ADJUSTMENT].
        """
        adjustment = 0.0
        
        # Age scoring
        if ctx.age_group:
            adjustment += self._age_scorer.score(
                item, ctx.age_group, ctx.coverage_prefs
            )
        
        # Weather scoring
        if ctx.weather:
            adjustment += self._weather_scorer.score(item, ctx.weather)
        
        # Cap total
        return max(-MAX_CONTEXT_ADJUSTMENT, min(MAX_CONTEXT_ADJUSTMENT, adjustment))
    
    def score_items(
        self,
        items: list,
        ctx: UserContext,
        score_field: str = "score",
        weight: float = 1.0,
    ) -> list:
        """
        Batch score items in-place.
        
        Adds `context_adjustment` field to each item.
        Modifies `score_field` by adding weighted adjustment.
        """
        for item in items:
            adj = self.score_item(item, ctx)
            item["context_adjustment"] = adj
            current = item.get(score_field, 0)
            if isinstance(current, (int, float)):
                item[score_field] = current + (adj * weight)
        return items
    
    def explain_item(self, item: dict, ctx: UserContext) -> dict:
        """
        Return detailed breakdown of scoring for debugging.
        Useful for Gradio UI / admin dashboard.
        """
        breakdown = {"total": 0.0}
        
        if ctx.age_group:
            age_adj = self._age_scorer.score(item, ctx.age_group, ctx.coverage_prefs)
            breakdown["age"] = age_adj
            breakdown["age_group"] = ctx.age_group.value
            breakdown["total"] += age_adj
        
        if ctx.weather:
            weather_adj = self._weather_scorer.score(item, ctx.weather)
            breakdown["weather"] = weather_adj
            breakdown["season"] = ctx.weather.season.value
            breakdown["temperature_c"] = ctx.weather.temperature_c
            breakdown["total"] += weather_adj
        
        breakdown["total"] = max(
            -MAX_CONTEXT_ADJUSTMENT,
            min(MAX_CONTEXT_ADJUSTMENT, breakdown["total"])
        )
        return breakdown
```

---

## Integration with Existing Systems

### A. Feed Pipeline (`src/recs/pipeline.py`)

Add to `get_feed_keyset()` after session scoring (Step 6b), before greedy reranker (Step 8):

```python
# Step 6c: Context scoring (age + weather)
from scoring.scorer import ContextScorer
from scoring.context_resolver import ContextResolver

# Built once per pipeline init
self._context_scorer = ContextScorer()
self._context_resolver = ContextResolver(weather_api_key=os.getenv("OPENWEATHER_API_KEY", ""))

# In get_feed_keyset():
user_context = self._context_resolver.resolve(
    user_id=user_id,
    jwt_user_metadata=None,  # Not available here; use cached address
    birthdate=user_state.onboarding_profile.birthdate if user_state.onboarding_profile else None,
    onboarding_profile=user_state.profile,
)

# Apply to candidates
for candidate in ranked_candidates:
    item_dict = candidate.to_scoring_dict()  # Need to add this method
    adj = self._context_scorer.score_item(item_dict, user_context)
    candidate.score = (candidate.score or 0) + adj
    candidate.context_adjustment = adj
```

**Note:** The feed pipeline doesn't have JWT metadata at the `get_feed_keyset()` level. We need to either:
- Pass `user_metadata` from the route handler down through the call chain, OR
- Cache the address on first resolve and look up by user_id thereafter

**Recommendation:** Add `user_metadata` as an optional parameter to `get_feed_keyset()` and have the API endpoint pass it from the JWT.

### B. Search Reranker (`src/search/reranker.py`)

Add context scoring as Step 3.75 (after profile scoring, before brand diversity):

```python
def rerank(self, results, user_profile=None, seen_ids=None,
           max_per_brand=4, session_scores=None, user_context=None):
    ...
    # Step 3.5: Session scoring (from previous plan)
    # Step 3.75: Context scoring (age + weather)
    if user_context:
        results = self._apply_context_scoring(results, user_context)
    # Step 4: Brand diversity
    ...

def _apply_context_scoring(self, results, user_context):
    scorer = ContextScorer()  # Lightweight, no state
    for item in results:
        adj = scorer.score_item(item, user_context)
        item["rrf_score"] = item.get("rrf_score", 0) + adj
        item["context_adjustment"] = adj
    results.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
    return results
```

### C. Search Route (`src/api/routes/search.py`)

The search route HAS the JWT user_metadata:

```python
def hybrid_search(request, user: SupabaseUser = Depends(require_auth)):
    ...
    # Build user context (age + weather)
    user_context = None
    try:
        from scoring.context_resolver import ContextResolver
        resolver = _get_context_resolver()  # Cached singleton
        user_context = resolver.resolve(
            user_id=user.id,
            jwt_user_metadata=user.user_metadata,
            birthdate=_get_user_birthdate(user.id),  # From profile cache
            onboarding_profile=user_profile,
        )
    except Exception:
        pass  # Context scoring is optional
    
    result = service.search(
        request=request,
        user_id=user.id,
        user_profile=user_profile,
        session_scores=session_scores,
        user_context=user_context,
    )
```

### D. Feed Endpoint (`src/recs/api_endpoints.py`)

Pass JWT metadata to pipeline:

```python
@v2_router.get("/feed/keyset")
def get_feed_keyset(..., user: SupabaseUser = Depends(require_auth)):
    pipeline = get_pipeline()
    result = pipeline.get_feed_keyset(
        ...,
        user_metadata=user.user_metadata,  # NEW parameter
    )
```

---

## Candidate Model Changes

### Add `to_scoring_dict()` to `Candidate` in `src/recs/models.py`

The `ContextScorer` works with plain dicts (shared between feed and search). Need a method to convert `Candidate` → scoring dict:

```python
def to_scoring_dict(self) -> dict:
    """Convert to dict format expected by scoring module."""
    return {
        "product_id": self.id,
        "article_type": self.article_type,
        "broad_category": self.broad_category,
        "brand": self.brand,
        "style_tags": self.style_tags,
        "occasions": self.occasions,
        "pattern": self.pattern,
        "formality": self.formality,
        "fit_type": self.fit,
        "neckline": self.neckline,
        "sleeve_type": self.sleeve,
        "length": self.length,
        "color_family": self.color_family,
        "seasons": self.seasons,
        "materials": self.materials,
        "image_url": self.image_url,
    }
```

---

## Helper Functions Needed

Both `age_scorer.py` and `weather_scorer.py` need these helpers (in a shared location, e.g., `src/scoring/item_utils.py`):

```python
def _get_canonical_type(item: dict) -> Optional[str]:
    """
    Get canonical article type from item dict.
    Tries: article_type → name-based inference → broad_category fallback.
    """
    at = (item.get("article_type") or "").lower().replace(" ", "_")
    if at:
        return at
    # Could add name-based inference here
    return None


def _get_broad_category(item: dict) -> Optional[str]:
    """Get broad category: tops, bottoms, dresses/one_piece, outerwear."""
    cat = (item.get("broad_category") or item.get("category") or "").lower()
    if cat in ("tops", "bottoms", "outerwear"):
        return cat
    if cat in ("dresses", "one_piece", "one-piece"):
        return "dresses"
    return None


def _detect_coverage_dimensions(item: dict) -> list:
    """Detect which coverage dimensions an item triggers."""
    dimensions = []
    
    article_type = _get_canonical_type(item)
    if article_type and article_type in ARTICLE_TYPE_COVERAGE:
        dimensions.append(ARTICLE_TYPE_COVERAGE[article_type])
    
    neckline = (item.get("neckline") or "").lower()
    if neckline in NECKLINE_COVERAGE:
        dimensions.append(NECKLINE_COVERAGE[neckline])
    
    # Check style_tags for coverage signals
    style_tags = item.get("style_tags") or []
    if isinstance(style_tags, str):
        style_tags = [style_tags]
    for tag in style_tags:
        tag_lower = tag.lower()
        if tag_lower in STYLE_TAG_COVERAGE:
            dimensions.append(STYLE_TAG_COVERAGE[tag_lower])
    
    # Check length for mini
    length = (item.get("length") or "").lower()
    if "mini" in length and "mini" not in dimensions:
        dimensions.append("mini")
    
    return list(set(dimensions))
```

---

## New Environment Variables

```bash
# Weather API (required for weather scoring; graceful degradation without it)
OPENWEATHER_API_KEY=your_openweathermap_api_key

# Redis (required for session scores — from previous plan)
REDIS_URL=redis://localhost:6379/0
```

Add to `src/config/settings.py`:
```python
openweather_api_key: str = Field("", description="OpenWeatherMap API key for weather scoring")
```

Add to `requirements.txt`:
```
requests>=2.31.0    # For OpenWeatherMap API calls (may already be a transitive dep)
redis>=5.0.0        # For session score persistence
```

---

## Scoring Weight Summary

### Age Scorer Weights (total cap: ±0.15)
| Signal | Weight | Notes |
|--------|--------|-------|
| Item frequency | ±0.05 | Boost common items, penalize uncommon |
| Style affinity | ±0.04 | Boost age-popular styles |
| Occasion affinity | ±0.03 | Boost age-popular occasions |
| Coverage tolerance | -0.08 max | Penalty only (never boosts revealing) |
| Fit preference | ±0.03 | Boost age-appropriate fits |
| Pattern loudness | ±0.02 | Boost age-appropriate patterns |

### Weather Scorer Weights (total cap: ±0.12)
| Signal | Weight | Notes |
|--------|--------|-------|
| Season match | ±0.06 | In-season boost, out-of-season penalty |
| Material match | ±0.04 | Weather-appropriate materials |
| Temperature × item type | ±0.05 | Don't show coats when it's 35°C |

### Overall Context Cap: ±0.20

### Comparison with Existing Scoring
| Scorer | Cap | Applied In |
|--------|-----|-----------|
| Profile boosts (search) | ±0.15 | search/reranker.py |
| Session scoring (search) | ±0.12 | search/reranker.py (planned) |
| Context scoring (shared) | ±0.20 | Both feed + search |
| Session × feed blend | 0.6/0.4 | recs/pipeline.py |

---

## Complete File List

### New Files (10 files)
| File | Est. Lines | Purpose |
|------|-----------|---------|
| `src/scoring/__init__.py` | ~20 | Public exports |
| `src/scoring/context.py` | ~55 | UserContext, AgeGroup, Season, WeatherContext |
| `src/scoring/context_resolver.py` | ~150 | Build UserContext from JWT + Supabase + Weather API |
| `src/scoring/scorer.py` | ~100 | ContextScorer orchestrator |
| `src/scoring/age_scorer.py` | ~180 | Age-affinity scoring |
| `src/scoring/weather_scorer.py` | ~120 | Weather/season scoring |
| `src/scoring/item_utils.py` | ~80 | Shared helpers (_get_canonical_type, etc.) |
| `src/scoring/constants/__init__.py` | ~5 | Package marker |
| `src/scoring/constants/age_item_frequency.py` | ~60 | Item frequency tables |
| `src/scoring/constants/age_style_affinity.py` | ~70 | Style rank tables + tag mapping |
| `src/scoring/constants/age_occasion_affinity.py` | ~50 | Occasion rank tables + mapping |
| `src/scoring/constants/age_coverage_tolerance.py` | ~70 | Coverage tolerance + detection maps |
| `src/scoring/constants/age_fit_preferences.py` | ~80 | Fit preferences by age × category |
| `src/scoring/constants/age_color_pattern.py` | ~60 | Color/pattern loudness tables |
| `src/scoring/constants/weather_materials.py` | ~70 | Material-season + temperature rules |

### Modified Files (7 files)
| File | Changes |
|------|---------|
| `src/recs/pipeline.py` | Add ContextScorer + ContextResolver init; add context scoring step to get_feed_keyset(); accept user_metadata param |
| `src/recs/models.py` | Add `to_scoring_dict()` to Candidate; add `context_adjustment` field |
| `src/recs/api_endpoints.py` | Pass user.user_metadata to pipeline.get_feed_keyset() |
| `src/search/reranker.py` | Add user_context param; add _apply_context_scoring() step |
| `src/search/hybrid_search.py` | Accept and forward user_context param |
| `src/api/routes/search.py` | Build UserContext; pass to service.search() |
| `src/config/settings.py` | Add openweather_api_key field |

### Test File
| File | Est. Lines | Coverage |
|------|-----------|---------|
| `tests/unit/test_context_scoring.py` | ~400 | AgeScorer (all 6 dimensions × 5 age groups), WeatherScorer (season/material/temp), ContextScorer (combined), ContextResolver (mock), UserContext building |

---

## Execution Order

1. Create `src/scoring/` package with `__init__.py`
2. Create `context.py` (dataclasses — no dependencies)
3. Create all `constants/` files (pure data — no dependencies)
4. Create `item_utils.py` (shared helpers)
5. Create `age_scorer.py` (depends on constants + item_utils)
6. Create `weather_scorer.py` (depends on constants + item_utils)
7. Create `scorer.py` (depends on age_scorer + weather_scorer)
8. Create `context_resolver.py` (depends on context.py + external APIs)
9. Write tests
10. Integrate into pipeline.py
11. Integrate into search reranker
12. Integrate into routes
13. Run full test suite

---

## Design Decisions

1. **Additive scoring, not multiplicative** — Context adjustments ADD to existing scores, they don't multiply. This means a great search/embedding match doesn't get zeroed out by being slightly out-of-season.

2. **Age defaults are priors, not hard filters** — A 45-year-old who actively engages with crop tops (via session scoring) will still see crop tops. The age penalty is small and gets overridden by behavioral signals.

3. **User explicit > age default > nothing** — Coverage preferences from onboarding (`no_revealing`, `styles_to_avoid`) are hard filters (feasibility filter). Age-based coverage penalties are soft scoring. Behavioral overrides (session scoring) can counteract age defaults.

4. **Weather degrades gracefully** — If no API key or weather fetch fails, falls back to season-from-date-and-hemisphere. If no address, skips weather entirely.

5. **Score caps are independent** — Age cap (±0.15) + Weather cap (±0.12) = max possible ±0.20 (capped by ContextScorer). This prevents context from dominating over relevance.

6. **Canonical type resolution** — Both scorers need article types. We reuse the `ARTICLE_TYPE_CANON` mapping from `feasibility_filter.py` to normalize Gemini types into canonical types.
