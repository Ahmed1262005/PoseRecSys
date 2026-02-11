"""
User context dataclasses for shared scoring.

Defines the core types that both age_scorer and weather_scorer operate on.
These are built once per request by ContextResolver and passed through
to all scoring components.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


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
    temperature_c: float
    feels_like_c: float
    condition: str                     # "clear", "rain", "snow", "clouds", etc.
    humidity: int                      # 0-100
    wind_speed_mps: float
    season: Season
    is_hot: bool = False               # feels_like > 25C
    is_cold: bool = False              # feels_like < 10C
    is_mild: bool = False              # 10-25C
    is_rainy: bool = False             # rain/drizzle/thunderstorm


@dataclass
class UserContext:
    """
    Complete user context for scoring. Built once per request.

    Both the feed pipeline and search reranker receive this and pass it
    to ContextScorer.score_item() for each candidate/result.
    """
    user_id: str
    age_group: Optional[AgeGroup] = None
    age_years: Optional[int] = None
    city: Optional[str] = None
    state: Optional[str] = None
    country: Optional[str] = None
    zip_code: Optional[str] = None
    weather: Optional[WeatherContext] = None
    # Derived from onboarding profile
    coverage_prefs: List[str] = field(default_factory=list)
    modesty_level: Optional[str] = None
