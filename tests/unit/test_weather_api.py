"""
Weather API & Weather Scoring Tests.

Comprehensive tests for:
1. _fetch_openweathermap() - OWM API parsing, edge cases, error handling
2. ContextResolver._resolve_weather() - cache hit, cache expiry, API fallback
3. ContextResolver._resolve_address() - JWT metadata, cached, Supabase fallback
4. WeatherScorer gaps - string inputs, negative cap, rainy+hot combo, mild path
5. Season derivation - all 12 months, both hemispheres, southern country variants
6. Coverage prefs - all 5 flags, styles_to_avoid edge cases
7. Full resolve() paths - country-only, no data, address extraction variants
8. Weather → feed/search integration - scoring effects on real item types

Run with: PYTHONPATH=src python -m pytest tests/unit/test_weather_api.py -v
"""

import json
import time
import threading
import pytest
from datetime import date
from unittest.mock import MagicMock, patch, PropertyMock

from scoring.context import Season, WeatherContext, AgeGroup, UserContext
from scoring.weather_scorer import (
    WeatherScorer,
    MAX_WEATHER_ADJUSTMENT,
    SEASON_MATCH_WEIGHT,
    MATERIAL_WEIGHT,
    TEMP_ITEM_WEIGHT,
)
from scoring.context_resolver import (
    ContextResolver,
    _extract_address_from_metadata,
    _compute_age,
    _age_to_group,
    _season_from_date_and_country,
    _extract_coverage_prefs,
    _fetch_openweathermap,
)


# =============================================================================
# Weather helpers
# =============================================================================

def _summer_hot():
    return WeatherContext(
        temperature_c=35.0, feels_like_c=38.0, condition="clear",
        humidity=40, wind_speed_mps=2.0, season=Season.SUMMER,
        is_hot=True, is_cold=False, is_mild=False, is_rainy=False,
    )

def _winter_cold():
    return WeatherContext(
        temperature_c=-5.0, feels_like_c=-10.0, condition="snow",
        humidity=70, wind_speed_mps=5.0, season=Season.WINTER,
        is_hot=False, is_cold=True, is_mild=False, is_rainy=False,
    )

def _spring_mild():
    return WeatherContext(
        temperature_c=18.0, feels_like_c=17.0, condition="clouds",
        humidity=55, wind_speed_mps=3.0, season=Season.SPRING,
        is_hot=False, is_cold=False, is_mild=True, is_rainy=False,
    )

def _rainy_fall():
    return WeatherContext(
        temperature_c=12.0, feels_like_c=10.0, condition="rain",
        humidity=85, wind_speed_mps=6.0, season=Season.FALL,
        is_hot=False, is_cold=False, is_mild=True, is_rainy=True,
    )

def _rainy_hot():
    """Tropical rain — hot + rainy simultaneously."""
    return WeatherContext(
        temperature_c=32.0, feels_like_c=36.0, condition="thunderstorm",
        humidity=90, wind_speed_mps=4.0, season=Season.SUMMER,
        is_hot=True, is_cold=False, is_mild=False, is_rainy=True,
    )


def _owm_response(
    temp=22.0, feels=21.0, condition="clear", humidity=60,
    wind=3.0, city="London", country="GB",
):
    """Build a realistic OpenWeatherMap JSON response dict."""
    return {
        "coord": {"lon": -0.13, "lat": 51.51},
        "weather": [{"id": 800, "main": condition, "description": "sky", "icon": "01d"}],
        "main": {
            "temp": temp, "feels_like": feels,
            "temp_min": temp - 1, "temp_max": temp + 1,
            "pressure": 1013, "humidity": humidity,
        },
        "wind": {"speed": wind, "deg": 200},
        "clouds": {"all": 10},
        "sys": {"country": country},
        "name": city,
        "cod": 200,
    }


# =============================================================================
# 1. _fetch_openweathermap()
# =============================================================================

class TestFetchOpenWeatherMap:
    """Tests for the raw OWM API call function."""

    def test_parses_clear_weather(self):
        """Standard clear-sky response parses correctly."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(
            temp=22.0, feels=21.0, condition="Clear", humidity=60, wind=3.5,
        )
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("London", "GB", "test-key")

        assert w.temperature_c == 22.0
        assert w.feels_like_c == 21.0
        assert w.condition == "clear"  # lowercased
        assert w.humidity == 60
        assert w.wind_speed_mps == 3.5
        assert w.is_mild is True
        assert w.is_hot is False
        assert w.is_cold is False
        assert w.is_rainy is False

    def test_parses_rain(self):
        """Rainy response sets is_rainy=True."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(condition="Rain", temp=15.0, feels=13.0)
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("London", "GB", "k")

        assert w.condition == "rain"
        assert w.is_rainy is True

    def test_parses_drizzle(self):
        """Drizzle is also rainy."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(condition="Drizzle", temp=14.0, feels=12.0)
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("London", "GB", "k")

        assert w.is_rainy is True

    def test_parses_thunderstorm(self):
        """Thunderstorm is also rainy."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(condition="Thunderstorm", temp=28.0, feels=30.0)
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("London", "GB", "k")

        assert w.is_rainy is True
        assert w.is_hot is True  # feels_like 30 > 25

    def test_parses_hot_weather(self):
        """Hot weather when feels_like > 25C."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(temp=30.0, feels=33.0)
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("Dubai", "AE", "k")

        assert w.is_hot is True
        assert w.is_cold is False
        assert w.is_mild is False

    def test_parses_cold_weather(self):
        """Cold weather when feels_like < 10C."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response(temp=5.0, feels=2.0)
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("Moscow", "RU", "k")

        assert w.is_cold is True
        assert w.is_hot is False

    def test_missing_weather_key_defaults_unknown(self):
        """Missing 'weather' key in response -> condition='unknown'."""
        data = _owm_response()
        del data["weather"]
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("X", "Y", "k")

        assert w.condition == "unknown"

    def test_missing_humidity_defaults_50(self):
        """Missing humidity defaults to 50."""
        data = _owm_response()
        del data["main"]["humidity"]
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("X", "Y", "k")

        assert w.humidity == 50

    def test_missing_wind_defaults_zero(self):
        """Missing wind defaults to 0."""
        data = _owm_response()
        del data["wind"]
        mock_resp = MagicMock()
        mock_resp.json.return_value = data
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp):
            w = _fetch_openweathermap("X", "Y", "k")

        assert w.wind_speed_mps == 0

    def test_api_error_raises(self):
        """HTTP errors should propagate (caller catches them)."""
        import requests
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.HTTPError("401 Unauthorized")

        with patch("requests.get", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                _fetch_openweathermap("X", "Y", "bad-key")

    def test_correct_params_sent(self):
        """Verify correct query params sent to OWM API."""
        mock_resp = MagicMock()
        mock_resp.json.return_value = _owm_response()
        mock_resp.raise_for_status = MagicMock()

        with patch("requests.get", return_value=mock_resp) as mock_get:
            _fetch_openweathermap("New York", "US", "my-api-key")

        mock_get.assert_called_once()
        call_kwargs = mock_get.call_args
        assert call_kwargs.kwargs["params"]["q"] == "New York,US"
        assert call_kwargs.kwargs["params"]["appid"] == "my-api-key"
        assert call_kwargs.kwargs["params"]["units"] == "metric"
        assert call_kwargs.kwargs["timeout"] == 5


# =============================================================================
# 2. ContextResolver._resolve_weather() - caching & fallback
# =============================================================================

class TestResolveWeather:
    """Tests for _resolve_weather() caching and fallback paths."""

    def test_api_success_returns_real_weather(self):
        """Successful API call returns parsed WeatherContext."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
            is_mild=True,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather):
            w = resolver._resolve_weather("London", "GB")

        assert w.temperature_c == 22.0
        assert w.condition == "clear"

    def test_cache_hit_skips_api(self):
        """Second call for same city uses cache, not API."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather) as mock_fetch:
            w1 = resolver._resolve_weather("London", "GB")
            w2 = resolver._resolve_weather("London", "GB")

        mock_fetch.assert_called_once()  # Only 1 API call
        assert w1.temperature_c == w2.temperature_c

    def test_different_cities_not_cached_together(self):
        """Different cities should make separate API calls."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather) as mock_fetch:
            resolver._resolve_weather("London", "GB")
            resolver._resolve_weather("Paris", "FR")

        assert mock_fetch.call_count == 2

    def test_cache_key_case_insensitive(self):
        """Cache key should be case-insensitive (london:gb == London:GB)."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather) as mock_fetch:
            resolver._resolve_weather("London", "GB")
            resolver._resolve_weather("london", "gb")

        mock_fetch.assert_called_once()

    def test_cache_ttl_expiry(self):
        """Expired cache should re-fetch from API."""
        resolver = ContextResolver(weather_api_key="test-key")
        resolver._WEATHER_TTL = 1  # 1 second TTL for test
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather) as mock_fetch:
            resolver._resolve_weather("London", "GB")
            # Manually expire the cache
            for k in resolver._weather_cache:
                resolver._weather_cache[k] = (time.time() - 10, resolver._weather_cache[k][1])
            resolver._resolve_weather("London", "GB")

        assert mock_fetch.call_count == 2

    def test_api_failure_falls_back_to_season(self):
        """API failure should fall back to season-from-date."""
        resolver = ContextResolver(weather_api_key="test-key")
        with patch("scoring.context_resolver._fetch_openweathermap", side_effect=Exception("timeout")):
            w = resolver._resolve_weather("London", "GB")

        assert w.temperature_c == 20.0  # Fallback default
        assert w.condition == "unknown"
        assert w.is_mild is True
        assert w.season is not None  # Derived from current date

    def test_no_api_key_uses_season_fallback(self):
        """No API key should skip API and use season fallback."""
        resolver = ContextResolver(weather_api_key="")
        with patch("scoring.context_resolver._fetch_openweathermap") as mock_fetch:
            w = resolver._resolve_weather("London", "GB")

        mock_fetch.assert_not_called()
        assert w.temperature_c == 20.0
        assert w.condition == "unknown"


# =============================================================================
# 3. Address resolution
# =============================================================================

class TestExtractAddress:
    """Tests for _extract_address_from_metadata."""

    def test_nested_address(self):
        """Nested address dict should be returned directly."""
        meta = {"address": {"city": "NYC", "country": "US", "state": "NY"}}
        addr = _extract_address_from_metadata(meta)
        assert addr == {"city": "NYC", "country": "US", "state": "NY"}

    def test_flat_fields(self):
        """Flat city/country fields should be assembled."""
        meta = {"city": "London", "country": "GB", "state": "England"}
        addr = _extract_address_from_metadata(meta)
        assert addr["city"] == "London"
        assert addr["country"] == "GB"
        assert addr["state"] == "England"

    def test_flat_with_zip(self):
        """Flat fields with 'zip' key."""
        meta = {"city": "NYC", "country": "US", "zip": "10001"}
        addr = _extract_address_from_metadata(meta)
        assert addr["zip"] == "10001"

    def test_flat_with_postal_code(self):
        """Flat fields with 'postal_code' key."""
        meta = {"city": "London", "country": "GB", "postal_code": "EC1A 1BB"}
        addr = _extract_address_from_metadata(meta)
        assert addr["zip"] == "EC1A 1BB"

    def test_empty_metadata(self):
        """Empty dict returns None."""
        assert _extract_address_from_metadata({}) is None

    def test_none_metadata(self):
        """None returns None."""
        assert _extract_address_from_metadata(None) is None

    def test_unrelated_metadata(self):
        """Metadata without address fields returns None."""
        meta = {"name": "John", "email": "john@example.com"}
        assert _extract_address_from_metadata(meta) is None

    def test_nested_address_non_dict(self):
        """'address' key with non-dict value falls through to flat check."""
        meta = {"address": "123 Main St", "city": "NYC", "country": "US"}
        addr = _extract_address_from_metadata(meta)
        assert addr["city"] == "NYC"


class TestResolveAddress:
    """Tests for ContextResolver._resolve_address()."""

    def test_jwt_metadata_preferred(self):
        """JWT metadata is used first when it has a city."""
        resolver = ContextResolver()
        addr = resolver._resolve_address("u1", {"city": "London", "country": "GB"})
        assert addr["city"] == "London"

    def test_jwt_without_city_falls_to_cache(self):
        """JWT without city should check cache."""
        resolver = ContextResolver()
        # Pre-populate cache
        resolver._address_cache["u1"] = (time.time(), {"city": "Paris", "country": "FR"})
        addr = resolver._resolve_address("u1", {"name": "John"})
        assert addr["city"] == "Paris"

    def test_cache_ttl_expiry(self):
        """Expired cache entry should trigger Supabase fallback."""
        resolver = ContextResolver()
        resolver._address_cache["u1"] = (time.time() - 100000, {"city": "Old", "country": "XX"})
        # Supabase fallback will fail (no real client), should return None
        with patch("config.database.get_supabase_client", side_effect=Exception("no db")):
            addr = resolver._resolve_address("u1", None)
        assert addr is None


# =============================================================================
# 4. WeatherScorer gaps
# =============================================================================

class TestWeatherScorerGaps:
    """Tests for previously uncovered WeatherScorer paths."""

    @pytest.fixture
    def scorer(self):
        return WeatherScorer()

    # -- String inputs --

    def test_season_as_string(self, scorer):
        """seasons field as a single string should still work."""
        item = {"article_type": "dress", "seasons": "Summer"}
        score = scorer._score_season(item, _summer_hot())
        assert score == SEASON_MATCH_WEIGHT

    def test_materials_as_string(self, scorer):
        """materials field as a single string should still work."""
        item = {"article_type": "dress", "materials": "linen"}
        score = scorer._score_materials(item, _summer_hot())
        assert score > 0

    def test_materials_whitespace_and_casing(self, scorer):
        """Materials should be normalized (stripped, lowered)."""
        item = {"article_type": "dress", "materials": ["  Wool ", " CASHMERE"]}
        score = scorer._score_materials(item, _summer_hot())
        assert score < 0  # Both bad for summer

    def test_materials_empty_strings_filtered(self, scorer):
        """Empty strings and None in materials list should be filtered."""
        item = {"article_type": "dress", "materials": ["", None, "cotton"]}
        score = scorer._score_materials(item, _summer_hot())
        # Only "cotton" counts (good for summer)
        assert score > 0

    def test_apparent_fabric_fallback(self, scorer):
        """apparent_fabric field used when materials is missing."""
        item = {"article_type": "dress", "apparent_fabric": "wool"}
        score = scorer._score_materials(item, _summer_hot())
        assert score < 0  # Wool bad for summer

    # -- Negative cap --

    def test_negative_cap(self, scorer):
        """Worst-case item (winter + wool + coat in summer) capped at -MAX."""
        item = {
            "article_type": "coat",
            "seasons": ["Winter"],
            "materials": ["wool", "cashmere", "fleece"],
        }
        score = scorer.score(item, _summer_hot())
        assert score == -MAX_WEATHER_ADJUSTMENT

    # -- Rainy paths --

    def test_rainy_jacket_boost(self, scorer):
        """Jacket boosted in rainy weather."""
        item = {"article_type": "jacket"}
        score = scorer._score_temperature(item, _rainy_fall())
        assert score == TEMP_ITEM_WEIGHT

    def test_rainy_fallthrough_to_temperature(self, scorer):
        """Rainy weather with non-rainy item falls through to temp rules."""
        # Dress is not in rainy boost/penalize, should fall through to mild rules
        item = {"article_type": "dress"}
        rainy_mild = _rainy_fall()
        score = scorer._score_temperature(item, rainy_mild)
        # "dress" is in mild boost, so should get TEMP_ITEM_WEIGHT
        assert score == TEMP_ITEM_WEIGHT

    def test_rainy_and_hot_combo(self, scorer):
        """Tropical rain: rainy first, then falls through to hot for non-rainy items."""
        # Tank top is NOT in rainy rules, so rainy check returns 0
        # Then falls through to hot rules where tank_top IS boosted
        item = {"article_type": "tank top"}
        score = scorer._score_temperature(item, _rainy_hot())
        assert score >= TEMP_ITEM_WEIGHT  # At least base weight (intensity may amplify)

    def test_rainy_and_hot_jacket(self, scorer):
        """Jacket in rainy+hot: rainy rules match first (jacket is rainy boost)."""
        item = {"article_type": "jacket"}
        score = scorer._score_temperature(item, _rainy_hot())
        # Jacket IS in rainy boost, so returns immediately without falling through
        assert score >= TEMP_ITEM_WEIGHT  # At least base weight (intensity may amplify)

    # -- Mild path --

    def test_mild_blazer_boost(self, scorer):
        """Blazer boosted in mild weather."""
        item = {"article_type": "blazer"}
        score = scorer._score_temperature(item, _spring_mild())
        assert score == TEMP_ITEM_WEIGHT

    def test_mild_puffer_penalized(self, scorer):
        """Puffer penalized in mild weather."""
        item = {"article_type": "puffer jacket"}
        score = scorer._score_temperature(item, _spring_mild())
        assert score == -TEMP_ITEM_WEIGHT

    def test_mild_dress_boost(self, scorer):
        """Dress boosted in mild weather."""
        item = {"article_type": "dress"}
        score = scorer._score_temperature(item, _spring_mild())
        assert score == TEMP_ITEM_WEIGHT

    # -- Unknown types --

    def test_unknown_article_type_neutral(self, scorer):
        """Unknown article type returns 0 for temperature scoring."""
        item = {"article_type": ""}
        score = scorer._score_temperature(item, _summer_hot())
        assert score == 0.0

    def test_check_rules_empty_dict(self, scorer):
        """_check_rules with empty rules dict returns 0."""
        assert scorer._check_rules("jacket", {}) == 0.0

    def test_check_rules_not_in_any_set(self, scorer):
        """_check_rules with item not in boost or penalize returns 0."""
        rules = {"boost": frozenset({"a"}), "penalize": frozenset({"b"})}
        assert scorer._check_rules("c", rules) == 0.0


# =============================================================================
# 5. Season derivation - all 12 months, both hemispheres
# =============================================================================

class TestSeasonDerivation:
    """Tests for _season_from_date_and_country across all months."""

    @pytest.mark.parametrize("month,expected", [
        (1, Season.WINTER), (2, Season.WINTER), (12, Season.WINTER),
        (3, Season.SPRING), (4, Season.SPRING), (5, Season.SPRING),
        (6, Season.SUMMER), (7, Season.SUMMER), (8, Season.SUMMER),
        (9, Season.FALL), (10, Season.FALL), (11, Season.FALL),
    ])
    def test_northern_hemisphere_all_months(self, month, expected):
        """Northern hemisphere season for each month."""
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, month, 15)
            mock_date.fromisoformat = date.fromisoformat
            result = _season_from_date_and_country("US")
        assert result == expected

    @pytest.mark.parametrize("month,expected", [
        (1, Season.SUMMER), (2, Season.SUMMER), (12, Season.SUMMER),
        (3, Season.FALL), (4, Season.FALL), (5, Season.FALL),
        (6, Season.WINTER), (7, Season.WINTER), (8, Season.WINTER),
        (9, Season.SPRING), (10, Season.SPRING), (11, Season.SPRING),
    ])
    def test_southern_hemisphere_all_months(self, month, expected):
        """Southern hemisphere season for each month."""
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, month, 15)
            mock_date.fromisoformat = date.fromisoformat
            result = _season_from_date_and_country("AU")
        assert result == expected

    @pytest.mark.parametrize("country", [
        "AU", "au", "australia", "NZ", "nz", "new zealand",
        "BR", "br", "brazil", "ZA", "za", "south africa",
        "AR", "ar", "argentina", "CL", "cl", "chile",
    ])
    def test_southern_country_variants(self, country):
        """Various southern hemisphere country codes/names recognized."""
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 15)
            mock_date.fromisoformat = date.fromisoformat
            result = _season_from_date_and_country(country)
        assert result == Season.SUMMER  # January = summer in south

    @pytest.mark.parametrize("country", [
        "US", "GB", "FR", "DE", "JP", "CA", "IT", "ES", "KR", "CN",
    ])
    def test_northern_countries(self, country):
        """Northern hemisphere countries get winter in January."""
        with patch("scoring.context_resolver.date") as mock_date:
            mock_date.today.return_value = date(2026, 1, 15)
            mock_date.fromisoformat = date.fromisoformat
            result = _season_from_date_and_country(country)
        assert result == Season.WINTER


# =============================================================================
# 6. Coverage preferences
# =============================================================================

class TestCoveragePrefs:
    """Tests for _extract_coverage_prefs."""

    def test_all_five_flags(self):
        """All 5 coverage flags recognized."""
        profile = {
            "no_sleeveless": True,
            "no_tanks": True,
            "no_crop": True,
            "no_athletic": True,
            "no_revealing": True,
        }
        prefs = _extract_coverage_prefs(profile)
        assert set(prefs) == {"no_sleeveless", "no_tanks", "no_crop", "no_athletic", "no_revealing"}

    def test_false_flags_excluded(self):
        """False flags should not be included."""
        profile = {"no_sleeveless": False, "no_crop": True}
        prefs = _extract_coverage_prefs(profile)
        assert prefs == ["no_crop"]

    def test_missing_flags_excluded(self):
        """Missing flags default to not included."""
        profile = {"no_crop": True}
        prefs = _extract_coverage_prefs(profile)
        assert prefs == ["no_crop"]

    def test_styles_to_avoid_appended(self):
        """styles_to_avoid list is appended to coverage prefs."""
        profile = {"no_crop": True, "styles_to_avoid": ["bodycon", "athleisure"]}
        prefs = _extract_coverage_prefs(profile)
        assert "no_crop" in prefs
        assert "bodycon" in prefs
        assert "athleisure" in prefs

    def test_styles_to_avoid_non_list_ignored(self):
        """Non-list styles_to_avoid should be ignored."""
        profile = {"styles_to_avoid": "bodycon"}
        prefs = _extract_coverage_prefs(profile)
        assert "bodycon" not in prefs

    def test_styles_to_avoid_none(self):
        """None styles_to_avoid should be handled."""
        profile = {"styles_to_avoid": None}
        prefs = _extract_coverage_prefs(profile)
        assert prefs == []

    def test_empty_profile(self):
        """Empty profile returns empty list."""
        assert _extract_coverage_prefs({}) == []


# =============================================================================
# 7. Full resolve() paths
# =============================================================================

class TestResolverFullPaths:
    """Tests for ContextResolver.resolve() edge cases."""

    def test_country_only_no_city(self):
        """Country without city: _resolve_address requires city, so weather needs
        the address to have both city+country. Use nested address to bypass."""
        resolver = ContextResolver()
        # Use nested address format where city is set but empty-ish
        # _resolve_address requires addr.get("city") to be truthy
        # So country-only via flat fields won't set ctx.country.
        # Instead, test the actual path: city+country both set
        ctx = resolver.resolve(
            user_id="u1",
            jwt_user_metadata={"city": "Unknown", "country": "US"},
        )
        assert ctx.weather is not None
        assert ctx.country == "US"

    def test_country_only_no_city_no_weather(self):
        """Country without city via flat JWT metadata: address resolver
        requires city, so no address is set and no weather is produced."""
        resolver = ContextResolver()
        ctx = resolver.resolve(
            user_id="u1",
            jwt_user_metadata={"country": "US"},
        )
        # _resolve_address returns None because addr.get("city") is None
        assert ctx.country is None
        assert ctx.weather is None

    def test_no_data_at_all(self):
        """No metadata, no birthdate, no profile -> minimal context."""
        resolver = ContextResolver()
        ctx = resolver.resolve(user_id="u1")
        assert ctx.age_group is None
        assert ctx.weather is None
        assert ctx.coverage_prefs == []

    def test_full_context_all_fields(self):
        """Full metadata + birthdate + profile resolves everything."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
            is_mild=True,
        )
        with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather):
            ctx = resolver.resolve(
                user_id="u1",
                jwt_user_metadata={"city": "London", "country": "GB"},
                birthdate="1990-06-15",
                onboarding_profile={
                    "no_sleeveless": True,
                    "no_crop": True,
                    "modesty": "moderate",
                },
            )
        assert ctx.age_years == 35
        assert ctx.age_group == AgeGroup.MID_CAREER
        assert ctx.city == "London"
        assert ctx.country == "GB"
        assert ctx.weather.temperature_c == 22.0
        assert "no_sleeveless" in ctx.coverage_prefs
        assert "no_crop" in ctx.coverage_prefs
        assert ctx.modesty_level == "moderate"

    def test_invalid_birthdate_still_resolves_weather(self):
        """Invalid birthdate should not block weather resolution."""
        resolver = ContextResolver()
        ctx = resolver.resolve(
            user_id="u1",
            jwt_user_metadata={"city": "London", "country": "GB"},
            birthdate="not-a-date",
        )
        assert ctx.age_group is None
        assert ctx.weather is not None  # Weather still resolved

    def test_modesty_level_alt_key(self):
        """'modesty_level' key should also be recognized."""
        resolver = ContextResolver()
        ctx = resolver.resolve(
            user_id="u1",
            onboarding_profile={"modesty_level": "conservative"},
        )
        assert ctx.modesty_level == "conservative"

    def test_zip_code_from_address(self):
        """Zip code extracted from JWT metadata."""
        resolver = ContextResolver()
        ctx = resolver.resolve(
            user_id="u1",
            jwt_user_metadata={"city": "NYC", "country": "US", "zip": "10001"},
        )
        assert ctx.zip_code == "10001"


# =============================================================================
# 8. Weather → Scoring Integration
# =============================================================================

class TestWeatherScoringIntegration:
    """Tests verifying weather scoring produces correct directional effects."""

    @pytest.fixture
    def scorer(self):
        return WeatherScorer()

    def test_winter_coat_vs_tank_top(self, scorer):
        """In cold winter: coat should score much higher than tank top."""
        coat = {"article_type": "coat", "seasons": ["Winter"], "materials": ["wool"]}
        tank = {"article_type": "tank top", "seasons": ["Summer"], "materials": ["cotton"]}
        coat_score = scorer.score(coat, _winter_cold())
        tank_score = scorer.score(tank, _winter_cold())
        assert coat_score > 0
        assert tank_score < 0
        assert coat_score > tank_score

    def test_summer_sundress_vs_puffer(self, scorer):
        """In hot summer: sundress should score higher than puffer."""
        sundress = {"article_type": "sundress", "seasons": ["Summer"], "materials": ["linen"]}
        puffer = {"article_type": "puffer jacket", "seasons": ["Winter"], "materials": ["down"]}
        sundress_score = scorer.score(sundress, _summer_hot())
        puffer_score = scorer.score(puffer, _summer_hot())
        assert sundress_score > 0
        assert puffer_score < 0

    def test_rainy_day_jacket_vs_silk_dress(self, scorer):
        """In rain: jacket should score higher than delicate silk dress."""
        jacket = {"article_type": "jacket", "seasons": ["Fall", "Spring"], "materials": ["polyester"]}
        silk = {"article_type": "slip dress", "seasons": ["Summer"], "materials": ["silk"]}
        jacket_score = scorer.score(jacket, _rainy_fall())
        silk_score = scorer.score(silk, _rainy_fall())
        assert jacket_score > silk_score

    def test_all_season_cotton_tshirt_neutral_in_any_weather(self, scorer):
        """All-season cotton tee should be roughly neutral in any weather."""
        tee = {
            "article_type": "t-shirt",
            "seasons": ["Spring", "Summer", "Fall", "Winter"],
            "materials": ["cotton"],
        }
        for weather_fn in [_summer_hot, _winter_cold, _spring_mild, _rainy_fall]:
            score = scorer.score(tee, weather_fn())
            # Season = 0 (all-season), material varies, temp varies
            # Should not be extremely positive or negative
            assert -MAX_WEATHER_ADJUSTMENT <= score <= MAX_WEATHER_ADJUSTMENT

    def test_context_scorer_weather_only(self):
        """ContextScorer with weather only (no age) still produces adjustments."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = UserContext(user_id="u1")
        ctx.weather = _winter_cold()

        coat = {"article_type": "coat", "seasons": ["Winter"], "materials": ["wool"]}
        tank = {"article_type": "tank top", "seasons": ["Summer"], "materials": ["mesh"]}

        coat_adj = scorer.score_item(coat, ctx)
        tank_adj = scorer.score_item(tank, ctx)
        assert coat_adj > tank_adj

    def test_context_scorer_batch_weather(self):
        """score_items() applies weather scoring to a batch."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = UserContext(user_id="u1")
        ctx.weather = _summer_hot()

        items = [
            {"article_type": "sundress", "seasons": ["Summer"], "materials": ["linen"], "score": 0.5},
            {"article_type": "coat", "seasons": ["Winter"], "materials": ["wool"], "score": 0.5},
        ]
        scorer.score_items(items, ctx, score_field="score")
        # Sundress should be boosted, coat penalized
        assert items[0]["score"] > items[1]["score"]

    def test_reranker_weather_affects_search_ordering(self):
        """Weather scoring in search reranker changes result order."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()
        ctx = UserContext(user_id="u1")
        ctx.weather = _winter_cold()

        coat = {
            "product_id": "coat-1", "name": "Warm Coat", "brand": "BrandA",
            "article_type": "coat", "seasons": ["Winter"], "materials": ["wool"],
            "image_url": "https://img/coat.jpg",
            "rrf_score": 0.10,
        }
        tank = {
            "product_id": "tank-1", "name": "Summer Tank", "brand": "BrandB",
            "article_type": "tank top", "seasons": ["Summer"], "materials": ["mesh"],
            "image_url": "https://img/tank.jpg",
            "rrf_score": 0.10,  # Same base score
        }
        reranked = reranker.rerank([tank, coat], user_context=ctx)
        # Coat should rank higher in winter
        assert reranked[0]["product_id"] == "coat-1"

    def test_dubai_summer_vs_nyc_winter_different_scoring(self):
        """Same item scored differently for Dubai (hot) vs NYC (cold)."""
        scorer = WeatherScorer()
        tank = {"article_type": "tank top", "seasons": ["Summer"], "materials": ["cotton"]}

        dubai = WeatherContext(
            temperature_c=40.0, feels_like_c=42.0, condition="clear",
            humidity=30, wind_speed_mps=2.0, season=Season.SUMMER,
            is_hot=True,
        )
        nyc = WeatherContext(
            temperature_c=-5.0, feels_like_c=-10.0, condition="snow",
            humidity=60, wind_speed_mps=5.0, season=Season.WINTER,
            is_cold=True,
        )
        dubai_score = scorer.score(tank, dubai)
        nyc_score = scorer.score(tank, nyc)
        assert dubai_score > 0  # Tank good for hot Dubai
        assert nyc_score < 0    # Tank bad for cold NYC
        assert dubai_score > nyc_score


# =============================================================================
# 9. Thread Safety
# =============================================================================

class TestThreadSafety:
    """Basic thread-safety tests for ContextResolver caching."""

    def test_concurrent_weather_resolve(self):
        """Multiple threads resolving weather for same city shouldn't crash."""
        resolver = ContextResolver(weather_api_key="test-key")
        mock_weather = WeatherContext(
            temperature_c=22.0, feels_like_c=21.0, condition="clear",
            humidity=60, wind_speed_mps=3.0, season=Season.WINTER,
        )
        errors = []

        def resolve_weather(city):
            try:
                with patch("scoring.context_resolver._fetch_openweathermap", return_value=mock_weather):
                    w = resolver._resolve_weather(city, "GB")
                    assert w.temperature_c == 22.0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve_weather, args=(f"City{i % 3}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_address_resolve(self):
        """Multiple threads resolving addresses shouldn't crash."""
        resolver = ContextResolver()
        errors = []

        def resolve(uid):
            try:
                resolver._resolve_address(uid, {"city": f"City{uid}", "country": "US"})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=resolve, args=(f"u{i}",)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
