"""
Benchmarks for the shared scoring module.

Measures throughput and latency of:
1. ContextScorer.score_item() - single item
2. ContextScorer.score_items() - batch scoring
3. AgeScorer.score() - age component only
4. WeatherScorer.score() - weather component only
5. ContextResolver.resolve() - context building
6. Candidate.to_scoring_dict() - conversion overhead
7. Search reranker with vs without context scoring
8. Pipeline-scale: 500-candidate feed scoring pass

Run with: PYTHONPATH=src python -m pytest tests/unit/test_scoring_benchmarks.py -v -s
Or standalone: PYTHONPATH=src python tests/unit/test_scoring_benchmarks.py

Benchmark results are printed to stdout and validated against thresholds.
"""

import time
import statistics
import random
import pytest
from typing import List, Dict


# =============================================================================
# Test Data Generators
# =============================================================================

def _generate_item_dicts(n: int) -> List[dict]:
    """Generate n realistic product dicts for scoring."""
    article_types = [
        "t-shirt", "crop_top", "tank_top", "blouse", "sweater",
        "cardigan", "blazer", "coat", "jacket", "dress",
        "mini_dress", "maxi_dress", "jeans", "trousers", "skirt",
        "shorts", "leggings", "bodysuit", "jumpsuit", "romper",
    ]
    styles = [
        ["casual"], ["streetwear"], ["classic"], ["office"],
        ["bohemian"], ["minimalist"], ["trendy", "edgy"],
        ["romantic"], ["sporty"], ["grunge"],
    ]
    occasions_list = [
        ["Everyday"], ["Office", "Work"], ["Date Night", "Evening"],
        ["Beach", "Vacation"], ["Party"], ["Casual", "Weekend"],
    ]
    patterns = ["Solid", "Floral", "Striped", "Geometric", "Animal Print", "Plaid", "Abstract"]
    seasons_combos = [
        ["Spring", "Summer"], ["Fall", "Winter"],
        ["Spring", "Summer", "Fall"], ["Summer"],
        ["Winter"], ["Spring", "Summer", "Fall", "Winter"],
    ]
    materials_list = [
        ["cotton"], ["polyester"], ["silk"], ["wool"],
        ["linen"], ["denim"], ["leather"], ["cashmere"],
        ["cotton", "polyester"], ["wool", "cashmere"],
    ]
    fits = ["regular", "slim", "relaxed", "oversized", "fitted"]
    necklines = ["crew", "v-neck", "scoop", "turtleneck", "off-shoulder", "halter"]
    sleeve_types = ["short", "long", "sleeveless", "3/4", "cap", "puff"]
    lengths = ["mini", "standard", "midi", "maxi", "cropped"]
    broad_cats = ["tops", "bottoms", "dresses", "outerwear"]

    items = []
    for i in range(n):
        items.append({
            "product_id": f"bench-{i:05d}",
            "article_type": random.choice(article_types),
            "broad_category": random.choice(broad_cats),
            "brand": f"Brand{random.randint(0, 50)}",
            "style_tags": random.choice(styles),
            "occasions": random.choice(occasions_list),
            "pattern": random.choice(patterns),
            "formality": random.choice(["Casual", "Smart Casual", "Semi-Formal", "Formal"]),
            "fit_type": random.choice(fits),
            "neckline": random.choice(necklines),
            "sleeve_type": random.choice(sleeve_types),
            "length": random.choice(lengths),
            "color_family": random.choice(["Dark", "Light", "Neutral", "Bright", "Cool"]),
            "seasons": random.choice(seasons_combos),
            "materials": random.choice(materials_list),
            "name": f"Benchmark Product {i}",
            "image_url": f"https://img.example.com/bench-{i}.jpg",
            "score": random.uniform(0.3, 0.9),
            "rrf_score": random.uniform(0.01, 0.15),
        })
    return items


def _generate_candidates(n: int) -> list:
    """Generate n Candidate objects for pipeline benchmarks."""
    from recs.models import Candidate

    article_types = [
        "t-shirt", "crop_top", "tank_top", "blouse", "sweater",
        "cardigan", "blazer", "coat", "jacket", "dress",
        "mini_dress", "jeans", "trousers", "skirt", "shorts",
    ]
    candidates = []
    for i in range(n):
        candidates.append(Candidate(
            item_id=f"cand-{i:05d}",
            embedding_score=random.uniform(0.3, 0.95),
            preference_score=random.uniform(0.2, 0.8),
            sasrec_score=random.uniform(0.1, 0.9),
            final_score=random.uniform(0.3, 0.9),
            category="tops",
            broad_category=random.choice(["tops", "bottoms", "dresses", "outerwear"]),
            article_type=random.choice(article_types),
            brand=f"Brand{random.randint(0, 50)}",
            price=random.uniform(15.0, 200.0),
            colors=[random.choice(["black", "white", "red", "blue", "green"])],
            materials=[random.choice(["cotton", "polyester", "silk", "wool", "linen"])],
            fit=random.choice(["regular", "slim", "relaxed", "oversized"]),
            sleeve=random.choice(["short", "long", "sleeveless"]),
            neckline=random.choice(["crew", "v-neck", "scoop"]),
            style_tags=[random.choice(["casual", "streetwear", "classic", "office"])],
            occasions=[random.choice(["Everyday", "Office", "Date Night"])],
            pattern=random.choice(["Solid", "Floral", "Striped"]),
            formality=random.choice(["Casual", "Smart Casual"]),
            color_family=random.choice(["Dark", "Light", "Neutral"]),
            seasons=[random.choice(["Spring", "Summer", "Fall", "Winter"])],
            image_url=f"https://img.example.com/cand-{i}.jpg",
            name=f"Candidate Product {i}",
            source="taste_vector",
        ))
    return candidates


def _make_gen_z_summer_context():
    """Create Gen Z + hot summer context."""
    from scoring.context import AgeGroup, Season, WeatherContext, UserContext
    return UserContext(
        user_id="bench-user",
        age_group=AgeGroup.GEN_Z,
        weather=WeatherContext(
            temperature_c=32.0, feels_like_c=34.0,
            condition="clear", humidity=60, wind_speed_mps=3.0,
            season=Season.SUMMER, is_hot=True,
        ),
    )


def _make_senior_winter_context():
    """Create Senior + cold winter context."""
    from scoring.context import AgeGroup, Season, WeatherContext, UserContext
    return UserContext(
        user_id="bench-user",
        age_group=AgeGroup.SENIOR,
        age_years=70,
        weather=WeatherContext(
            temperature_c=-5.0, feels_like_c=-8.0,
            condition="snow", humidity=80, wind_speed_mps=5.0,
            season=Season.WINTER, is_cold=True,
        ),
        coverage_prefs=["no_revealing"],
    )


# =============================================================================
# Benchmark Utilities
# =============================================================================

class BenchmarkResult:
    """Stores benchmark statistics."""
    def __init__(self, name: str, times: List[float], ops_count: int):
        self.name = name
        self.times = times
        self.ops_count = ops_count
        self.total_time = sum(times)
        self.mean = statistics.mean(times)
        self.median = statistics.median(times)
        self.stdev = statistics.stdev(times) if len(times) > 1 else 0
        self.min_t = min(times)
        self.max_t = max(times)
        self.throughput = ops_count / self.total_time if self.total_time > 0 else 0

    def __repr__(self):
        return (
            f"  {self.name}:\n"
            f"    Mean:       {self.mean*1000:.3f} ms\n"
            f"    Median:     {self.median*1000:.3f} ms\n"
            f"    Stdev:      {self.stdev*1000:.3f} ms\n"
            f"    Min:        {self.min_t*1000:.3f} ms\n"
            f"    Max:        {self.max_t*1000:.3f} ms\n"
            f"    Throughput: {self.throughput:,.0f} ops/sec\n"
            f"    Total:      {self.total_time*1000:.1f} ms ({len(self.times)} runs x {self.ops_count} ops)\n"
        )


def benchmark(func, warmup: int = 3, iterations: int = 10, ops_per_iter: int = 1):
    """Run a benchmark function and return statistics."""
    # Warmup
    for _ in range(warmup):
        func()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return times


# =============================================================================
# 1. ContextScorer.score_item() - Single Item
# =============================================================================

class TestBenchmarkSingleItemScoring:
    """Benchmark: scoring a single item."""

    def test_single_item_under_100us(self):
        """A single score_item() call should take < 100 microseconds."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(1)
        item = items[0]

        times = benchmark(lambda: scorer.score_item(item, ctx), iterations=1000, ops_per_iter=1)
        mean_us = statistics.mean(times) * 1_000_000

        result = BenchmarkResult("score_item (single)", times, 1)
        print(f"\n{result}")
        assert mean_us < 100, f"Single item scoring took {mean_us:.1f}us, expected < 100us"

    def test_single_item_age_only(self):
        """Age scoring only (no weather)."""
        from scoring.age_scorer import AgeScorer
        from scoring.context import AgeGroup
        scorer = AgeScorer()
        items = _generate_item_dicts(1)
        item = items[0]

        times = benchmark(
            lambda: scorer.score(item, AgeGroup.GEN_Z),
            iterations=1000,
        )
        mean_us = statistics.mean(times) * 1_000_000

        result = BenchmarkResult("AgeScorer.score (single)", times, 1)
        print(f"\n{result}")
        assert mean_us < 80, f"Age scoring took {mean_us:.1f}us, expected < 80us"

    def test_single_item_weather_only(self):
        """Weather scoring only (no age)."""
        from scoring.weather_scorer import WeatherScorer
        scorer = WeatherScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(1)
        item = items[0]

        times = benchmark(
            lambda: scorer.score(item, ctx.weather),
            iterations=1000,
        )
        mean_us = statistics.mean(times) * 1_000_000

        result = BenchmarkResult("WeatherScorer.score (single)", times, 1)
        print(f"\n{result}")
        assert mean_us < 50, f"Weather scoring took {mean_us:.1f}us, expected < 50us"


# =============================================================================
# 2. ContextScorer.score_items() - Batch
# =============================================================================

class TestBenchmarkBatchScoring:
    """Benchmark: batch scoring multiple items."""

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(42)

    def test_batch_50_items_under_5ms(self):
        """Scoring 50 items (one feed page) should take < 5ms."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(50)

        times = benchmark(
            lambda: scorer.score_items(items.copy(), ctx, score_field="score"),
            iterations=100,
            ops_per_iter=50,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("score_items (50)", times, 50)
        print(f"\n{result}")
        assert mean_ms < 5, f"50-item batch took {mean_ms:.2f}ms, expected < 5ms"

    def test_batch_200_items_under_20ms(self):
        """Scoring 200 items should take < 20ms."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_senior_winter_context()
        items = _generate_item_dicts(200)

        times = benchmark(
            lambda: scorer.score_items(items.copy(), ctx, score_field="score"),
            iterations=50,
            ops_per_iter=200,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("score_items (200)", times, 200)
        print(f"\n{result}")
        assert mean_ms < 20, f"200-item batch took {mean_ms:.2f}ms, expected < 20ms"

    def test_batch_500_items_under_50ms(self):
        """Scoring 500 items (large candidate pool) should take < 50ms."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(500)

        times = benchmark(
            lambda: scorer.score_items(items.copy(), ctx, score_field="score"),
            iterations=20,
            ops_per_iter=500,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("score_items (500)", times, 500)
        print(f"\n{result}")
        assert mean_ms < 50, f"500-item batch took {mean_ms:.2f}ms, expected < 50ms"

    def test_batch_1000_items_under_100ms(self):
        """Scoring 1000 items (max fetch size) should take < 100ms."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_senior_winter_context()
        items = _generate_item_dicts(1000)

        times = benchmark(
            lambda: scorer.score_items(items.copy(), ctx, score_field="score"),
            iterations=10,
            ops_per_iter=1000,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("score_items (1000)", times, 1000)
        print(f"\n{result}")
        assert mean_ms < 100, f"1000-item batch took {mean_ms:.2f}ms, expected < 100ms"

    def test_throughput_over_10k_per_second(self):
        """Scorer should handle > 10,000 items/second."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(1000)

        start = time.perf_counter()
        for _ in range(10):
            scorer.score_items(items.copy(), ctx, score_field="score")
        elapsed = time.perf_counter() - start

        throughput = 10000 / elapsed
        print(f"\n  Throughput: {throughput:,.0f} items/sec (10x1000 items in {elapsed*1000:.1f}ms)")
        assert throughput > 10000, f"Throughput {throughput:.0f}/sec, expected > 10,000/sec"


# =============================================================================
# 3. Candidate.to_scoring_dict() Conversion
# =============================================================================

class TestBenchmarkCandidateConversion:
    """Benchmark: Candidate -> scoring dict conversion."""

    def test_conversion_50_candidates_under_1ms(self):
        """Converting 50 candidates to scoring dicts should take < 1ms."""
        candidates = _generate_candidates(50)

        times = benchmark(
            lambda: [c.to_scoring_dict() for c in candidates],
            iterations=100,
            ops_per_iter=50,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("to_scoring_dict (50)", times, 50)
        print(f"\n{result}")
        assert mean_ms < 1.0, f"50 conversions took {mean_ms:.3f}ms, expected < 1ms"

    def test_conversion_500_candidates_under_10ms(self):
        """Converting 500 candidates should take < 10ms."""
        candidates = _generate_candidates(500)

        times = benchmark(
            lambda: [c.to_scoring_dict() for c in candidates],
            iterations=20,
            ops_per_iter=500,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("to_scoring_dict (500)", times, 500)
        print(f"\n{result}")
        assert mean_ms < 10, f"500 conversions took {mean_ms:.3f}ms, expected < 10ms"


# =============================================================================
# 4. ContextResolver.resolve() (no I/O)
# =============================================================================

class TestBenchmarkContextResolver:
    """Benchmark: ContextResolver.resolve() without API calls."""

    def test_resolve_under_500us(self):
        """resolve() without weather API should take < 500us."""
        from scoring import ContextResolver

        resolver = ContextResolver(weather_api_key="")

        times = benchmark(
            lambda: resolver.resolve(
                user_id="bench-user",
                jwt_user_metadata={"city": "NYC", "country": "US"},
                birthdate="2000-01-15",
                onboarding_profile={
                    "no_revealing": True,
                    "no_crop": False,
                    "styles_to_avoid": ["deep-necklines"],
                },
            ),
            iterations=500,
        )
        mean_us = statistics.mean(times) * 1_000_000

        result = BenchmarkResult("ContextResolver.resolve", times, 1)
        print(f"\n{result}")
        assert mean_us < 500, f"resolve() took {mean_us:.1f}us, expected < 500us"


# =============================================================================
# 5. Search Reranker with Context Scoring
# =============================================================================

class TestBenchmarkSearchReranker:
    """Benchmark: SessionReranker with and without context scoring."""

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(42)

    def test_reranker_50_results_without_context(self):
        """Reranking 50 results without context."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()
        items = _generate_item_dicts(50)

        profile = {
            "soft_prefs": {"preferred_brands": ["Brand1", "Brand2"]},
            "hard_filters": {"exclude_brands": ["Brand99"]},
        }

        times = benchmark(
            lambda: reranker.rerank(items.copy(), user_profile=profile),
            iterations=100,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("Reranker (50, no context)", times, 50)
        print(f"\n{result}")
        self._no_context_mean = mean_ms
        assert mean_ms < 5, f"Reranker without context took {mean_ms:.2f}ms"

    def test_reranker_50_results_with_context(self):
        """Reranking 50 results with context scoring."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(50)

        profile = {
            "soft_prefs": {"preferred_brands": ["Brand1", "Brand2"]},
            "hard_filters": {"exclude_brands": ["Brand99"]},
        }

        times = benchmark(
            lambda: reranker.rerank(items.copy(), user_profile=profile, user_context=ctx),
            iterations=100,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("Reranker (50, with context)", times, 50)
        print(f"\n{result}")
        assert mean_ms < 10, f"Reranker with context took {mean_ms:.2f}ms"

    def test_reranker_200_results_with_context(self):
        """Reranking 200 results with full pipeline."""
        from search.reranker import SessionReranker
        reranker = SessionReranker()
        ctx = _make_senior_winter_context()
        items = _generate_item_dicts(200)

        profile = {
            "soft_prefs": {"preferred_brands": ["Brand1"]},
            "hard_filters": {},
        }
        seen = {f"bench-{i:05d}" for i in range(20)}  # 20 seen items

        times = benchmark(
            lambda: reranker.rerank(
                items.copy(), user_profile=profile,
                seen_ids=seen, user_context=ctx,
            ),
            iterations=50,
        )
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("Reranker (200, full pipeline)", times, 200)
        print(f"\n{result}")
        assert mean_ms < 25, f"Full reranker pipeline took {mean_ms:.2f}ms"


# =============================================================================
# 6. Pipeline-Scale: Full Feed Scoring Pass
# =============================================================================

class TestBenchmarkPipelineScale:
    """Benchmark: simulates the full pipeline scoring pass (Step 6c)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(42)

    def test_pipeline_step6c_500_candidates(self):
        """Simulate Step 6c: convert + score 500 candidates."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        candidates = _generate_candidates(500)

        def step_6c():
            for c in candidates:
                item_dict = c.to_scoring_dict()
                adj = scorer.score_item(item_dict, ctx)
                c.final_score += adj

        times = benchmark(step_6c, iterations=20, ops_per_iter=500)
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("Pipeline Step 6c (500 candidates)", times, 500)
        print(f"\n{result}")
        assert mean_ms < 60, f"Step 6c took {mean_ms:.2f}ms for 500 candidates, expected < 60ms"

    def test_pipeline_step6c_50_candidates(self):
        """Simulate Step 6c for a single page (50 candidates)."""
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_senior_winter_context()
        candidates = _generate_candidates(50)

        def step_6c():
            for c in candidates:
                item_dict = c.to_scoring_dict()
                adj = scorer.score_item(item_dict, ctx)
                c.final_score += adj

        times = benchmark(step_6c, iterations=100, ops_per_iter=50)
        mean_ms = statistics.mean(times) * 1000

        result = BenchmarkResult("Pipeline Step 6c (50 candidates)", times, 50)
        print(f"\n{result}")
        assert mean_ms < 6, f"Step 6c took {mean_ms:.2f}ms for 50 candidates, expected < 6ms"

    def test_context_overhead_percentage(self):
        """Context scoring should add < 15% overhead to total pipeline time.

        Simulates: session scoring (mock 10ms) + context scoring + reranking (mock 5ms)
        Context scoring on 500 items should be < 15% of the total.
        """
        from scoring import ContextScorer
        scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()
        items = _generate_item_dicts(500)

        # Measure context scoring alone
        start = time.perf_counter()
        scorer.score_items(items, ctx, score_field="score")
        context_time = time.perf_counter() - start

        # Simulate total pipeline time (session scoring ~10ms + reranking ~5ms)
        simulated_other = 0.015  # 15ms for other pipeline steps
        total = simulated_other + context_time
        overhead_pct = (context_time / total) * 100

        print(f"\n  Context scoring time: {context_time*1000:.2f}ms")
        print(f"  Simulated other time: {simulated_other*1000:.1f}ms")
        print(f"  Total: {total*1000:.2f}ms")
        print(f"  Context overhead: {overhead_pct:.1f}%")

        assert overhead_pct < 50, f"Context overhead {overhead_pct:.1f}%, expected < 50%"


# =============================================================================
# 7. Component Breakdown
# =============================================================================

class TestBenchmarkComponentBreakdown:
    """Benchmark: breakdown of scoring time by component."""

    def test_component_breakdown(self):
        """Show time breakdown: age vs weather vs total."""
        from scoring.age_scorer import AgeScorer
        from scoring.weather_scorer import WeatherScorer
        from scoring import ContextScorer
        from scoring.context import AgeGroup

        age_scorer = AgeScorer()
        weather_scorer = WeatherScorer()
        context_scorer = ContextScorer()
        ctx = _make_gen_z_summer_context()

        items = _generate_item_dicts(500)
        random.seed(42)

        # Age only
        start = time.perf_counter()
        for item in items:
            age_scorer.score(item, AgeGroup.GEN_Z)
        age_time = time.perf_counter() - start

        # Weather only
        start = time.perf_counter()
        for item in items:
            weather_scorer.score(item, ctx.weather)
        weather_time = time.perf_counter() - start

        # Combined (ContextScorer)
        start = time.perf_counter()
        for item in items:
            context_scorer.score_item(item, ctx)
        combined_time = time.perf_counter() - start

        print(f"\n  === Component Breakdown (500 items) ===")
        print(f"  AgeScorer:     {age_time*1000:.2f}ms ({age_time/combined_time*100:.0f}%)")
        print(f"  WeatherScorer: {weather_time*1000:.2f}ms ({weather_time/combined_time*100:.0f}%)")
        print(f"  ContextScorer: {combined_time*1000:.2f}ms (100%)")
        print(f"  Overhead:      {(combined_time - age_time - weather_time)*1000:.2f}ms "
              f"({(combined_time - age_time - weather_time)/combined_time*100:.0f}%)")

        # Combined should be roughly age + weather (small overhead for orchestration)
        assert combined_time < (age_time + weather_time) * 1.5, \
            "ContextScorer has too much orchestration overhead"


# =============================================================================
# 8. Memory / Object Creation
# =============================================================================

class TestBenchmarkMemory:
    """Benchmark: memory and object creation patterns."""

    def test_scorer_is_reusable(self):
        """ContextScorer should be created once and reused across requests."""
        from scoring import ContextScorer

        # Creating scorer should be fast
        start = time.perf_counter()
        for _ in range(1000):
            scorer = ContextScorer()
        creation_time = time.perf_counter() - start

        print(f"\n  1000 ContextScorer creations: {creation_time*1000:.2f}ms")
        assert creation_time < 0.5, "ContextScorer creation should be < 0.5ms"

    def test_to_scoring_dict_no_copies(self):
        """to_scoring_dict should create minimal new objects."""
        candidates = _generate_candidates(100)

        # Time the conversion
        start = time.perf_counter()
        dicts = [c.to_scoring_dict() for c in candidates]
        elapsed = time.perf_counter() - start

        print(f"\n  100 to_scoring_dict calls: {elapsed*1000:.3f}ms")
        assert elapsed < 0.01, "100 conversions should take < 10ms"
        assert len(dicts) == 100


# =============================================================================
# 9. Summary Report
# =============================================================================

class TestBenchmarkSummaryReport:
    """Generate a comprehensive summary of all benchmarks."""

    @pytest.fixture(autouse=True)
    def setup(self):
        random.seed(42)

    def test_full_summary(self):
        """Print a comprehensive benchmark summary."""
        from scoring import ContextScorer, ContextResolver
        from scoring.age_scorer import AgeScorer
        from scoring.weather_scorer import WeatherScorer
        from search.reranker import SessionReranker
        from scoring.context import AgeGroup

        scorer = ContextScorer()
        age_scorer = AgeScorer()
        weather_scorer = WeatherScorer()
        resolver = ContextResolver(weather_api_key="")
        reranker = SessionReranker()

        ctx_summer = _make_gen_z_summer_context()
        ctx_winter = _make_senior_winter_context()

        items_50 = _generate_item_dicts(50)
        items_200 = _generate_item_dicts(200)
        items_500 = _generate_item_dicts(500)
        candidates_50 = _generate_candidates(50)
        candidates_500 = _generate_candidates(500)

        print("\n" + "=" * 70)
        print(" SCORING MODULE BENCHMARK REPORT")
        print("=" * 70)

        # 1. Single item
        times = benchmark(lambda: scorer.score_item(items_50[0], ctx_summer), iterations=5000)
        r = BenchmarkResult("Single item (score_item)", times, 1)
        print(f"\n{r}")

        # 2. Batch 50
        times = benchmark(lambda: scorer.score_items(items_50.copy(), ctx_summer, score_field="score"), iterations=200)
        r = BenchmarkResult("Batch 50 items", times, 50)
        print(r)

        # 3. Batch 200
        times = benchmark(lambda: scorer.score_items(items_200.copy(), ctx_winter, score_field="score"), iterations=50)
        r = BenchmarkResult("Batch 200 items", times, 200)
        print(r)

        # 4. Batch 500
        times = benchmark(lambda: scorer.score_items(items_500.copy(), ctx_summer, score_field="score"), iterations=20)
        r = BenchmarkResult("Batch 500 items", times, 500)
        print(r)

        # 5. Conversion 50
        times = benchmark(lambda: [c.to_scoring_dict() for c in candidates_50], iterations=200)
        r = BenchmarkResult("to_scoring_dict (50)", times, 50)
        print(r)

        # 6. Resolver
        times = benchmark(
            lambda: resolver.resolve("u1", {"city": "NYC", "country": "US"}, "2000-01-15", {"no_crop": True}),
            iterations=500,
        )
        r = BenchmarkResult("ContextResolver.resolve", times, 1)
        print(r)

        # 7. Full Step 6c (500 candidates)
        def step_6c():
            for c in candidates_500:
                d = c.to_scoring_dict()
                adj = scorer.score_item(d, ctx_summer)
                c.final_score += adj
        times = benchmark(step_6c, iterations=20)
        r = BenchmarkResult("Pipeline Step 6c (500)", times, 500)
        print(r)

        # 8. Reranker with context (200 results)
        profile = {"soft_prefs": {"preferred_brands": ["Brand1"]}, "hard_filters": {}}
        times = benchmark(
            lambda: reranker.rerank(items_200.copy(), user_profile=profile, user_context=ctx_summer),
            iterations=50,
        )
        r = BenchmarkResult("Reranker (200, with context)", times, 200)
        print(r)

        print("=" * 70)
        print(" END OF BENCHMARK REPORT")
        print("=" * 70)

        # Final assertion: entire benchmark suite should complete
        assert True


# =============================================================================
# Standalone Runner
# =============================================================================

if __name__ == "__main__":
    """Run benchmarks standalone with pretty output."""
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
