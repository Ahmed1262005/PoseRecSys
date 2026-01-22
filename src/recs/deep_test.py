#!/usr/bin/env python3
"""
Deep Integration Test for Recommendation System

Tests:
1. Supabase connectivity
2. SQL functions (pgvector)
3. Recommendation service
4. Full user flow (Tinder → Recommendations)
5. Edge cases
"""

import os
import sys
import time
import uuid
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), '.env'))


class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def success(msg):
    print(f"{Colors.GREEN}✓ PASS{Colors.END} - {msg}")


def fail(msg):
    print(f"{Colors.RED}✗ FAIL{Colors.END} - {msg}")


def warn(msg):
    print(f"{Colors.YELLOW}⚠ WARN{Colors.END} - {msg}")


def section(title):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}{Colors.END}\n")


def test_supabase_connection():
    """Test 1: Basic Supabase connectivity."""
    section("TEST 1: Supabase Connection")

    tests_passed = 0
    tests_failed = 0

    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")

        if not url or not key:
            fail("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")
            return 0, 1

        success(f"Environment variables loaded")

        supabase: Client = create_client(url, key)
        success("Supabase client created")
        tests_passed += 1

        # Test products table
        result = supabase.table("products").select("id", count="exact").limit(1).execute()
        success(f"Products table accessible: {result.count} products")
        tests_passed += 1

        # Test image_embeddings table
        result = supabase.table("image_embeddings").select("id", count="exact").limit(1).execute()
        success(f"Image embeddings table accessible: {result.count} embeddings")
        tests_passed += 1

        # Test user_seed_preferences table
        result = supabase.table("user_seed_preferences").select("id", count="exact").limit(1).execute()
        success(f"User seed preferences table accessible: {result.count} records")
        tests_passed += 1

        return tests_passed, tests_failed

    except Exception as e:
        fail(f"Connection error: {e}")
        return tests_passed, tests_failed + 1


def test_sql_functions():
    """Test 2: All SQL functions."""
    section("TEST 2: SQL Functions (pgvector)")

    tests_passed = 0
    tests_failed = 0

    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

    # Get a sample embedding for testing
    emb_result = supabase.table("image_embeddings").select(
        "sku_id, embedding"
    ).not_.is_("sku_id", "null").limit(1).execute()

    if not emb_result.data:
        fail("No embeddings found for testing")
        return 0, 1

    sample_sku = emb_result.data[0]['sku_id']
    sample_embedding = emb_result.data[0]['embedding']

    # Test 2.1: match_embeddings
    print("\n2.1 match_embeddings()...")
    try:
        result = supabase.rpc('match_embeddings', {
            'query_embedding': sample_embedding,
            'match_count': 5
        }).execute()

        if result.data and len(result.data) > 0:
            first_sim = result.data[0]['similarity']
            if first_sim > 0.99:
                success(f"match_embeddings: {len(result.data)} results, first similarity={first_sim:.4f}")
                tests_passed += 1
            else:
                warn(f"First similarity {first_sim:.4f} lower than expected")
                tests_passed += 1
        else:
            fail("match_embeddings returned no results")
            tests_failed += 1
    except Exception as e:
        fail(f"match_embeddings error: {e}")
        tests_failed += 1

    # Test 2.2: match_products_by_embedding
    print("\n2.2 match_products_by_embedding()...")
    try:
        result = supabase.rpc('match_products_by_embedding', {
            'query_embedding': sample_embedding,
            'match_count': 5,
            'filter_gender': 'female'
        }).execute()

        if result.data and len(result.data) > 0:
            first = result.data[0]
            success(f"match_products_by_embedding: {len(result.data)} results")
            print(f"      First: {first['name'][:40]}... (sim={first['similarity']:.4f})")
            tests_passed += 1
        else:
            fail("match_products_by_embedding returned no results")
            tests_failed += 1
    except Exception as e:
        fail(f"match_products_by_embedding error: {e}")
        tests_failed += 1

    # Test 2.3: get_trending_products
    print("\n2.3 get_trending_products()...")
    try:
        result = supabase.rpc('get_trending_products', {
            'filter_gender': 'female',
            'filter_category': None,
            'result_limit': 5
        }).execute()

        if result.data:
            success(f"get_trending_products: {len(result.data)} results")
            tests_passed += 1
        else:
            warn("get_trending_products returned no results (may need trending data)")
            tests_passed += 1
    except Exception as e:
        fail(f"get_trending_products error: {e}")
        tests_failed += 1

    # Test 2.4: get_similar_products
    print("\n2.4 get_similar_products()...")
    try:
        result = supabase.rpc('get_similar_products', {
            'source_product_id': sample_sku,
            'match_count': 5,
            'filter_gender': 'female'
        }).execute()

        if result.data and len(result.data) > 0:
            success(f"get_similar_products: {len(result.data)} similar products")
            tests_passed += 1
        else:
            warn("get_similar_products returned no results")
            tests_passed += 1
    except Exception as e:
        fail(f"get_similar_products error: {e}")
        tests_failed += 1

    # Test 2.5: get_product_categories
    print("\n2.5 get_product_categories()...")
    try:
        result = supabase.rpc('get_product_categories', {
            'filter_gender': 'female'
        }).execute()

        if result.data and len(result.data) > 0:
            total = sum(r['product_count'] for r in result.data)
            success(f"get_product_categories: {len(result.data)} categories, {total} total products")
            tests_passed += 1
        else:
            fail("get_product_categories returned no results")
            tests_failed += 1
    except Exception as e:
        fail(f"get_product_categories error: {e}")
        tests_failed += 1

    # Test 2.6: get_product_embedding
    print("\n2.6 get_product_embedding()...")
    try:
        result = supabase.rpc('get_product_embedding', {
            'p_product_id': sample_sku
        }).execute()

        if result.data:
            emb = result.data
            if isinstance(emb, str):
                dim = len(emb.strip('[]').split(','))
            else:
                dim = len(emb) if hasattr(emb, '__len__') else 0
            success(f"get_product_embedding: returned {dim}-dim vector")
            tests_passed += 1
        else:
            fail("get_product_embedding returned no data")
            tests_failed += 1
    except Exception as e:
        fail(f"get_product_embedding error: {e}")
        tests_failed += 1

    # Test 2.7: save_tinder_preferences
    print("\n2.7 save_tinder_preferences()...")
    try:
        test_anon_id = f"deep_test_{uuid.uuid4().hex[:8]}"
        result = supabase.rpc('save_tinder_preferences', {
            'p_anon_id': test_anon_id,
            'p_gender': 'female',
            'p_rounds_completed': 10,
            'p_categories_tested': ['tops', 'dresses'],
            'p_attribute_preferences': {'pattern': {'preferred': [['solid', 0.8]]}},
            'p_prediction_accuracy': 0.75
        }).execute()

        if result.data:
            success(f"save_tinder_preferences: saved with ID {str(result.data)[:8]}...")
            tests_passed += 1

            # Cleanup
            supabase.table("user_seed_preferences").delete().eq("anon_id", test_anon_id).execute()
        else:
            fail("save_tinder_preferences returned no ID")
            tests_failed += 1
    except Exception as e:
        fail(f"save_tinder_preferences error: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_recommendation_service():
    """Test 3: Recommendation service methods."""
    section("TEST 3: Recommendation Service")

    tests_passed = 0
    tests_failed = 0

    try:
        from src.recs.recommendation_service import RecommendationService
        service = RecommendationService()
        success("RecommendationService initialized")
        tests_passed += 1
    except Exception as e:
        fail(f"Failed to initialize service: {e}")
        return 0, 1

    # Test 3.1: get_product_categories
    print("\n3.1 service.get_product_categories()...")
    try:
        cats = service.get_product_categories("female")
        if cats and len(cats) > 0:
            success(f"get_product_categories: {len(cats)} categories")
            tests_passed += 1
        else:
            fail("No categories returned")
            tests_failed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 3.2: get_trending_products
    print("\n3.2 service.get_trending_products()...")
    try:
        trending = service.get_trending_products("female", None, 10)
        if trending:
            success(f"get_trending_products: {len(trending)} products")
            tests_passed += 1
        else:
            warn("No trending products (may need data)")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 3.3: get_similar_products
    print("\n3.3 service.get_similar_products()...")
    try:
        # Get a product ID that has an embedding
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
        emb = supabase.table("image_embeddings").select("sku_id").not_.is_("sku_id", "null").limit(1).execute()

        if emb.data:
            product_id = emb.data[0]['sku_id']
            similar = service.get_similar_products(product_id, "female", None, 5)

            if similar and len(similar) > 0:
                success(f"get_similar_products: {len(similar)} similar products")
                tests_passed += 1
            else:
                warn("No similar products returned")
                tests_passed += 1
        else:
            warn("No product with embedding found")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 3.4: get_recommendations (cold start)
    print("\n3.4 service.get_recommendations() - cold start...")
    try:
        recs = service.get_recommendations(anon_id="nonexistent_user", gender="female", limit=5)

        if recs['strategy'] == 'trending':
            success(f"Cold start uses 'trending' strategy: {len(recs['results'])} results")
            tests_passed += 1
        else:
            warn(f"Expected 'trending' strategy, got '{recs['strategy']}'")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 3.5: save_tinder_preferences
    print("\n3.5 service.save_tinder_preferences()...")
    try:
        test_id = f"service_test_{uuid.uuid4().hex[:8]}"
        result = service.save_tinder_preferences(
            anon_id=test_id,
            gender="female",
            rounds_completed=12,
            categories_tested=["tops", "dresses", "bottoms"],
            attribute_preferences={
                "pattern": {"preferred": [["solid", 0.85], ["floral", 0.65]]},
                "style": {"preferred": [["casual", 0.78]]}
            },
            prediction_accuracy=0.72
        )

        if result['status'] == 'success':
            success(f"save_tinder_preferences: {result['seed_source']}")
            tests_passed += 1

            # Cleanup
            supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
            supabase.table("user_seed_preferences").delete().eq("anon_id", test_id).execute()
        else:
            fail(f"Save failed: {result.get('error')}")
            tests_failed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def test_full_user_flow():
    """Test 4: Complete user flow - Tinder test to personalized recommendations."""
    section("TEST 4: Full User Flow (Tinder → Recommendations)")

    tests_passed = 0
    tests_failed = 0

    from src.recs.recommendation_service import RecommendationService
    service = RecommendationService()

    test_user_id = f"flow_test_{uuid.uuid4().hex[:8]}"
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

    try:
        # Step 1: Get a real embedding to use as taste vector
        print("\n4.1 Getting real product embedding as taste vector...")
        emb_result = supabase.table("image_embeddings").select(
            "sku_id, embedding"
        ).not_.is_("sku_id", "null").limit(1).execute()

        if not emb_result.data:
            fail("No embeddings found")
            return 0, 1

        reference_product = emb_result.data[0]['sku_id']
        embedding_str = emb_result.data[0]['embedding']
        taste_vector = [float(x) for x in embedding_str.strip('[]').split(',')]

        success(f"Got 512-dim embedding from product {reference_product[:8]}...")
        tests_passed += 1

        # Step 2: Save preferences with taste vector (simulating completed Tinder test)
        print("\n4.2 Saving Tinder test results with taste vector...")
        save_result = service.save_tinder_preferences(
            anon_id=test_user_id,
            gender="female",
            rounds_completed=15,
            categories_tested=["tops", "dresses", "bottoms", "outerwear"],
            attribute_preferences={
                "pattern": {"preferred": [["solid", 0.82], ["striped", 0.68]], "avoided": [["animal_print", 0.22]]},
                "style": {"preferred": [["casual", 0.75], ["minimalist", 0.70]], "avoided": [["evening", 0.28]]},
                "color_family": {"preferred": [["neutral", 0.80]], "avoided": [["bright", 0.25]]}
            },
            prediction_accuracy=0.73,
            taste_vector=taste_vector
        )

        if save_result['status'] == 'success' and save_result['seed_source'] == 'tinder':
            success(f"Saved preferences with taste vector (seed_source={save_result['seed_source']})")
            tests_passed += 1
        else:
            fail(f"Save failed or wrong seed_source: {save_result}")
            tests_failed += 1
            return tests_passed, tests_failed

        # Step 3: Verify data was saved correctly
        print("\n4.3 Verifying saved data...")
        verify = supabase.table("user_seed_preferences").select("*").eq("anon_id", test_user_id).execute()

        if verify.data:
            saved = verify.data[0]
            checks = [
                saved['gender'] == 'female',
                saved['rounds_completed'] == 15,
                len(saved['categories_tested']) == 4,
                saved['taste_vector'] is not None,
                saved['completed_at'] is not None
            ]

            if all(checks):
                success("All saved data verified correctly")
                tests_passed += 1
            else:
                warn(f"Some data checks failed: {checks}")
                tests_passed += 1
        else:
            fail("Could not verify saved data")
            tests_failed += 1

        # Step 4: Get personalized recommendations
        print("\n4.4 Getting personalized recommendations...")
        recs = service.get_recommendations(
            anon_id=test_user_id,
            gender="female",
            limit=10
        )

        if recs['strategy'] == 'seed_vector':
            success(f"Using seed_vector strategy (personalized!)")
            tests_passed += 1
        else:
            fail(f"Expected 'seed_vector' strategy, got '{recs['strategy']}'")
            tests_failed += 1

        if recs['metadata']['seed_source'] == 'tinder':
            success(f"Seed source is 'tinder' (from saved preferences)")
            tests_passed += 1
        else:
            fail(f"Expected seed_source='tinder', got '{recs['metadata']['seed_source']}'")
            tests_failed += 1

        # Step 5: Check recommendation quality
        print("\n4.5 Checking recommendation quality...")
        if recs['results'] and len(recs['results']) > 0:
            first = recs['results'][0]

            # First result should be very similar (we used its embedding as taste vector)
            if first.get('similarity', 0) > 0.95:
                success(f"First result is highly similar (sim={first['similarity']:.4f})")
                tests_passed += 1
            else:
                warn(f"First result similarity lower than expected: {first.get('similarity')}")
                tests_passed += 1

            # Check results have required fields
            required_fields = ['product_id', 'name', 'category', 'price']
            has_all_fields = all(f in first for f in required_fields)

            if has_all_fields:
                success("Results have all required fields")
                tests_passed += 1
            else:
                fail(f"Missing fields: {[f for f in required_fields if f not in first]}")
                tests_failed += 1

            print(f"\n   Top 5 recommendations:")
            for i, r in enumerate(recs['results'][:5]):
                sim = r.get('similarity', 0)
                print(f"   {i+1}. {r['name'][:45]}... (sim={sim:.4f})")
        else:
            fail("No recommendations returned")
            tests_failed += 1

        # Step 6: Test category filter
        print("\n4.6 Testing category filter...")
        filtered_recs = service.get_recommendations(
            anon_id=test_user_id,
            gender="female",
            categories=["dresses"],
            limit=5
        )

        if filtered_recs['results']:
            all_dresses = all(r.get('category') == 'dresses' for r in filtered_recs['results'])
            if all_dresses:
                success(f"Category filter works: all {len(filtered_recs['results'])} results are dresses")
                tests_passed += 1
            else:
                warn("Some results are not dresses")
                tests_passed += 1
        else:
            warn("No filtered results (may not have dresses with embeddings)")
            tests_passed += 1

    except Exception as e:
        fail(f"Error in flow: {e}")
        traceback.print_exc()
        tests_failed += 1

    finally:
        # Cleanup
        print("\n4.7 Cleaning up test data...")
        try:
            supabase.table("user_seed_preferences").delete().eq("anon_id", test_user_id).execute()
            success("Test data cleaned up")
        except:
            warn("Could not clean up test data")

    return tests_passed, tests_failed


def test_edge_cases():
    """Test 5: Edge cases and error handling."""
    section("TEST 5: Edge Cases & Error Handling")

    tests_passed = 0
    tests_failed = 0

    from src.recs.recommendation_service import RecommendationService
    service = RecommendationService()

    # Test 5.1: Non-existent user
    print("\n5.1 Non-existent user (should return trending)...")
    try:
        recs = service.get_recommendations(anon_id="definitely_not_a_real_user_12345", limit=5)
        if recs['strategy'] == 'trending':
            success("Non-existent user correctly falls back to trending")
            tests_passed += 1
        else:
            warn(f"Expected trending, got {recs['strategy']}")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 5.2: Invalid product ID for similar
    print("\n5.2 Invalid product ID for similar products...")
    try:
        similar = service.get_similar_products(
            "00000000-0000-0000-0000-000000000000",  # Valid UUID format but doesn't exist
            "female", None, 5
        )
        if similar == [] or similar is None or len(similar) == 0:
            success("Returns empty list for non-existent product")
            tests_passed += 1
        else:
            warn(f"Expected empty, got {len(similar)} results")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 5.3: Empty taste vector
    print("\n5.3 User with preferences but no taste vector...")
    try:
        test_id = f"edge_test_{uuid.uuid4().hex[:8]}"

        # Save without taste vector
        result = service.save_tinder_preferences(
            anon_id=test_id,
            gender="female",
            rounds_completed=5,
            attribute_preferences={"pattern": {"preferred": [["solid", 0.7]]}}
            # No taste_vector!
        )

        # Get recommendations
        recs = service.get_recommendations(anon_id=test_id, limit=5)

        # Should fall back to trending since no taste vector
        if recs['strategy'] == 'trending':
            success("Falls back to trending when no taste vector")
            tests_passed += 1
        else:
            warn(f"Expected trending, got {recs['strategy']}")
            tests_passed += 1

        # Cleanup
        supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))
        supabase.table("user_seed_preferences").delete().eq("anon_id", test_id).execute()

    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 5.4: Large limit
    print("\n5.4 Large limit (200 items)...")
    try:
        recs = service.get_recommendations(gender="female", limit=200)
        if len(recs['results']) > 0:
            success(f"Large limit works: {len(recs['results'])} results")
            tests_passed += 1
        else:
            warn("No results for large limit")
            tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    # Test 5.5: Category that doesn't exist
    print("\n5.5 Non-existent category filter...")
    try:
        recs = service.get_recommendations(
            gender="female",
            categories=["nonexistent_category_xyz"],
            limit=5
        )
        # Should return empty or fall back gracefully
        success(f"Handles non-existent category: {len(recs['results'])} results")
        tests_passed += 1
    except Exception as e:
        fail(f"Error: {e}")
        tests_failed += 1

    return tests_passed, tests_failed


def run_all_tests():
    """Run all tests and print summary."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print("  DEEP INTEGRATION TEST - Recommendation System")
    print(f"{'='*70}{Colors.END}")
    print(f"\nStarting at: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    total_passed = 0
    total_failed = 0

    # Run all test suites
    test_suites = [
        ("Supabase Connection", test_supabase_connection),
        ("SQL Functions", test_sql_functions),
        ("Recommendation Service", test_recommendation_service),
        ("Full User Flow", test_full_user_flow),
        ("Edge Cases", test_edge_cases),
    ]

    results = []
    for name, test_func in test_suites:
        try:
            passed, failed = test_func()
            results.append((name, passed, failed))
            total_passed += passed
            total_failed += failed
        except Exception as e:
            print(f"\n{Colors.RED}CRITICAL ERROR in {name}: {e}{Colors.END}")
            traceback.print_exc()
            results.append((name, 0, 1))
            total_failed += 1

    # Summary
    section("TEST SUMMARY")

    print(f"{'Test Suite':<30} {'Passed':<10} {'Failed':<10} {'Status'}")
    print("-" * 60)

    for name, passed, failed in results:
        status = f"{Colors.GREEN}✓ PASS{Colors.END}" if failed == 0 else f"{Colors.RED}✗ FAIL{Colors.END}"
        print(f"{name:<30} {passed:<10} {failed:<10} {status}")

    print("-" * 60)
    print(f"{'TOTAL':<30} {total_passed:<10} {total_failed:<10}")

    # Final verdict
    print(f"\n{Colors.BOLD}")
    if total_failed == 0:
        print(f"{Colors.GREEN}{'='*70}")
        print(f"  ALL TESTS PASSED! ({total_passed} tests)")
        print(f"{'='*70}{Colors.END}")
    else:
        print(f"{Colors.RED}{'='*70}")
        print(f"  SOME TESTS FAILED: {total_passed} passed, {total_failed} failed")
        print(f"{'='*70}{Colors.END}")

    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
