"""
Diagnostic: trace pagination exhaustion signals across different result sizes.

For each test case:
1. Initial search (P1)
2. Filter refinement (narrow results)
3. Paginate page 2, 3, ... until server says stop (or safety cap)

Logs EVERY response's: results count, has_more, cursor presence, total_results.
Goal: find where the client gets stuck in an infinite pagination loop.

Usage: PYTHONPATH=src python scripts/diag_pagination.py
"""

import os, sys, time, json, requests
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
import jwt as pyjwt

API = os.getenv("API_URL", "http://localhost:8000")
secret = os.getenv("SUPABASE_JWT_SECRET")
now = int(time.time())
token = pyjwt.encode({
    "sub": "diag", "aud": "authenticated", "role": "authenticated",
    "email": "diag@test.com", "aal": "aal1", "exp": now + 3600,
    "iat": now, "is_anonymous": False,
}, secret, algorithm="HS256")
HEADERS = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

def search(body):
    r = requests.post(f"{API}/api/search/hybrid", json=body, headers=HEADERS, timeout=120)
    r.raise_for_status()
    return r.json()

# ── Test cases: (query, filters, label) ──
# Mix of narrow and broad to see different exhaustion behaviors
CASES = [
    # Very narrow: brand filter on vague → small result set
    ("athletic get together outfits", {"brands": ["Boohoo"]}, "NARROW: vague+brand"),
    # Medium: specific + price cap
    ("summer dress", {"max_price": 25.0}, "MEDIUM: specific+price"),
    # Narrow: specific + brand + color
    ("black sequin mini dress", {"brands": ["Princess Polly"]}, "NARROW: specific+brand"),
    # Broad: exact intent + on_sale (lots of results)
    ("dresses", {"on_sale_only": True}, "BROAD: exact+sale"),
    # Very narrow: specific + combined
    ("linen pants", {"brands": ["Boohoo"], "max_price": 30.0}, "NARROW: specific+brand+price"),
]

MAX_PAGES = 12

def run_case(query, filters, label):
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Query: \"{query}\"  Filters: {filters}")
    print(f"{'='*80}")

    # P1: initial search
    t0 = time.time()
    p1 = search({"query": query, "page_size": 50})
    ms = int((time.time() - t0) * 1000)
    n = len(p1.get("results", []))
    hm = p1["pagination"]["has_more"]
    cur = "yes" if p1.get("cursor") else "NONE"
    total = p1["pagination"].get("total_results", "?")
    intent = p1.get("intent", "?")
    sid = p1.get("search_session_id")
    print(f"  P1 (full pipeline): {n} results | has_more={hm} | cursor={cur} | total={total} | intent={intent} | {ms}ms")

    if not sid:
        print("  !! No session ID — cannot continue")
        return

    # Refine
    t0 = time.time()
    ref = search({"query": query, "page_size": 50, "search_session_id": sid, **filters})
    ms = int((time.time() - t0) * 1000)
    n = len(ref.get("results", []))
    hm = ref["pagination"]["has_more"]
    cur = "yes" if ref.get("cursor") else "NONE"
    total = ref["pagination"].get("total_results", "?")
    mode = "REFINE" if ref.get("timing", {}).get("filter_refine") else "FULL"
    new_sid = ref.get("search_session_id")
    print(f"  Refine ({mode}):    {n} results | has_more={hm} | cursor={cur} | total={total} | {ms}ms")

    # Paginate
    cursor = ref.get("cursor")
    session = new_sid
    page = 1
    total_fetched = n

    while True:
        if not cursor:
            print(f"  --> STOPPED: cursor=None after page {page}")
            break
        page += 1
        if page > MAX_PAGES:
            print(f"  --> SAFETY STOP at page {page}")
            break

        t0 = time.time()
        try:
            body = {"query": query, "page_size": 50, "search_session_id": session, "cursor": cursor, **filters}
            pn = search(body)
        except Exception as e:
            print(f"  Page {page}: ERROR {e}")
            break
        ms = int((time.time() - t0) * 1000)

        n = len(pn.get("results", []))
        hm = pn["pagination"]["has_more"]
        cur = "yes" if pn.get("cursor") else "NONE"
        total = pn["pagination"].get("total_results", "?")
        timing = pn.get("timing", {})
        is_endless = timing.get("endless_semantic", False)
        is_extend = timing.get("extend_search", False)
        rounds = timing.get("rounds", "-")
        path = "endless" if is_endless else ("extend" if is_extend else "?")

        total_fetched += n
        print(f"  Page {page} ({path:>7s}): {n:>3d} results | has_more={str(hm):>5s} | cursor={cur:>4s} | total={total} | rounds={rounds} | {ms}ms")

        cursor = pn.get("cursor")
        if not pn["pagination"]["has_more"]:
            print(f"  --> STOPPED: has_more=False after page {page}")
            break
        if n == 0 and cursor:
            print(f"  --> BUG: 0 results but cursor={cur} and has_more={hm}")
            # Keep going to see if it recovers or loops
            if page > 5:
                print(f"  --> FORCE STOP: likely infinite loop")
                break

    print(f"  TOTAL FETCHED: {total_fetched} across {page} pages")


if __name__ == "__main__":
    print("PAGINATION EXHAUSTION DIAGNOSTIC")
    print(f"Server: {API}")
    print(f"Max pages per case: {MAX_PAGES}")
    for query, filters, label in CASES:
        run_case(query, filters, label)
    print("\nDONE")
