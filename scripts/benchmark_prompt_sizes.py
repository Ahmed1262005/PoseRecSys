#!/usr/bin/env python3
"""
Benchmark Llama 4 Scout at 3 prompt sizes to isolate the follow-up latency.

Variant A: Full production prompt (~11K tokens)
Variant B: FU-only minimal prompt (~2K tokens)
Variant C: Mid-size S1+S2+S3+S8+S9 (~5K tokens)

Usage:
    PYTHONPATH=src python scripts/benchmark_prompt_sizes.py
"""

import json
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from openai import OpenAI
from search.query_planner import _SYSTEM_PROMPT, _get_brand_list_addendum

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1/",
    timeout=30,
)

full_prompt = _SYSTEM_PROMPT + _get_brand_list_addendum()

# ---- VARIANT B: FU-only minimal prompt ----
prompt_fu_only = """You are a fashion search query planner for a women's fashion e-commerce store. Decompose the user's query into a structured search plan.

## OUTPUT FORMAT

Return a JSON object with these fields:
- intent: "exact" | "specific" | "vague"
- algolia_query: string (product-name keywords for text search)
- semantic_query: string (rich visual description for FashionCLIP)
- semantic_queries: string[] (2-4 DIVERSE FashionCLIP queries)
- modes: string[] (mode tags)
- attributes: object (positive filter values)
- avoid: object (negative filter values)
- brand: string | null
- max_price: number | null
- min_price: number | null
- follow_ups: array (REQUIRED — 0-3 follow-up questions. Use [] if query is specific enough. NEVER omit this field.)
- confidence: number (0.0-1.0)

## FOLLOW-UP QUESTIONS

Generate 1-3 follow-up questions that materially narrow results by clarifying the LOOK.
Multiple-choice (2-4 options each), answerable quickly.

Rules:
1) Never ask about price, budget, age, size, or weather.
2) Optimize for LOOK: category, formality, vibe, coverage, silhouette, color, fabric, occasion.
3) Each question must map to concrete filters.
4) 0-3 questions. If query is already specific, return 0.
5) 2-4 mutually exclusive options per question. "No preference" only when harmless.

Each follow-up object:
- "dimension": one of ["category", "setting", "formality", "vibe", "coverage", "silhouette", "color_palette", "fabric_vibe"]
- "question": natural question text
- "options": 2-4 choices with "label" and "filters" (filter patch object)

Example: query = "Brunch outfit" (vague):
```json
"follow_ups": [
  {
    "dimension": "category",
    "question": "What kind of brunch look are you going for?",
    "options": [
      {"label": "Dress", "filters": {"product_types": ["dress"]}},
      {"label": "Top + bottom", "filters": {"product_types": ["top", "bottom"]}},
      {"label": "Matching set", "filters": {"product_types": ["set"]}},
      {"label": "Jumpsuit", "filters": {"product_types": ["jumpsuit"]}}
    ]
  },
  {
    "dimension": "vibe",
    "question": "What look do you want to give?",
    "options": [
      {"label": "Effortless casual", "filters": {"vibe": ["classic"]}},
      {"label": "Polished chic", "filters": {"vibe": ["minimal", "classic"]}},
      {"label": "Feminine & cute", "filters": {"vibe": ["romantic"]}},
      {"label": "Trendy statement", "filters": {"vibe": ["trendy"]}}
    ]
  }
]
```

Example: query = "black satin midi dress with long sleeves" (specific):
```json
"follow_ups": []
```

Return ONLY valid JSON. No markdown, no explanation.
IMPORTANT: The "follow_ups" key is REQUIRED. For vague/occasion queries, generate 1-3 follow-ups. For specific queries, return "follow_ups": []. NEVER omit the key."""

# ---- VARIANT C: Mid-size ----
lines = full_prompt.split('\n')
s1_start = s1_end = s2_start = s2_end = s3_start = s3_end = 0
s8_start = s8_end = s9_start = s9_end = 0
for i, line in enumerate(lines):
    if '## SECTION 1:' in line: s1_start = i
    elif '## SECTION 2:' in line: s1_end = i; s2_start = i
    elif '## SECTION 3:' in line: s2_end = i; s3_start = i
    elif '## SECTION 4:' in line and 'SECTION 4b' not in line: s3_end = i
    elif '## SECTION 8:' in line: s8_start = i
    elif '## SECTION 9:' in line: s8_end = i; s9_start = i
    elif '## SECTION 10:' in line: s9_end = i

prompt_mid = '\n'.join(lines[0:2]) + '\n'
prompt_mid += '\n'.join(lines[s1_start:s1_end]) + '\n'
prompt_mid += '\n'.join(lines[s2_start:s2_end]) + '\n'
prompt_mid += '\n'.join(lines[s3_start:s3_end]) + '\n'
prompt_mid += '\n'.join(lines[s8_start:s8_end]) + '\n'
prompt_mid += '\n'.join(lines[s9_start:s9_end]) + '\n'
prompt_mid += '\nReturn ONLY valid JSON. No markdown, no explanation, no code blocks.'
prompt_mid += '\nIMPORTANT: The "follow_ups" key is REQUIRED in your JSON output. For vague/occasion queries, generate 1-3 follow-ups. For specific queries, return "follow_ups": []. NEVER omit the key.'

# ---- VARIANTS ----
variants = [
    ("A: Full (11K)", full_prompt),
    ("B: FU-only (~2K)", prompt_fu_only),
    ("C: Mid S1+S2+S3+S8+S9", prompt_mid),
]

queries = [
    "date night outfit",
    "something for a wedding",
    "vacation clothes",
    "outfit for brunch",
    "help me find something nice",
    "summer vibes",
    "black midi dress for office",
    "floral maxi dress under $50",
]

MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"

print(f"Model: {MODEL}")
print(f"Queries: {len(queries)}")
print()

# Detailed per-query view
for name, prompt in variants:
    est_tok = len(prompt) // 4
    print(f"{'='*90}")
    print(f"  {name} (~{est_tok:,} tokens, {len(prompt):,} chars)")
    print(f"{'='*90}")
    print(f"  {'ms':>7} {'in_tok':>7} {'out_tok':>8} {'intent':>10} {'FU#':>4} {'FU key':>7}  query")
    print(f"  {'-'*85}")

    latencies = []
    fu_keys = 0
    fu_has = 0
    out_toks = []

    for q in queries:
        t0 = time.perf_counter()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": q},
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
            max_tokens=1600,
        )
        ms = (time.perf_counter() - t0) * 1000
        raw = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        otok = usage.completion_tokens if usage else 0
        latencies.append(ms)
        out_toks.append(otok)

        try:
            data = json.loads(raw)
            intent = data.get("intent", "?")
            key_exists = "follow_ups" in data
            if key_exists:
                fu_keys += 1
            fu = data.get("follow_ups") or []
            fu_count = len(fu)
            if fu_count > 0:
                fu_has += 1
        except:
            intent = "ERR"
            fu_count = 0
            key_exists = False

        print(f"  {ms:>6.0f}ms {in_tok:>7} {otok:>8} {intent:>10} {fu_count:>4} {str(key_exists):>7}  {q}")

    avg_ms = sum(latencies) / len(latencies)
    min_ms = min(latencies)
    max_ms = max(latencies)
    avg_out = sum(out_toks) / len(out_toks)
    n = len(queries)
    print(f"  {'-'*85}")
    print(f"  avg={avg_ms:.0f}ms  min={min_ms:.0f}ms  max={max_ms:.0f}ms  |  FU key: {fu_keys}/{n}  has FU: {fu_has}/{n}  |  avg_out: {avg_out:.0f}tok")
    print()

# ---- Summary ----
print(f"\n{'='*90}")
print(f"  SUMMARY")
print(f"{'='*90}")
print(f"  {'Variant':<32} {'~tokens':>8} | {'avg ms':>7} {'min':>7} {'max':>7} | {'FU key':>7} {'has FU':>7} | {'avg out':>8}")
print(f"  {'-'*32} {'-'*8} | {'-'*7} {'-'*7} {'-'*7} | {'-'*7} {'-'*7} | {'-'*8}")

# Re-run is too expensive, so we just save results from above.
# (Printed in per-variant blocks already)
