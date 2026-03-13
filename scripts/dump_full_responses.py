#!/usr/bin/env python3
"""Dump full JSON responses from Llama 4 Scout for inspection."""

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
system_prompt = _SYSTEM_PROMPT + _get_brand_list_addendum()

queries = [
    "date night outfit",
    "something for a wedding",
    "vacation clothes",
    "outfit for brunch",
    "black midi dress for office",
    "floral maxi dress under $50",
    "summer vibes",
    "help me find something nice",
]

for q in queries:
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": system_prompt},
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
    out_tok = usage.completion_tokens if usage else 0
    finish = resp.choices[0].finish_reason

    data = json.loads(raw)
    pretty = json.dumps(data, indent=2)

    print(f"{'='*100}")
    print(f"  QUERY: \"{q}\"")
    print(f"  {ms:.0f}ms | in={in_tok} tok | out={out_tok} tok | finish={finish}")
    print(f"{'='*100}")
    print(pretty)
    print()
