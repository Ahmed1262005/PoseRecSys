#!/usr/bin/env python
"""Generate an HTML report showing outfit-avoids analysis for 10 edge-case products.

For each product:
  1. Calls OutfitEngine.build_outfit() to get current recommendations
  2. Rebuilds AestheticProfiles for source + every result
  3. Runs the avoids system (hard-filter + soft penalties) on each pair
  4. Shows before / after ranking comparison in a single HTML page

Usage:
    PYTHONPATH=src .venv/bin/python scripts/avoids_edge_case_report.py
"""

import logging
import os
import sys
import time
from datetime import datetime
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("avoids_report")
logger.setLevel(logging.INFO)

from config.database import get_supabase_client
from services.outfit_engine import (
    AestheticProfile,
    OutfitEngine,
    compute_compatibility_score,
)
from services.outfit_avoids import (
    PENALTY_CAP,
    compute_avoid_penalties,
    filter_hard_avoids,
)

# ============================================================================
# Edge-case product list
# ============================================================================

EDGE_CASES = [
    {
        "pid": "bc3efaa3-dd5d-4bb5-aba7-5ad069425fa1",
        "label": "A) Hoodie",
        "tests": "I1 delicate/dressy, F2 satin+sweats, B1 formality gap",
        "desc": "Casual hoodie should NOT get silk/satin skirts, sequin pieces, or very dressy blouses",
    },
    {
        "pid": "44ea618a-9230-4d8a-9e2a-b11d689c5827",
        "label": "B) Formal Gown",
        "tests": "I3 formal+casual-outer, G3 formal+denim-jacket, HF1",
        "desc": "Formal gown should NOT get denim jacket, hoodie, or bomber as outerwear",
    },
    {
        "pid": "a1f91793-26ec-4c00-9098-3e4252212ee1",
        "label": "C) Blazer",
        "tests": "I2 blazer+gym, C2 sporty-L2+formal-L2, HF3",
        "desc": "Tailored blazer should NOT get track pants, windbreakers, or gym hoodies",
    },
    {
        "pid": "10f1a29b-c375-4ddf-ab90-05d3ff364c30",
        "label": "D) Leggings",
        "tests": "I4 legging+formal, HF4 activewear+formal",
        "desc": "Casual leggings should NOT get tux shirts or very formal evening tops",
    },
    {
        "pid": "9fff9b97-35af-4a63-8d25-9a30c40c12f1",
        "label": "E) Black Jeans",
        "tests": "D1 denim-on-denim (material matchy)",
        "desc": "Denim jeans should NOT get denim jacket as default complement (too matchy)",
    },
    {
        "pid": "0a88342d-d16f-43b7-a5a1-6c579978a64a",
        "label": "F) Joggers",
        "tests": "C2 sporty+formal, F1 party+casual, HF2/HF3",
        "desc": "Casual joggers should NOT get sequin tops, tailored blazers, or formal pieces",
    },
    {
        "pid": "565a2dde-ef69-49aa-a070-ed7cde81682f",
        "label": "G) Corset Top",
        "tests": "F3 delicate+distressed, B1 formality, coverage",
        "desc": "Smart-casual corset should NOT get distressed/destroyed bottoms or very casual sweats",
    },
    {
        "pid": "5b2fe608-b731-4660-903e-61d0a84561f6",
        "label": "H) Silk Tank Top",
        "tests": "A2 hot+cold-outer, F2 silk+sweats, G1 occasion",
        "desc": "Delicate silk tank should NOT get heavy puffer/fleece outerwear or fleece joggers",
    },
    {
        "pid": "4249a1d4-32b0-482f-88e6-3ca2f65a5990",
        "label": "I) Cable Knit Sweater",
        "tests": "A1 light-over-heavy, E1 oversized+oversized, D1 knit matchy",
        "desc": "Heavy cable knit should NOT get light sheer cardigan or another oversized chunky knit",
    },
    {
        "pid": "4a46ad5b-bac6-447a-ad91-7b55e1f688c8",
        "label": "J) Sweatshirt",
        "tests": "I1 hoodie+delicate, F2 satin+sweats, F1 party+casual",
        "desc": "Casual sweatshirt should NOT get silk/satin skirts, sequin pieces, or evening wear",
    },
]

# ============================================================================
# Supabase helpers
# ============================================================================

_sb = None


def get_sb():
    global _sb
    if _sb is None:
        _sb = get_supabase_client()
    return _sb


def fetch_profile_for_pid(pid: str) -> AestheticProfile:
    """Fetch product + attributes from Supabase, return AestheticProfile."""
    sb = get_sb()
    pr = sb.from_("products").select("*").eq("id", pid).maybe_single().execute()
    if not pr.data:
        raise ValueError(f"Product {pid} not found")
    ar = (
        sb.from_("product_attributes")
        .select("*")
        .eq("sku_id", pid)
        .maybe_single()
        .execute()
    )
    attrs = ar.data or {}
    profile = AestheticProfile.from_product_and_attrs(pr.data, attrs)
    return profile


# ============================================================================
# Main analysis
# ============================================================================


def analyse_case(engine: OutfitEngine, case: dict) -> dict:
    """Run build_outfit + avoids analysis for one edge-case product."""
    pid = case["pid"]
    logger.info("Analysing %s -- %s", case["label"], pid)
    t0 = time.time()

    # 1. Run build_outfit (anonymous, no user_id)
    try:
        result = engine.build_outfit(pid)
    except Exception as e:
        logger.error("build_outfit failed for %s: %s", pid, e)
        return {"case": case, "error": str(e)}

    elapsed = time.time() - t0
    logger.info("  build_outfit took %.1fs", elapsed)

    # 2. Rebuild source profile
    source_profile = fetch_profile_for_pid(pid)

    # 3. Walk result categories, rebuild candidate profiles, run avoids
    #    Response: recommendations={cat: {items:[...], pagination:{...}}}
    categories_analysis = {}
    recs = result.get("recommendations", {})
    for cat_name, cat_data in recs.items():
        items = cat_data.get("items", []) if isinstance(cat_data, dict) else []
        cat_items = []
        for item in items:
            cpid = item.get("product_id") if isinstance(item, dict) else None
            if not cpid:
                continue
            try:
                cand_profile = fetch_profile_for_pid(cpid)
            except Exception:
                cand_profile = None

            # Run avoids
            avoid_penalty = 0.0
            avoid_rules = []
            hard_filtered = False
            if cand_profile:
                surviving = filter_hard_avoids(source_profile, [cand_profile])
                hard_filtered = len(surviving) == 0
                avoid_penalty, avoid_rules = compute_avoid_penalties(
                    source_profile,
                    cand_profile,
                )

            cat_items.append(
                {
                    "item": item,
                    "profile": cand_profile,
                    "avoid_penalty": avoid_penalty,
                    "avoid_rules": avoid_rules,
                    "hard_filtered": hard_filtered,
                    "new_tattoo": item.get("tattoo_score", 0) + avoid_penalty,
                }
            )

        # Compute new ranks based on new_tattoo
        ranked = sorted(
            enumerate(cat_items), key=lambda x: x[1]["new_tattoo"], reverse=True
        )
        for new_rank, (orig_idx, ci) in enumerate(ranked, 1):
            ci["new_rank"] = new_rank

        categories_analysis[cat_name] = cat_items

    return {
        "case": case,
        "source": result.get("source_product", {}),
        "source_profile": source_profile,
        "scoring_info": result.get("scoring_info", {}),
        "categories": categories_analysis,
        "elapsed": elapsed,
    }


# ============================================================================
# HTML generation
# ============================================================================

RULE_DESCRIPTIONS = {
    "A1": "Light layer over heavy base",
    "A2": "Cold outerwear + hot anchor",
    "A3": "Hot layer + cold anchor",
    "B1": "Formality distance &ge;3",
    "C1": "Technical + tailored material",
    "C2": "Sporty L2 + formal L2",
    "D1": "Same statement material",
    "E1": "Both oversized (non-outerwear)",
    "E2": "Cropped outer over longline heavy",
    "E3": "Bulky top + wide-leg bottom",
    "F1": "Party fabric + casual piece",
    "F2": "Satin/silk + fleece/sweats",
    "F3": "Distressed + lingerie-coded",
    "G1": "Beach occasion + heavy knit",
    "G2": "Office + clubwear",
    "G3": "Formal + casual outerwear",
    "I1": "Hoodie/sweatshirt + delicate",
    "I2": "Blazer/tailored + gym",
    "I3": "Formal dress + casual outer",
    "I4": "Perf leggings + formal top",
    "J1": "Competing statement prints",
    "J2": "Neon + neon",
    "HF1": "HARD: Formal &harr; sporty casual",
    "HF2": "HARD: Sporty &harr; party fabric",
    "HF3": "HARD: Tailored &harr; gym",
    "HF4": "HARD: Activewear &harr; formal",
}


def _attr_chip(label, value):
    if not value:
        return ""
    return (
        f'<span class="attr-chip">{escape(str(label))}: '
        f"<b>{escape(str(value))}</b></span>"
    )


def _rule_chip(rule_id):
    desc = RULE_DESCRIPTIONS.get(rule_id, rule_id)
    cls = "rule-chip-hard" if rule_id.startswith("HF") else "rule-chip"
    return f'<span class="{cls}" title="{escape(desc)}">{escape(rule_id)}</span>'


def _penalty_color(penalty):
    if penalty <= -0.10:
        return "#d32f2f"
    if penalty <= -0.05:
        return "#e65100"
    if penalty < 0:
        return "#f9a825"
    return "#2e7d32"


def _img_tag(url, size=160):
    if not url:
        return (
            f'<div style="width:{size}px;height:{int(size*1.3)}px;background:#f0f0f0;'
            f'display:flex;align-items:center;justify-content:center;border-radius:8px;'
            f'color:#999;font-size:11px;">No img</div>'
        )
    return (
        f'<img src="{escape(url)}" style="width:{size}px;height:{int(size*1.3)}px;'
        f'object-fit:cover;object-position:top;border-radius:8px;" loading="lazy" '
        f'onerror="this.style.display=\'none\'">'
    )


def generate_html(results):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    total_items = sum(
        len(items)
        for r in results
        if "categories" in r
        for items in r["categories"].values()
    )
    total_penalized = sum(
        1
        for r in results
        if "categories" in r
        for items in r["categories"].values()
        for ci in items
        if ci["avoid_penalty"] < 0
    )
    total_hard = sum(
        1
        for r in results
        if "categories" in r
        for items in r["categories"].values()
        for ci in items
        if ci["hard_filtered"]
    )
    total_clean = total_items - total_penalized - total_hard

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Outfit Avoids Edge Case Report</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  background:#fafafa;color:#222;padding:32px;max-width:1460px;margin:auto}}
h1{{font-size:28px;margin-bottom:4px}}
.sub{{color:#666;font-size:14px;margin-bottom:32px}}
.sbar{{display:flex;gap:24px;margin-bottom:36px;padding:16px 20px;
  background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}}
.sstat{{text-align:center}}.sstat .v{{font-size:28px;font-weight:700}}.sstat .l{{font-size:12px;color:#888}}
.cs{{background:#fff;border-radius:12px;margin-bottom:32px;
  box-shadow:0 1px 4px rgba(0,0,0,.08);overflow:hidden}}
.ch{{padding:20px 24px;border-bottom:1px solid #eee}}
.ch h2{{font-size:18px}}.ch .d{{color:#666;font-size:13px;margin-top:4px}}
.ch .t{{color:#999;font-size:12px;font-family:monospace;margin-top:2px}}
.sb{{display:flex;align-items:flex-start;gap:16px;padding:16px 24px;
  background:#f8f8ff;border-bottom:1px solid #eee}}
.sa{{display:flex;flex-wrap:wrap;gap:6px;margin-top:6px}}
.attr-chip{{background:#e8eaf6;color:#333;font-size:11px;padding:2px 8px;
  border-radius:10px;white-space:nowrap}}
.catsec{{padding:16px 24px}}
.catsec h3{{font-size:14px;text-transform:uppercase;letter-spacing:.5px;color:#666;margin-bottom:12px}}
table{{width:100%;border-collapse:collapse;font-size:13px}}
th{{text-align:left;padding:6px 8px;border-bottom:2px solid #ddd;
  font-size:11px;text-transform:uppercase;letter-spacing:.3px;color:#888}}
td{{padding:8px;border-bottom:1px solid #f0f0f0;vertical-align:top}}
tr.hf{{background:#fce4ec}}tr.pen{{background:#fff8e1}}tr.ok{{background:#fff}}
.rc{{font-weight:700;font-size:14px;width:40px;text-align:center}}
.sc{{font-family:monospace;font-size:12px}}
.pb{{display:inline-block;padding:2px 8px;border-radius:10px;color:#fff;font-size:11px;font-weight:600}}
.rule-chip{{background:#ffecb3;color:#795548;font-size:10px;padding:1px 6px;
  border-radius:6px;margin:1px;display:inline-block}}
.rule-chip-hard{{background:#ef9a9a;color:#b71c1c;font-size:10px;padding:1px 6px;
  border-radius:6px;margin:1px;display:inline-block;font-weight:700}}
.ru{{color:#2e7d32;font-size:11px}}.rd{{color:#d32f2f;font-size:11px}}.rs{{color:#999;font-size:11px}}
.in{{font-weight:500;font-size:12px;max-width:180px}}.ib{{color:#888;font-size:11px}}
.err{{padding:24px;color:#d32f2f}}
.leg{{padding:12px 24px;background:#f5f5f5;border-top:1px solid #eee;font-size:12px;color:#666}}
.leg span{{margin-right:16px}}
</style></head><body>
<h1>Outfit Avoids &mdash; Edge Case Report</h1>
<p class="sub">Generated {ts} &middot; 10 edge-case products &middot; avoids v1.0</p>
<div class="sbar">
  <div class="sstat"><div class="v">{total_items}</div><div class="l">Total Recs</div></div>
  <div class="sstat"><div class="v" style="color:#d32f2f">{total_hard}</div><div class="l">Hard-Filtered</div></div>
  <div class="sstat"><div class="v" style="color:#e65100">{total_penalized}</div><div class="l">Soft-Penalized</div></div>
  <div class="sstat"><div class="v" style="color:#2e7d32">{total_clean}</div><div class="l">Clean</div></div>
</div>""")

    for res in results:
        case = res["case"]
        parts.append(f"""<div class="cs"><div class="ch">
  <h2>{escape(case["label"])}</h2>
  <div class="d">{escape(case["desc"])}</div>
  <div class="t">Tests: {escape(case["tests"])}</div>
</div>""")

        if "error" in res:
            parts.append(f'<div class="err">Error: {escape(res["error"])}</div></div>')
            continue

        src = res["source"]
        sp = res.get("source_profile")
        parts.append(f"""<div class="sb">{_img_tag(src.get("image_url",""),100)}<div>
  <div style="font-weight:600;font-size:14px">{escape(src.get("name",""))}</div>
  <div style="color:#666;font-size:12px">{escape(src.get("brand",""))} &middot; ${src.get("price",0):.0f}</div>
  <div class="sa">""")
        if sp:
            for lbl, val in [
                ("L2", sp.gemini_category_l2),
                ("Formality", f"{sp.formality} ({sp.formality_level})" if sp.formality else None),
                ("Fabric", sp.material_family),
                ("Weight", sp.fabric_weight),
                ("Layer", sp.layer_role),
                ("Temp", sp.temp_band),
                ("Style", ", ".join(sp.style_tags[:3]) if sp.style_tags else None),
                ("Pattern", sp.pattern),
                ("Silhouette", sp.silhouette),
                ("Shine", sp.shine_level if sp.shine_level and sp.shine_level != "matte" else None),
            ]:
                parts.append(_attr_chip(lbl, val))
        parts.append("</div></div></div>")

        for cat_name, cat_items in res.get("categories", {}).items():
            parts.append(f"""<div class="catsec"><h3>{escape(cat_name)} ({len(cat_items)} items)</h3>
<table><tr><th>#</th><th></th><th>Product</th><th>L2 / Attrs</th>
<th>TATTOO</th><th>Compat</th><th>Cosine</th>
<th>Avoid&nbsp;Adj</th><th>Rules</th><th>New&nbsp;TATTOO</th><th>&Delta;</th></tr>""")

            for ci in cat_items:
                item = ci["item"]
                cp = ci.get("profile")
                pen = ci["avoid_penalty"]
                rules = ci["avoid_rules"]
                hf = ci["hard_filtered"]
                old_r = item.get("rank", 0)
                new_r = ci.get("new_rank", old_r)
                new_t = ci["new_tattoo"]
                rc = "hf" if hf else ("pen" if pen < 0 else "ok")
                pc = _penalty_color(pen)

                if hf:
                    delta = '<span class="rd">FILTERED</span>'
                elif new_r < old_r:
                    delta = f'<span class="ru">&uarr;{old_r - new_r}</span>'
                elif new_r > old_r:
                    delta = f'<span class="rd">&darr;{new_r - old_r}</span>'
                else:
                    delta = '<span class="rs">&mdash;</span>'

                ahtml = ""
                if cp:
                    bits = [
                        _attr_chip("L2", cp.gemini_category_l2),
                        _attr_chip("Form", cp.formality_level),
                        _attr_chip("Fab", cp.material_family),
                    ]
                    if cp.shine_level and cp.shine_level != "matte":
                        bits.append(_attr_chip("Shine", cp.shine_level))
                    if cp.silhouette:
                        bits.append(_attr_chip("Sil", cp.silhouette))
                    ahtml = " ".join(b for b in bits if b)

                rhtml = " ".join(_rule_chip(r) for r in rules)
                if hf:
                    rhtml = _rule_chip("HF") + " " + rhtml

                parts.append(f"""<tr class="{rc}">
  <td class="rc">{old_r}</td>
  <td>{_img_tag(item.get("image_url",""),56)}</td>
  <td><div class="in">{escape(item.get("name","")[:48])}</div>
      <div class="ib">{escape(item.get("brand",""))} &middot; ${item.get("price",0):.0f}</div></td>
  <td>{ahtml}</td>
  <td class="sc">{item.get("tattoo_score",0):.3f}</td>
  <td class="sc">{item.get("compatibility_score",0):.3f}</td>
  <td class="sc">{item.get("cosine_similarity",0):.3f}</td>
  <td><span class="pb" style="background:{pc}">{pen:+.3f}</span></td>
  <td>{rhtml}</td>
  <td class="sc" style="font-weight:700">{new_t:.3f}</td>
  <td>{delta}</td></tr>""")

            parts.append("</table></div>")

        parts.append("""<div class="leg">
<span style="background:#fce4ec;padding:2px 6px;border-radius:4px">Hard-filtered</span>
<span style="background:#fff8e1;padding:2px 6px;border-radius:4px">Soft-penalized</span>
<span style="background:#fff;padding:2px 6px;border-radius:4px;border:1px solid #eee">Clean</span>
</div></div>""")

    parts.append("</body></html>")
    return "\n".join(parts)


# ============================================================================
# Main
# ============================================================================


def main():
    logger.info("Initializing OutfitEngine ...")
    t0 = time.time()
    engine = OutfitEngine(get_sb())
    logger.info("Engine ready in %.1fs", time.time() - t0)

    results = []
    for i, case in enumerate(EDGE_CASES):
        logger.info("[%d/%d] %s", i + 1, len(EDGE_CASES), case["label"])
        res = analyse_case(engine, case)
        results.append(res)
        n_cats = len(res.get("categories", {}))
        logger.info("  done -- %d categories", n_cats)

    html = generate_html(results)
    out_path = ROOT / "scripts" / "avoids_report.html"
    out_path.write_text(html, encoding="utf-8")
    logger.info("Report written to %s (%d bytes)", out_path, len(html))
    print(f"\nReport saved: {out_path}")


if __name__ == "__main__":
    main()
