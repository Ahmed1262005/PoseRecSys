"""
Test ~230 real-world user queries through the full hybrid search pipeline.
Covers coverage, occasion, aesthetic, concrete attributes, fit/body, modesty,
color, season, brand, price, fabric, constraints, sizing, stress tests, dialogue.

Usage:
    PYTHONPATH=src python scripts/test_20_queries.py
"""

import json
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from search.hybrid_search import HybridSearchService
from search.models import HybridSearchRequest, SortBy


QUERIES = [
    # --- Coverage / body concerns ---
    "Help me find a top that hides my arms",
    "Help me find a dress that doesn't show my stomach",
    "Help me find something that looks expensive",
    "Help me find an outfit for a first date",
    "Help me find something cute but not too try-hard",
    "Help me find something modest but not frumpy",
    "Help me find something sexy but classy",
    "Help me find something I can wear 3 different ways",
    "Help me find something breathable for hot weather",
    "Help me find something warm but not bulky",
    "Help me find something that won't wrinkle",
    "Help me find something that travels well",
    "Help me find something that won't show sweat",
    "Help me find something that doesn't show underwear lines",
    # --- Occasion ---
    "Outfit for a wedding guest",
    "Dress for a black tie wedding",
    "Semi formal wedding outfit",
    "Business casual outfits for work",
    "Interview outfit for a creative job",
    "Outfit for a work dinner",
    "What to wear to a conference",
    "Vacation outfits for Europe",
    "Beach dinner outfit",
    "Brunch outfit",
    "Night out outfit in winter",
    "Outfit for clubbing but not too revealing",
    "Date night outfit",
    "Casual Friday outfit",
    "Outfit for family gathering",
    "Funeral outfit (simple and respectful)",
    "Eid outfit",
    "Ramadan iftar dinner outfit",
    "Graduation dress",
    "Birthday outfit",
    "Engagement party outfit",
    # --- Aesthetic / vibe ---
    "Quiet luxury outfit",
    "Old money style dress",
    "Clean girl outfit",
    "Model off duty look",
    "French girl outfit",
    "Scandi minimalist outfit",
    "Coastal grandmother outfit",
    "Y2K top",
    "Soft girl dress",
    "Mob wife coat",
    "Balletcore skirt",
    "Office siren outfit",
    "Dark academia outfit",
    "Coquette top with bows",
    "Edgy streetwear jacket",
    "Boho maxi dress like Anthropologie",
    "Like Zara but better quality",
    "Like Aritzia vibe basics",
    "Reformation-style dress but cheaper",
    # --- Concrete attributes: outerwear ---
    "Jacket with zippered pockets",
    "Waterproof rain jacket with hood",
    "Windbreaker with adjustable waist",
    "Puffer jacket that's not too puffy",
    "Wool coat with belt",
    "Trench coat with storm flap",
    "Leather jacket oversized",
    "Bomber jacket cropped",
    "Denim jacket lined",
    "Coat with hidden buttons",
    "Jacket with two-way zipper",
    "Long coat that covers my butt",
    # --- Concrete attributes: tops ---
    "Ribbed knit top with square neckline",
    "Button down that doesn't gape at the chest",
    "Wrap top that stays closed",
    "T-shirt that's thick not see-through",
    "Blouse with covered buttons",
    "Top with longer sleeves",
    "Cropped top but not too cropped",
    "Longline tank",
    "Bodysuit with snap closure",
    "Top that isn't clingy",
    # --- Concrete attributes: bottoms ---
    "High rise wide leg jeans",
    "Mid rise straight jeans no rips",
    "Low rise baggy jeans",
    "Pants with elastic waistband but look tailored",
    "Trousers with pleats and belt loops",
    "Skirt with shorts underneath",
    "Maxi skirt with slit",
    "Pockets that don't flare out",
    "Leggings squat proof",
    "Shorts with 5 inch inseam",
    # --- Concrete attributes: dresses ---
    "Midi dress with sleeves",
    "Maxi dress with open back",
    "Wrap dress but not too low cut",
    "Dress with corset bodice",
    "Slip dress satin",
    "Bodycon dress thick material",
    "Dress with pockets",
    "Dress with adjustable straps",
    "Dress with higher neckline",
    "Dress that covers shoulders",
    # --- Fit / body type ---
    "Jeans for short legs",
    "Pants for tall girls 5'10",
    "Petite blazer",
    "Long torso bodysuit",
    "Wide calf boots",
    "Skirt for big hips small waist",
    "Dresses for apple shape",
    "Outfit to hide belly",
    "Jeans that don't gap at waist",
    "Dress that doesn't cling to thighs",
    "Plus size wedding guest dress",
    "Maternity dress for wedding",
    "Postpartum friendly outfits",
    "Outfits that hide upper arms",
    "Outfits that cover my back",
    # --- Coverage / modesty ---
    "Long sleeves but lightweight",
    "Not see through",
    "No cleavage",
    "High neck top",
    "Full length maxi dress no slit",
    "Midi skirt that's not tight",
    "Loose fit pants modest",
    "Longline blazer for coverage",
    "No backless",
    "Not cropped",
    "Covers shoulders",
    "Hijab-friendly dress",
    "Modest wedding guest dress",
    # --- Color / print ---
    "Chocolate brown dress",
    "Butter yellow top",
    "Cherry red mini dress",
    "Navy blazer",
    "Black dress not boring",
    "White top that isn't see-through",
    "Leopard print skirt",
    "Floral dress not grandma",
    "Striped knit top",
    "Polka dot midi dress",
    "Solid color basics",
    "Neutral capsule wardrobe pieces",
    "Monochrome beige outfit",
    "Colorful summer set",
    # --- Season / weather ---
    "Winter work outfits",
    "Summer dresses breathable",
    "Hot weather pants",
    "Layering tops for cold office",
    "Rainy day outfit",
    "Vacation outfits for humid weather",
    "Coat for 10 degrees",
    "Outfit for windy weather",
    "Transitional spring jacket",
    "Fall capsule wardrobe",
    "Ski trip outfits",
    # --- Styling / outfit building ---
    "Outfit ideas with a black blazer",
    "What goes with wide leg jeans",
    "Top to wear with a satin skirt",
    "Shoes that go with this dress",
    "Layering piece for this top",
    "Find matching set",
    "Find pants to match this blazer",
    "Find a top that matches these pants",
    "Find an outfit for this skirt",
    "Find similar items to this",
    "Complete the look",
    # --- Similarity ---
    "Similar to this jacket but cheaper",
    "Similar to this dress but longer",
    "Similar but with sleeves",
    "Same style but in black",
    "Same but more modest",
    "Same but less bodycon",
    "Dupes for this",
    "Something like this from Zara",
    "Something like Aritzia effortless pants",
    "Reformation-style floral midi",
    "Skims-like bodysuit but thicker",
    "Lululemon dupe leggings",
    # --- Brand ---
    "Aritzia-style basics",
    "Anthropologie boho dress",
    "COS minimalist dress",
    "Zara blazer",
    "H&M satin skirt",
    "Abercrombie jeans curve love",
    "Levi's 501 style jeans",
    "Sézane cardigan style",
    "Sandro-style tweed jacket",
    "Like The Row vibes",
    "Skims dupe",
    "Reformation dupe",
    # --- Price / deals ---
    "Under $50 date night dress",
    "Affordable work pants",
    "On sale coats",
    "Best value basics",
    "Cheap but doesn't look cheap",
    "Designer-looking baggy jeans under $80",
    "Wedding guest dresses under $100",
    "Only show items on sale",
    "Discounted matching sets",
    "Clearance party dresses",
    # --- Fabric / feel ---
    "100% cotton t-shirt",
    "Linen pants not see-through",
    "Wool coat",
    "Cashmere sweater",
    "Silk blouse",
    "Non-itchy sweater",
    "Soft loungewear",
    "Breathable fabrics",
    "Sweat-wicking top",
    "Quick dry shorts",
    "Stretchy but structured pants",
    "No pilling knit",
    "Machine washable blazer",
    "Wrinkle resistant dress",
    # --- Constraints / features ---
    "With pockets",
    "With belt loops",
    "With adjustable straps",
    "With built in shorts",
    "With lining",
    "Double lined",
    "No polyester",
    "No itchy material",
    "No dry clean",
    "No rips / no distressing",
    "No logos",
    "No shoulder pads",
    "No slit",
    "No low back",
    # --- Sizing ---
    "Runs small?",
    "True to size?",
    "I'm between sizes",
    "I'm 5'3 which length should I get",
    "Petite friendly",
    "Tall friendly",
    "Plus size options",
    "What size should I buy",
    "This brand sizing is weird",
    "Something that fits a 34DD",
    # --- Stress test ---
    "black mini going out",
    "top for jeans cute",
    "wedding guest dress not ugly",
    "coat warm cute",
    "jeans no gap waist",
    "not itchy sweater pls",
    "need something last minute",
    "something like this but long sleeve",
    "tight but not too tight",
    "formal but chill",
    "office but hot",
    "zip pockets jacket",
    # --- Dialogue / natural language ---
    "Help me find a top that will be good for a work dinner but still cute",
    "I need a jacket that has zippered pockets and looks minimal and modern",
    "I need a dress that's flattering but I don't want to show my arms",
    "I want pants that look tailored but feel like sweatpants",
    "I'm going to a wedding and want something elegant not too revealing and not super expensive",
    "I want a cardigan that looks expensive and isn't itchy",
    "I need a going-out top that isn't cropped and doesn't show cleavage",
    "Find me a travel outfit that's comfortable and still looks put together",
    "I need a blazer that doesn't look boxy on me",
    "I want an outfit for cold weather that still looks cute for a night out",
]

TOP_N = 5  # Show top N results per query


def run_tests():
    print("Initializing hybrid search service...")
    service = HybridSearchService()
    print("Service ready.\n")

    all_results = []
    total_start = time.time()

    for idx, query in enumerate(QUERIES, 1):
        print(f"{'=' * 100}")
        print(f"[{idx:3d}/{len(QUERIES)}] \"{query}\"")
        print(f"{'=' * 100}")

        request = HybridSearchRequest(
            query=query,
            page=1,
            page_size=20,
            sort_by=SortBy.RELEVANCE,
        )

        t_start = time.time()
        try:
            response = service.search(request)
            elapsed_ms = int((time.time() - t_start) * 1000)

            print(f"\n  Intent:     {response.intent}")
            print(f"  Sort:       {response.sort_by}")
            print(f"  Total:      {response.pagination.total_results or len(response.results)} results")
            print(f"  Timing:     {response.timing}")
            print()

            if response.results:
                print(f"  Top {min(TOP_N, len(response.results))} Results:")
                print(f"  {'#':<4} {'Name':<50} {'Brand':<20} {'Price':>8} {'Pattern':<15} {'Category L2':<20}")
                print(f"  {'-'*4} {'-'*50} {'-'*20} {'-'*8} {'-'*15} {'-'*20}")

                for i, product in enumerate(response.results[:TOP_N], 1):
                    name = (product.name[:47] + "...") if len(product.name) > 50 else product.name
                    brand = (product.brand[:17] + "...") if len(product.brand) > 20 else product.brand
                    pattern = (product.pattern or "-")[:15]
                    cat_l2 = (product.category_l2 or "-")[:20]
                    print(f"  {i:<4} {name:<50} {brand:<20} ${product.price:>7.2f} {pattern:<15} {cat_l2:<20}")

                # Show source distribution
                algolia_count = sum(1 for r in response.results if r.algolia_rank)
                semantic_count = sum(1 for r in response.results if r.semantic_rank)
                both_count = sum(1 for r in response.results if r.algolia_rank and r.semantic_rank)
                print(f"\n  Sources: {algolia_count} Algolia, {semantic_count} Semantic, {both_count} in both")
            else:
                print("  NO RESULTS")

            all_results.append({
                "query": query,
                "intent": response.intent,
                "count": response.pagination.total_results or len(response.results),
                "timing": response.timing,
                "top_results": [
                    {
                        "name": p.name,
                        "brand": p.brand,
                        "price": p.price,
                        "pattern": p.pattern,
                        "category_l2": p.category_l2,
                        "algolia_rank": p.algolia_rank,
                        "semantic_rank": p.semantic_rank,
                    }
                    for p in response.results[:TOP_N]
                ],
                "elapsed_ms": elapsed_ms,
                "success": True,
            })

        except Exception as e:
            elapsed_ms = int((time.time() - t_start) * 1000)
            print(f"\n  ERROR: {e}")
            all_results.append({
                "query": query,
                "elapsed_ms": elapsed_ms,
                "success": False,
                "error": str(e),
            })

        print()

    # Summary
    total_elapsed = int(time.time() - total_start)
    succeeded = sum(1 for r in all_results if r["success"])
    with_results = sum(1 for r in all_results if r.get("count", 0) > 0)

    print(f"\n{'=' * 100}")
    print(f"SUMMARY")
    print(f"{'=' * 100}")
    print(f"  Queries:     {len(QUERIES)}")
    print(f"  Succeeded:   {succeeded}/{len(QUERIES)}")
    print(f"  With results: {with_results}/{len(QUERIES)}")
    print(f"  Total time:  {total_elapsed}s")
    print()

    # Per-query summary table
    print(f"  {'#':<4} {'Query':<50} {'Intent':<10} {'Count':>6} {'Time':>8}")
    print(f"  {'-'*4} {'-'*50} {'-'*10} {'-'*6} {'-'*8}")
    for i, r in enumerate(all_results, 1):
        q = (r["query"][:47] + "...") if len(r["query"]) > 50 else r["query"]
        if r["success"]:
            print(f"  {i:<4} {q:<50} {r['intent']:<10} {r['count']:>6} {r['elapsed_ms']:>7}ms")
        else:
            print(f"  {i:<4} {q:<50} {'FAILED':<10} {'---':>6} {r['elapsed_ms']:>7}ms")


if __name__ == "__main__":
    run_tests()
