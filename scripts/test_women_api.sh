#!/bin/bash
# Women's Fashion API Test Script
# Usage: bash test_women_api.sh

API_BASE="http://ecommerce.api.outrove.ai:8080"
USER_ID="test_user_$(date +%s)"

echo "======================================"
echo "Women's Fashion API Test"
echo "Base URL: $API_BASE"
echo "User ID: $USER_ID"
echo "======================================"
echo ""

# 1. Health Check
echo "1. Health Check"
echo "---------------"
curl -s "$API_BASE/api/women/health" | python3 -m json.tool
echo ""

# 2. Get Options
echo "2. Get Options (summary)"
echo "------------------------"
curl -s "$API_BASE/api/women/options" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Categories: {len(d[\"categories\"])}')
for c in d['categories']:
    print(f'  - {c[\"id\"]}: {c[\"label\"]}')
print(f'Attributes: {list(d[\"attributes\"].keys())}')
print(f'Total Items: {d[\"total_items\"]}')
"
echo ""

# 3. Start Session
echo "3. Start Session"
echo "----------------"
RESPONSE=$(curl -s -X POST "$API_BASE/api/women/session/start" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\": \"$USER_ID\", \"selected_categories\": [\"dresses\", \"tops_knitwear\"]}")

echo "$RESPONSE" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Status: {d[\"status\"]}')
print(f'Category: {d[\"test_info\"][\"category_label\"]}')
print(f'Items:')
for item in d['items']:
    print(f'  - {item[\"id\"]}')
"

# Extract first item ID for next step
FIRST_ITEM=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['items'][0]['id'])")
echo ""

# 4. Make 3 Choices
echo "4. Making 3 choices..."
echo "----------------------"
for i in 1 2 3; do
    RESPONSE=$(curl -s -X POST "$API_BASE/api/women/session/choose" \
      -H "Content-Type: application/json" \
      -d "{\"user_id\": \"$USER_ID\", \"winner_id\": \"$FIRST_ITEM\"}")

    STATUS=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('status',''))")
    ROUND=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('round',0))")
    CATEGORY=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('test_info',{}).get('category',''))")

    echo "  Round $ROUND: $CATEGORY ($STATUS)"

    if [ "$STATUS" = "complete" ]; then
        echo "  Session complete!"
        break
    fi

    # Get first item for next round
    FIRST_ITEM=$(echo "$RESPONSE" | python3 -c "import json,sys; d=json.load(sys.stdin); items=d.get('items',[]); print(items[0]['id'] if items else '')")
done
echo ""

# 5. Get Summary
echo "5. Get Session Summary"
echo "----------------------"
curl -s "$API_BASE/api/women/session/$USER_ID/summary" | python3 -c "
import json, sys
d = json.load(sys.stdin)
summary = d.get('summary', {})
print(f'Total Swipes: {summary.get(\"total_swipes\", 0)}')
print(f'Likes: {summary.get(\"likes\", 0)}')
print(f'Dislikes: {summary.get(\"dislikes\", 0)}')
print()
print('Learned Preferences:')
for attr, data in summary.get('attribute_preferences', {}).items():
    preferred = data.get('preferred', [])[:2]
    if preferred:
        prefs = ', '.join([f'{p[0]}({p[1]:.2f})' for p in preferred])
        print(f'  {attr}: {prefs}')
"
echo ""

# 6. Get Feed
echo "6. Get Personalized Feed"
echo "------------------------"
curl -s "$API_BASE/api/women/feed/$USER_ID?items_per_category=3" | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Total Items: {d[\"total_items\"]}')
print()
for cat, info in d['feed'].items():
    print(f'{info[\"label\"]} ({info[\"count\"]} items):')
    for item in info['items'][:2]:
        attrs = item.get('attributes', {})
        print(f'  - {item[\"id\"]}')
        print(f'    pattern: {attrs.get(\"pattern\")}, style: {attrs.get(\"style\")}, color: {attrs.get(\"color_family\")}')
        print(f'    similarity: {item.get(\"similarity\", 0):.3f}')
"
echo ""

# 7. Test Image URL
echo "7. Test Image Serving"
echo "---------------------"
IMAGE_URL="$API_BASE/women-images/dresses/dresses/1.webp"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$IMAGE_URL")
echo "Image URL: $IMAGE_URL"
echo "HTTP Status: $HTTP_CODE"
echo ""

echo "======================================"
echo "Test Complete!"
echo "======================================"
echo ""
echo "API Documentation: $API_BASE/docs"
echo "Frontend Demo: $API_BASE/women"
