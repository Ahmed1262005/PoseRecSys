"""
Gradio Demo for Amazon Fashion API
Tests all endpoints through the FastAPI backend
Uses local file paths for images (works with public Gradio URL)
"""

import gradio as gr
import requests
import json
import os
from typing import List, Tuple, Optional

API_BASE = "http://localhost:8000"
IMAGES_DIR = "/home/ubuntu/recSys/outfitTransformer/data/amazon_fashion/images"


def get_image_path(item_id: str) -> Optional[str]:
    """Get local image path for an item."""
    path = os.path.join(IMAGES_DIR, f"{item_id}.jpg")
    if os.path.exists(path):
        return path
    return None


def check_api_health():
    """Check if API is healthy."""
    try:
        resp = requests.get(f"{API_BASE}/amazon/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return (
                f"Status: {data['status']}\n"
                f"Items: {data['items_count']:,}\n"
                f"Users: {data['users_count']:,}\n"
                f"SASRec: {'Loaded' if data['sasrec_loaded'] else 'Not loaded'}\n"
                f"Metadata: {'Loaded' if data['metadata_loaded'] else 'Not loaded'}"
            )
        return f"API returned status {resp.status_code}"
    except Exception as e:
        return f"API not available: {e}"


def get_random_user(min_history: int = 5):
    """Get a random user with purchase history."""
    try:
        resp = requests.get(f"{API_BASE}/amazon/random-user?min_history={min_history}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            return data['user_id'], f"History: {data['history_count']} items\nRecent: {', '.join(data['recent_items'][:3])}"
        return "", f"Error: {resp.status_code}"
    except Exception as e:
        return "", f"Error: {e}"


def get_random_items(count: int = 12):
    """Get random items for display."""
    try:
        resp = requests.get(f"{API_BASE}/amazon/random-items?count={count}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            items = data['items']
            gallery_items = []
            for item in items:
                img_path = get_image_path(item['item_id'])
                if img_path:
                    label = f"{item['item_id']}\n{item['title'][:40]}..."
                    gallery_items.append((img_path, label))
            return gallery_items, json.dumps([i['item_id'] for i in items])
        return [], "[]"
    except Exception as e:
        return [], f"Error: {e}"


def get_user_feed(user_id: str, page: int = 1, page_size: int = 12):
    """Get personalized feed for a user."""
    if not user_id:
        return "Please enter a user ID", [], ""

    try:
        resp = requests.post(
            f"{API_BASE}/amazon/feed",
            json={"user_id": user_id, "page": page, "page_size": page_size},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()
            items = data['items']

            gallery_items = []
            for item in items:
                img_path = get_image_path(item['item_id'])
                if img_path:
                    label = f"{item['title'][:30]}...\nScore: {item['score']:.2f} ({item['source']})"
                    gallery_items.append((img_path, label))

            status = (
                f"User: {data['user_id']}\n"
                f"History: {data['history_count']} items\n"
                f"Page {data['pagination']['page']}/{data['pagination']['total_pages']}\n"
                f"Total recommendations: {data['pagination']['total_items']}"
            )
            return status, gallery_items, json.dumps([i['item_id'] for i in items])
        return f"Error: {resp.status_code} - {resp.text}", [], ""
    except Exception as e:
        return f"Error: {e}", [], ""


def get_similar_items(item_id: str, k: int = 10):
    """Get visually similar items."""
    if not item_id:
        return "Please enter an item ID", [], [], ""

    try:
        resp = requests.post(
            f"{API_BASE}/amazon/similar",
            json={"item_id": item_id.strip(), "k": k},
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()

            # Query item
            query_img = get_image_path(item_id.strip())
            query_info = data.get('item_info', {})
            query_label = f"Query: {item_id}\n{query_info.get('title', '')[:40]}"
            query_gallery = [(query_img, query_label)] if query_img else []

            # Similar items
            similar_gallery = []
            for item in data['similar_items']:
                img_path = get_image_path(item['item_id'])
                if img_path:
                    label = f"{item['title'][:25]}...\nSim: {item['score']:.3f}"
                    similar_gallery.append((img_path, label))

            status = f"Found {len(data['similar_items'])} similar items for {item_id}"
            return status, query_gallery, similar_gallery, json.dumps([i['item_id'] for i in data['similar_items']])
        elif resp.status_code == 404:
            return "Item not found", [], [], ""
        return f"Error: {resp.status_code}", [], [], ""
    except Exception as e:
        return f"Error: {e}", [], [], ""


def get_item_details(item_id: str):
    """Get details for a specific item."""
    if not item_id:
        return "Please enter an item ID", None

    try:
        resp = requests.get(f"{API_BASE}/amazon/item/{item_id.strip()}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            img_path = get_image_path(item_id.strip())

            categories = data.get('category', [])
            cat_str = ' > '.join(categories[:4]) if categories else 'N/A'

            details = (
                f"Item ID: {data['item_id']}\n"
                f"Title: {data['title']}\n"
                f"Brand: {data['brand'] or 'N/A'}\n"
                f"Price: {data['price'] or 'N/A'}\n"
                f"Category: {cat_str}\n"
                f"Has Embedding: {data['has_embedding']}"
            )
            return details, img_path
        elif resp.status_code == 404:
            return "Item not found", None
        return f"Error: {resp.status_code}", None
    except Exception as e:
        return f"Error: {e}", None


def style_quiz_recommend(liked_items_str: str, disliked_items_str: str = "", num_recs: int = 12):
    """Get recommendations based on style quiz."""
    if not liked_items_str:
        return "Please enter at least one liked item ID", [], []

    liked = [i.strip() for i in liked_items_str.split(',') if i.strip()]
    disliked = [i.strip() for i in disliked_items_str.split(',') if i.strip()] if disliked_items_str else None

    try:
        payload = {"liked_items": liked, "num_recommendations": num_recs}
        if disliked:
            payload["disliked_items"] = disliked

        resp = requests.post(
            f"{API_BASE}/amazon/style-quiz",
            json=payload,
            timeout=30
        )
        if resp.status_code == 200:
            data = resp.json()

            # Liked items gallery
            liked_gallery = []
            for item_id in liked:
                img_path = get_image_path(item_id)
                if img_path:
                    liked_gallery.append((img_path, f"Liked: {item_id}"))

            # Recommendations
            rec_gallery = []
            for item in data['recommendations']:
                img_path = get_image_path(item['item_id'])
                if img_path:
                    label = f"{item['title'][:25]}...\nMatch: {item['score']:.3f}"
                    rec_gallery.append((img_path, label))

            status = f"Based on {data['based_on_items']} liked items -> {len(data['recommendations'])} recommendations"
            return status, liked_gallery, rec_gallery
        elif resp.status_code == 400:
            return f"Error: {resp.json().get('detail', 'Bad request')}", [], []
        return f"Error: {resp.status_code}", [], []
    except Exception as e:
        return f"Error: {e}", [], []


def like_item(user_id: str, item_id: str):
    """Like an item."""
    if not user_id or not item_id:
        return "Please enter both user ID and item ID"

    try:
        resp = requests.post(
            f"{API_BASE}/amazon/like",
            json={"user_id": user_id.strip(), "item_id": item_id.strip()},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return f"Liked! Total likes: {data['liked_count']}, dislikes: {data['disliked_count']}"
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def dislike_item(user_id: str, item_id: str):
    """Dislike an item."""
    if not user_id or not item_id:
        return "Please enter both user ID and item ID"

    try:
        resp = requests.post(
            f"{API_BASE}/amazon/dislike",
            json={"user_id": user_id.strip(), "item_id": item_id.strip()},
            timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            return f"Disliked! Total likes: {data['liked_count']}, dislikes: {data['disliked_count']}"
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def get_user_profile(user_id: str):
    """Get user profile."""
    if not user_id:
        return "Please enter a user ID"

    try:
        resp = requests.get(f"{API_BASE}/amazon/user/{user_id.strip()}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            recent = ', '.join(data['recent_purchases'][:5]) if data['recent_purchases'] else 'None'
            likes = ', '.join(data['api_likes'][:5]) if data['api_likes'] else 'None'

            return (
                f"User ID: {data['user_id']}\n"
                f"Purchase History: {data['purchase_history_count']} items\n"
                f"Recent Purchases: {recent}\n"
                f"API Likes: {likes}\n"
                f"API Dislikes: {data['api_dislikes_count']}"
            )
        return f"Error: {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"


def get_user_history_images(user_id: str):
    """Get images of user's purchase history."""
    if not user_id:
        return []

    try:
        resp = requests.get(f"{API_BASE}/amazon/user/{user_id.strip()}", timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            gallery = []
            for item_id in data.get('recent_purchases', [])[:10]:
                img_path = get_image_path(item_id)
                if img_path:
                    gallery.append((img_path, item_id))
            return gallery
        return []
    except:
        return []


# Build Gradio Interface
with gr.Blocks(title="Amazon Fashion API Demo") as demo:
    gr.Markdown("""
    # Amazon Men's Fashion API Demo

    **Testing the FastAPI backend** with hybrid recommendations:
    - **FashionCLIP** for visual style similarity
    - **SASRec** for sequential behavior modeling
    - **59,413 items** | **304,422 users** | **2.4M interactions**
    """)

    # Health Check
    with gr.Row():
        health_btn = gr.Button("Check API Health", variant="primary")
        health_output = gr.Textbox(label="API Status", lines=5)
    health_btn.click(check_api_health, outputs=health_output)

    with gr.Tabs():
        # Tab 1: Personalized Feed
        with gr.TabItem("Personalized Feed"):
            gr.Markdown("### Get personalized recommendations for an existing user")

            with gr.Row():
                feed_user_id = gr.Textbox(label="User ID", placeholder="Enter user ID or click Random User")
                feed_page = gr.Slider(1, 10, value=1, step=1, label="Page")
                feed_page_size = gr.Slider(6, 24, value=12, step=6, label="Items per page")

            with gr.Row():
                random_user_btn = gr.Button("Get Random User")
                feed_btn = gr.Button("Get Feed", variant="primary")

            feed_user_info = gr.Textbox(label="User Info", lines=2)

            gr.Markdown("**User's Purchase History**")
            history_gallery = gr.Gallery(label="History", columns=5, height=180)

            feed_status = gr.Textbox(label="Feed Status", lines=4)

            gr.Markdown("**Personalized Recommendations** (FashionCLIP candidates -> SASRec ranked)")
            feed_gallery = gr.Gallery(label="Recommendations", columns=4, height=400)
            feed_item_ids = gr.Textbox(label="Item IDs (for reference)", visible=False)

            def load_random_user():
                user_id, info = get_random_user(10)
                history = get_user_history_images(user_id)
                return user_id, info, history

            random_user_btn.click(load_random_user, outputs=[feed_user_id, feed_user_info, history_gallery])
            feed_btn.click(
                get_user_feed,
                inputs=[feed_user_id, feed_page, feed_page_size],
                outputs=[feed_status, feed_gallery, feed_item_ids]
            )

        # Tab 2: Visual Similarity
        with gr.TabItem("Visual Similarity"):
            gr.Markdown("### Find visually similar items using FashionCLIP")

            with gr.Row():
                similar_item_id = gr.Textbox(label="Item ID", placeholder="Enter item ID (ASIN)")
                similar_k = gr.Slider(5, 20, value=10, step=1, label="Number of similar items")

            with gr.Row():
                random_item_btn = gr.Button("Get Random Item")
                similar_btn = gr.Button("Find Similar", variant="primary")

            similar_status = gr.Textbox(label="Status", lines=1)

            gr.Markdown("**Query Item**")
            query_gallery = gr.Gallery(label="Query", columns=1, height=250)

            gr.Markdown("**Similar Items** (by FashionCLIP cosine similarity)")
            similar_gallery = gr.Gallery(label="Similar Items", columns=5, height=400)
            similar_item_ids = gr.Textbox(visible=False)

            def load_random_item():
                gallery, ids_json = get_random_items(1)
                if gallery:
                    item_id = json.loads(ids_json)[0] if ids_json != "[]" else ""
                    return item_id
                return ""

            random_item_btn.click(load_random_item, outputs=similar_item_id)
            similar_btn.click(
                get_similar_items,
                inputs=[similar_item_id, similar_k],
                outputs=[similar_status, query_gallery, similar_gallery, similar_item_ids]
            )

        # Tab 3: Item Details
        with gr.TabItem("Item Details"):
            gr.Markdown("### View detailed information for a specific item")

            with gr.Row():
                detail_item_id = gr.Textbox(label="Item ID (ASIN)", placeholder="e.g., B000ZOWSMO")
                detail_btn = gr.Button("Get Details", variant="primary")

            with gr.Row():
                detail_output = gr.Textbox(label="Item Details", lines=8)
                detail_image = gr.Image(label="Item Image", height=300)

            detail_btn.click(
                get_item_details,
                inputs=detail_item_id,
                outputs=[detail_output, detail_image]
            )

        # Tab 4: Style Quiz (Cold Start)
        with gr.TabItem("Style Quiz"):
            gr.Markdown("""
            ### Cold Start Style Quiz
            For new users without purchase history. Enter item IDs you like,
            and we'll recommend similar styles using FashionCLIP.
            """)

            # Show random items to choose from
            gr.Markdown("**Browse random items to find ones you like:**")
            browse_gallery = gr.Gallery(label="Random Items (click to see IDs below)", columns=6, height=250)
            browse_ids = gr.Textbox(label="Item IDs (copy from here)", lines=1)
            browse_btn = gr.Button("Load Random Items")
            browse_btn.click(get_random_items, inputs=gr.Number(value=12, visible=False), outputs=[browse_gallery, browse_ids])

            gr.Markdown("---")

            with gr.Row():
                quiz_liked = gr.Textbox(
                    label="Liked Item IDs (comma-separated)",
                    placeholder="e.g., B000ZOWSMO, B00E41UVSM"
                )
                quiz_disliked = gr.Textbox(
                    label="Disliked Item IDs (optional)",
                    placeholder="e.g., B001KZHPCA"
                )

            quiz_num = gr.Slider(5, 24, value=12, step=1, label="Number of recommendations")
            quiz_btn = gr.Button("Get Recommendations", variant="primary")

            quiz_status = gr.Textbox(label="Status", lines=1)

            gr.Markdown("**Your Liked Items**")
            quiz_liked_gallery = gr.Gallery(label="Liked", columns=4, height=200)

            gr.Markdown("**Recommended For You** (based on style similarity)")
            quiz_rec_gallery = gr.Gallery(label="Recommendations", columns=4, height=400)

            quiz_btn.click(
                style_quiz_recommend,
                inputs=[quiz_liked, quiz_disliked, quiz_num],
                outputs=[quiz_status, quiz_liked_gallery, quiz_rec_gallery]
            )

        # Tab 5: User Interactions
        with gr.TabItem("User Interactions"):
            gr.Markdown("### Like/Dislike items and view user profile")

            with gr.Row():
                interact_user_id = gr.Textbox(label="User ID", placeholder="Enter or create a user ID")
                interact_item_id = gr.Textbox(label="Item ID", placeholder="Item to like/dislike")

            with gr.Row():
                like_btn = gr.Button("Like", variant="primary")
                dislike_btn = gr.Button("Dislike", variant="secondary")
                profile_btn = gr.Button("Get Profile")

            interact_status = gr.Textbox(label="Action Result", lines=2)
            profile_output = gr.Textbox(label="User Profile", lines=6)

            like_btn.click(like_item, inputs=[interact_user_id, interact_item_id], outputs=interact_status)
            dislike_btn.click(dislike_item, inputs=[interact_user_id, interact_item_id], outputs=interact_status)
            profile_btn.click(get_user_profile, inputs=interact_user_id, outputs=profile_output)

    gr.Markdown("""
    ---
    **Architecture:**
    1. **FashionCLIP** generates 512-dim visual embeddings for each item
    2. **Faiss** enables fast nearest-neighbor search for candidate generation
    3. **SASRec** re-ranks candidates based on user's sequential purchase history

    **API Endpoints Tested:**
    `/amazon/health`, `/amazon/feed`, `/amazon/similar`, `/amazon/item/{id}`,
    `/amazon/style-quiz`, `/amazon/like`, `/amazon/dislike`, `/amazon/user/{id}`,
    `/amazon/random-items`, `/amazon/random-user`
    """)


if __name__ == "__main__":
    print("Starting Amazon Fashion API Demo...")
    print("Make sure the API is running at http://localhost:8000")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=True,
        allowed_paths=[IMAGES_DIR]
    )
