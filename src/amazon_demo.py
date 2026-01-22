"""
Gradio Demo for Amazon Fashion Hybrid Recommendation System
FashionCLIP (style) + SASRec (behavior) = Personalized Feed
"""

import os
import sys
os.chdir("/home/ubuntu/recSys/outfitTransformer")

# Ensure we use standard RecBole
if '/home/ubuntu/recSys/DuoRec' in sys.path:
    sys.path.remove('/home/ubuntu/recSys/DuoRec')

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
import random

# Import our feed system
from amazon_feed import AmazonFashionFeed

# Global feed instance
feed = None

def load_feed():
    """Initialize the feed system."""
    global feed
    if feed is None:
        print("Loading Amazon Fashion Feed (this may take a minute)...")
        feed = AmazonFashionFeed(
            duorec_checkpoint="models/sasrec_amazon/SASRec-Dec-12-2025_01-35-54.pth"
        )
    return feed


def get_item_image_path(item_id: str) -> str:
    """Get image path for an item."""
    img_path = f"data/amazon_fashion/images/{item_id}.jpg"
    if os.path.exists(img_path):
        return img_path
    return None


def get_user_feed(user_id: str, num_recommendations: int = 20):
    """Get personalized feed for a user."""
    f = load_feed()

    if not user_id or user_id not in f.user_history:
        return "User not found. Try a different user ID.", [], []

    # Get user history
    history = f.user_history[user_id]
    history_images = []
    for item_id in history[-10:]:  # Last 10 items
        img_path = get_item_image_path(item_id)
        if img_path:
            history_images.append((img_path, item_id))

    # Get recommendations
    recommendations = f.get_feed(user_id, limit=num_recommendations)
    rec_images = []
    for rec in recommendations:
        item_id = rec['item_id']
        score = rec['score']
        source = rec['source']
        img_path = get_item_image_path(item_id)
        if img_path:
            rec_images.append((img_path, f"{item_id}\nScore: {score:.2f}\n({source})"))

    status = f"User: {user_id}\nHistory: {len(history)} items\nShowing last {len(history_images)} items → {len(rec_images)} recommendations"

    return status, history_images, rec_images


def get_similar_items(item_id: str, num_similar: int = 10):
    """Get visually similar items."""
    f = load_feed()

    if not item_id or item_id not in f.embeddings_dict:
        return "Item not found.", [], []

    # Get query image
    query_img_path = get_item_image_path(item_id)
    query_images = []
    if query_img_path:
        query_images = [(query_img_path, f"Query: {item_id}")]

    # Get similar items
    similar = f.get_similar_items(item_id, k=num_similar)
    similar_images = []
    for item in similar:
        sim_id = item['item_id']
        sim_score = item['similarity']
        img_path = get_item_image_path(sim_id)
        if img_path:
            similar_images.append((img_path, f"{sim_id}\nSimilarity: {sim_score:.3f}"))

    status = f"Query item: {item_id}\nFound {len(similar_images)} similar items"

    return status, query_images, similar_images


def get_random_user():
    """Get a random user ID with decent history."""
    f = load_feed()
    # Get users with at least 5 items in history and some with images
    good_users = [
        uid for uid, items in f.user_history.items()
        if len(items) >= 5 and any(get_item_image_path(i) for i in items[-5:])
    ]
    if good_users:
        return random.choice(good_users[:1000])  # Sample from first 1000 good users
    return list(f.user_history.keys())[0]


def get_random_item():
    """Get a random item ID that has an image."""
    f = load_feed()
    items_with_images = [
        item_id for item_id in f.item_ids[:5000]
        if get_item_image_path(item_id)
    ]
    if items_with_images:
        return random.choice(items_with_images)
    return f.item_ids[0]


def style_quiz_feed(liked_items_str: str, num_recommendations: int = 20):
    """Generate feed based on style quiz (liked items)."""
    f = load_feed()

    # Parse liked items
    liked_items = [item.strip() for item in liked_items_str.split(',') if item.strip()]

    if not liked_items:
        return "Please enter at least one item ID.", [], []

    # Filter to valid items
    valid_items = [item for item in liked_items if item in f.embeddings_dict]

    if not valid_items:
        return "No valid items found. Please check the item IDs.", [], []

    # Build user vector from liked items
    user_vec = f.style_quiz_init(valid_items)

    if user_vec is None:
        return "Could not build style profile.", [], []

    # Get candidates via CLIP
    candidates = f.get_clip_candidates(user_vec, k=num_recommendations, exclude_items=valid_items)

    # Show liked items
    liked_images = []
    for item_id in valid_items:
        img_path = get_item_image_path(item_id)
        if img_path:
            liked_images.append((img_path, f"Liked: {item_id}"))

    # Show recommendations
    rec_images = []
    for item_id, score in candidates:
        img_path = get_item_image_path(item_id)
        if img_path:
            rec_images.append((img_path, f"{item_id}\nMatch: {score:.3f}"))

    status = f"Style quiz with {len(valid_items)} liked items → {len(rec_images)} recommendations"

    return status, liked_images, rec_images


# Pre-load feed
print("Pre-loading feed...", flush=True)
load_feed()
print("Feed loaded!", flush=True)

# Get some sample IDs for examples
print("Finding sample users...", flush=True)
sample_users = [get_random_user() for _ in range(3)]
print("Finding sample items...", flush=True)
sample_items = [get_random_item() for _ in range(3)]
print("Sample IDs found!", flush=True)

# Build Gradio interface
with gr.Blocks(title="Amazon Fashion Recommender") as demo:
    gr.Markdown("""
    # Amazon Men's Fashion Recommendation Demo

    **Hybrid System**: FashionCLIP (visual style) + SASRec (sequential behavior)

    - **59,413 items** with visual embeddings
    - **304,422 users** with purchase history
    - **SASRec model** trained on 2.4M interactions (NDCG@10: 29.3%)
    """)

    with gr.Tabs():
        # Tab 1: User Feed
        with gr.TabItem("Personalized Feed"):
            gr.Markdown("### Get recommendations for an existing user")
            with gr.Row():
                user_input = gr.Textbox(
                    label="User ID",
                    placeholder="Enter user ID...",
                    value=sample_users[0] if sample_users else ""
                )
                num_recs = gr.Slider(5, 50, value=20, step=5, label="Number of recommendations")
                random_user_btn = gr.Button("Random User")

            feed_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("**User's Recent History** (what they've purchased)")
            history_gallery = gr.Gallery(label="History", columns=5, height=200)

            gr.Markdown("**Personalized Recommendations** (FashionCLIP candidates → SASRec ranked)")
            rec_gallery = gr.Gallery(label="Recommendations", columns=5, height=400)

            get_feed_btn = gr.Button("Get Recommendations", variant="primary")

            random_user_btn.click(get_random_user, outputs=user_input)
            get_feed_btn.click(
                get_user_feed,
                inputs=[user_input, num_recs],
                outputs=[feed_status, history_gallery, rec_gallery]
            )

        # Tab 2: Similar Items
        with gr.TabItem("Visual Similarity"):
            gr.Markdown("### Find visually similar items using FashionCLIP")
            with gr.Row():
                item_input = gr.Textbox(
                    label="Item ID",
                    placeholder="Enter item ID...",
                    value=sample_items[0] if sample_items else ""
                )
                num_similar = gr.Slider(5, 30, value=10, step=5, label="Number of similar items")
                random_item_btn = gr.Button("Random Item")

            similar_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("**Query Item**")
            query_gallery = gr.Gallery(label="Query", columns=1, height=200)

            gr.Markdown("**Similar Items** (by FashionCLIP cosine similarity)")
            similar_gallery = gr.Gallery(label="Similar Items", columns=5, height=400)

            get_similar_btn = gr.Button("Find Similar", variant="primary")

            random_item_btn.click(get_random_item, outputs=item_input)
            get_similar_btn.click(
                get_similar_items,
                inputs=[item_input, num_similar],
                outputs=[similar_status, query_gallery, similar_gallery]
            )

        # Tab 3: Style Quiz
        with gr.TabItem("Style Quiz (Cold Start)"):
            gr.Markdown("""
            ### New user? Build your style profile!
            Enter item IDs you like, and we'll recommend similar styles.
            This simulates a style quiz for new users without purchase history.
            """)

            liked_input = gr.Textbox(
                label="Liked Item IDs (comma-separated)",
                placeholder="e.g., B001WAL3G2, B00BMBZLKU, B00VKVWVVE",
                value=", ".join(sample_items) if sample_items else ""
            )
            quiz_num_recs = gr.Slider(5, 50, value=20, step=5, label="Number of recommendations")

            quiz_status = gr.Textbox(label="Status", interactive=False)

            gr.Markdown("**Your Liked Items**")
            liked_gallery = gr.Gallery(label="Liked", columns=5, height=200)

            gr.Markdown("**Recommended For You** (based on style)")
            quiz_rec_gallery = gr.Gallery(label="Recommendations", columns=5, height=400)

            quiz_btn = gr.Button("Get Style Recommendations", variant="primary")

            quiz_btn.click(
                style_quiz_feed,
                inputs=[liked_input, quiz_num_recs],
                outputs=[quiz_status, liked_gallery, quiz_rec_gallery]
            )

    gr.Markdown("""
    ---
    **Architecture**:
    1. **FashionCLIP** generates 512-dim visual embeddings for each item
    2. **Faiss** enables fast nearest-neighbor search for candidate generation
    3. **SASRec** re-ranks candidates based on user's sequential purchase history

    **Dataset**: Amazon Men's Clothing & Accessories (filtered from Clothing, Shoes & Jewelry)
    """)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Creates public URL
    )
