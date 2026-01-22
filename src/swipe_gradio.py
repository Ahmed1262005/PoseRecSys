"""
Gradio UI for Swipe-based Style Learning (v5 - Semantic)

Key improvements:
- RESPECTS consecutive dislikes - moves away fast
- REJECTS visual clusters with strong negative signal
- Uses anti-taste vector to avoid similar items
- LEARNS within categories using taxonomy (archetypes, visual anchors)
- Provides semantic insights into style preferences
"""

import gradio as gr
from PIL import Image
from pathlib import Path
from swipe_engine import SwipeEngine, UserPreferences, SwipeAction
import json

# Initialize engine
engine = SwipeEngine()

# Global user state
current_user: UserPreferences = None
current_item_id: str = None


def start_session(colors_to_avoid: list) -> tuple:
    """Start a new swipe session with color filters."""
    global current_user, current_item_id

    avoid_set = set()
    if colors_to_avoid:
        for color in colors_to_avoid:
            avoid_set.add(color.lower().strip())

    current_user = UserPreferences(
        user_id="gradio_user",
        colors_to_avoid=avoid_set
    )

    result = engine.get_next_item(current_user)
    current_item_id, session_complete = result

    if current_item_id and not session_complete:
        return get_display_data()
    else:
        return None, "No items available", "{}", gr.update(visible=True)


def get_display_data(session_complete: bool = False) -> tuple:
    """Get current item display data."""
    global current_item_id

    if session_complete or not current_item_id:
        summary = engine.get_preference_summary(current_user)

        # Format category preferences
        cat_text = ""
        for v in summary['attribute_preferences'].get('category', []):
            cat_text += f"- **{v['value']}**: {v['score']:.2f} ({v['likes']}/{v['total']})\n"

        # Format top brands
        brand_text = ""
        for brand, data in list(summary['brand_preferences'].items())[:3]:
            brand_text += f"- **{brand}**: {data['likes']}/{data['total']} liked ({data['score']:.2f})\n"

        # Format color preferences
        color_text = ""
        for v in summary['attribute_preferences'].get('color', [])[:3]:
            color_text += f"- **{v['value']}**: {v['score']:.2f} ({v['likes']}/{v['total']})\n"

        # Format style profile (archetypes)
        style = summary.get('style_profile', {})
        archetype_text = ""
        for arch in style.get('archetypes', [])[:2]:  # Top 2
            direction = "+" if arch['preference'] > 0 else ""
            archetype_text += f"- **{arch['archetype']}**: {direction}{arch['preference']:.3f}\n"

        # Format visual anchors
        anchor_text = ""
        for anchor in style.get('visual_anchors', [])[:3]:  # Top 3
            archetype_text += f"- **{anchor['anchor']}**: {anchor['direction']}\n"

        final_text = f"""## Session Complete!

**Total Swipes:** {summary['total_swipes']}
**Likes:** {summary['likes']} | **Dislikes:** {summary['dislikes']}
**Taste Stability:** {summary.get('taste_stability', 0):.2f}

### Your Style Profile
{archetype_text}

### Category Preferences
{cat_text}

### Color Preferences
{color_text}

### Top Brands
{brand_text}

*Check JSON panel for full semantic profile →*
"""
        return None, final_text, get_preferences_json(), gr.update(visible=False)

    info = engine.get_item_info(current_item_id)

    # Load image
    image_path = info.get('image_path')
    if image_path and Path(image_path).exists():
        image = Image.open(image_path)
    else:
        image = None

    # Progress
    progress = current_user.total_swipes
    summary = engine.get_preference_summary(current_user)

    # Feedback status
    consec = current_user.get_consecutive_dislikes()
    feedback_status = f"⚠️ {consec} dislikes in a row" if consec >= 2 else ""

    # Get cluster profile for semantic context
    cluster_id = info.get('cluster', 0)
    cluster_data = summary.get('cluster_health', {}).get(cluster_id, {})
    cluster_profile = cluster_data.get('profile', 'exploring...')

    # Format info text
    info_text = f"""**{info.get('category', 'Unknown')}** | {info.get('archetype', 'classic')}

**Brand:** {info.get('brand', 'Unknown')}
**Color:** {info.get('color', 'Unknown')}
**Fit:** {info.get('fit', 'Unknown')}

*Swipe {progress + 1} | Cluster: {cluster_profile}*
*Coverage: {summary.get('coverage', '0%')} | Stability: {summary.get('taste_stability', 0):.2f}*
{feedback_status}
"""

    return image, info_text, get_preferences_json(), gr.update(visible=True)


def get_preferences_json() -> str:
    """Get current preferences as formatted JSON."""
    if not current_user:
        return "{}"

    summary = engine.get_preference_summary(current_user)
    return json.dumps(summary, indent=2)


def swipe(action: str) -> tuple:
    """Handle swipe action."""
    global current_user, current_item_id

    if not current_user or not current_item_id:
        return None, "Start a session first!", "{}", gr.update(visible=False)

    swipe_action = {
        "like": SwipeAction.LIKE,
        "dislike": SwipeAction.DISLIKE,
        "skip": SwipeAction.SKIP
    }.get(action, SwipeAction.SKIP)

    current_user = engine.record_swipe(current_user, current_item_id, swipe_action)

    result = engine.get_next_item(current_user)
    current_item_id, session_complete = result

    return get_display_data(session_complete)


def like():
    return swipe("like")

def dislike():
    return swipe("dislike")

def skip():
    return swipe("skip")


# Build Gradio UI
with gr.Blocks(title="Style Swipe") as demo:
    gr.Markdown("# Style Swipe")
    gr.Markdown("*Discover your style through swipes*")

    with gr.Row():
        # Left: Setup
        with gr.Column(scale=1):
            gr.Markdown("### Setup")
            color_filter = gr.Dropdown(
                choices=["pink", "yellow", "neon", "orange", "purple", "red", "green", "blue", "brown", "grey", "black", "white", "navy", "beige"],
                multiselect=True,
                label="Colors to Avoid",
                value=["pink", "yellow", "neon"]
            )
            start_btn = gr.Button("Start Session", variant="primary")

            gr.Markdown("""
            ### How It Works
            - **12 visual clusters** for diversity
            - **Semantic taxonomy** for interpretation
            - **Feedback-driven**: dislikes move away
            - **~20 swipes** to learn your style
            """)

        # Middle: Swipe area
        with gr.Column(scale=2):
            item_image = gr.Image(
                label="",
                height=400,
                show_label=False
            )
            item_info = gr.Markdown("*Click 'Start Session' to begin*")

            with gr.Row(visible=False) as button_row:
                dislike_btn = gr.Button("👎 Dislike", variant="secondary", scale=1)
                skip_btn = gr.Button("⏭️ Skip", variant="secondary", scale=1)
                like_btn = gr.Button("👍 Like", variant="primary", scale=1)

        # Right: Learned preferences
        with gr.Column(scale=1):
            gr.Markdown("### Learned Preferences")
            preferences_json = gr.Code(
                label="",
                language="json",
                value="{}",
                lines=25
            )

    # Event handlers
    outputs = [item_image, item_info, preferences_json, button_row]

    start_btn.click(
        fn=start_session,
        inputs=[color_filter],
        outputs=outputs
    )

    like_btn.click(fn=like, outputs=outputs)
    dislike_btn.click(fn=dislike, outputs=outputs)
    skip_btn.click(fn=skip, outputs=outputs)

    gr.Markdown("""
    ---
    **How Learning Works:**
    1. **Visual clusters** ensure diversity (explore different styles)
    2. **Taxonomy** interprets WHY you like things (archetypes, visual anchors)
    3. **Consecutive dislikes** → system moves away FAST
    4. **Taste vector** learns your preferences, anti-taste avoids similar dislikes
    5. **No category rejection** - we learn WITHIN categories what you prefer
    """)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
