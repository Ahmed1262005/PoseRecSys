# Swipe System Algorithms & Concepts

This document explains all the algorithms, mathematical concepts, and design decisions used across the style learning system.

---

## Table of Contents
1. [System Overview](#system-overview)
2. [Core Concepts](#core-concepts)
3. [SwipeEngine (Base)](#swipeengine-base)
4. [FourChoiceEngine](#fourchoiceengine)
5. [RankingEngine](#rankingengine)
6. [AttributeTestEngine](#attributetestengine)
7. [Information Theory Analysis](#information-theory-analysis)
8. [Comparison of Approaches](#comparison-of-approaches)

---

## System Overview

The style learning system uses **four different engines** for learning user preferences:

| Engine | Interaction Type | Info/Round | Convergence | Best For |
|--------|------------------|------------|-------------|----------|
| SwipeEngine | Like/Dislike | 1 bit | 40+ swipes | Continuous discovery |
| FourChoiceEngine | Pick 1 of 4 | 2 bits | 10-15 rounds | Quick preferences |
| RankingEngine | Rank 6 items | 9.5 bits | 4-6 rounds | Fastest convergence |
| AttributeTestEngine | Pick 1 of 4 (structured) | 2 bits + semantic | 12-20 rounds | Interpretable preferences |

---

## Core Concepts

### 1. CLIP Embeddings (FashionCLIP)

All engines use **FashionCLIP** 512-dimensional embeddings as the visual representation.

```
Image → FashionCLIP → 512-dim vector → Normalized to unit sphere
```

**Properties:**
- **Semantic similarity**: Similar-looking items have high cosine similarity
- **Trained on fashion**: Better than generic CLIP for clothing
- **Unit normalized**: `||embedding|| = 1` for cosine similarity as dot product

### 2. Taste Vector

A learned representation of what the user likes in embedding space.

```python
taste_vector = weighted_mean(embeddings of liked items)
taste_vector = taste_vector / ||taste_vector||  # Normalize
```

**Update rule:**
```python
# Exponential moving average
taste_vector = α * old_taste + (1-α) * new_embedding
# α = 0.7 for swipe, 0.6 for ranking (more weight on new signal)
```

### 3. Anti-Taste Vector

A learned representation of what the user dislikes (for contrastive selection).

```python
anti_taste_vector = weighted_mean(embeddings of recent dislikes)
```

**Usage:**
```python
score = cosine_sim(item, taste_vector) - β * cosine_sim(item, anti_taste_vector)
# β = 0.3-0.5 (penalty weight)
```

### 4. Visual Clustering (K-Means)

Items are clustered into **12 visual clusters** for exploration coverage.

```python
kmeans = KMeans(n_clusters=12)
cluster_labels = kmeans.fit_predict(normalized_embeddings)
```

**Purpose:**
- Ensure user sees items from different visual regions
- Track which regions user has explored
- Reject clusters with strong negative signal

### 5. Cluster Health (Bayesian)

Tracks user preference for each cluster using Bayesian smoothing:

```python
cluster_health = (likes + 1) / (total + 2)  # Bayesian prior
```

**Rejection rule:**
```python
if dislike_rate >= 0.80 and total_samples >= 2:
    mark_cluster_as_rejected(cluster)
```

### 6. Taxonomy System (Archetypes & Anchors)

Items are scored on semantic dimensions:

**Archetypes** (style types):
- `classic` - timeless, clean
- `creative_artistic` - bold graphics, unique
- `natural_sporty` - athletic, performance
- `dramatic_street` - streetwear, high contrast
- `minimalist` - understated, simple

**Anchors** (visual features):
- `solids` - solid colors
- `graphics` - graphic prints
- `typography` - text elements
- `athletic_cues` - sporty details
- `contrast` - high contrast designs

---

## SwipeEngine (Base)

The foundation engine using binary like/dislike feedback.

### Algorithm: Feedback-Driven Selection

```
┌─────────────────────────────────────────────────────────────┐
│                    SELECTION STRATEGY                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  if consecutive_dislikes >= 2:                              │
│      → RECOVERY MODE: Select items close to taste,          │
│        far from anti-taste (escape bad region)              │
│                                                              │
│  else if underexplored_clusters exist:                      │
│      → EXPLORATION: Random item from least-explored cluster │
│        (weighted by cluster health)                          │
│                                                              │
│  else:                                                       │
│      → EXPLOITATION: 50% taste-based, 50% random            │
│        (with diversity constraints)                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Taste Vector Update

```python
def compute_taste_vector(liked_ids):
    embeddings = [normalize(get_embedding(id)) for id in liked_ids]
    taste = mean(embeddings)
    return normalize(taste)
```

### Anti-Taste Vector (Recency-Weighted)

```python
def compute_anti_taste_vector(recent_actions):
    recent_dislikes = [id for id, action in recent_actions[-5:]
                       if action == DISLIKE]
    embeddings = [normalize(get_embedding(id)) for id in recent_dislikes]
    anti_taste = mean(embeddings)
    return normalize(anti_taste)
```

### Diversity Enforcement

```python
def is_diverse_enough(item, recent_items):
    """Ensure item differs from recent items in color OR category."""
    for recent in recent_items:
        if item.color == recent.color AND item.category == recent.category:
            return False
    return True
```

### Stopping Conditions

1. **Minimum swipes**: 40 swipes required
2. **Coverage**: 70% of non-rejected clusters explored
3. **Taste stability**: `cosine_sim(recent_vectors) >= 0.95`

---

## FourChoiceEngine

Pick 1 of 4 items, providing ~2 bits of information per round.

### Algorithm: Contrast-Based Selection

Each set of 4 items maximizes **contrast** across dimensions:

```python
CONTRAST_DIMENSIONS = ['archetype', 'category', 'color_family', 'fit']

def select_four_items(prefs, available):
    selected = []
    used_values = {dim: set() for dim in CONTRAST_DIMENSIONS}

    for candidate in available:
        contrast_score = calculate_contrast(candidate, selected, used_values)
        if contrast_score > threshold:
            selected.append(candidate)
            update_used_values(candidate, used_values)

        if len(selected) >= 4:
            break

    return selected
```

### Contrastive Taste Update

```python
def update_taste(prefs, winner, losers):
    winner_emb = normalize(get_embedding(winner))
    loser_embs = [normalize(get_embedding(l)) for l in losers]
    loser_mean = normalize(mean(loser_embs))

    # Move toward winner
    prefs.taste_vector = 0.7 * prefs.taste_vector + 0.3 * winner_emb
    prefs.taste_vector = normalize(prefs.taste_vector)

    # Move anti-taste toward losers
    prefs.anti_taste_vector = 0.8 * prefs.anti_taste_vector + 0.2 * loser_mean
    prefs.anti_taste_vector = normalize(prefs.anti_taste_vector)
```

### Color Families for Contrast

```python
COLOR_FAMILIES = {
    'dark': {'black', 'navy', 'charcoal'},
    'light': {'white', 'cream', 'beige'},
    'cool': {'blue', 'teal', 'purple'},
    'warm': {'red', 'orange', 'yellow'},
    'earth': {'brown', 'olive', 'tan'},
}
```

---

## RankingEngine

Rank 6 items from best to worst, providing ~9.5 bits per round.

### Information Theory

```
6 items → 6! = 720 possible orderings
Information = log₂(720) ≈ 9.5 bits per round
```

**Comparison:**
- Swipe: 1 bit (like/dislike)
- Four-choice: log₂(4) = 2 bits
- Ranking 6: log₂(720) ≈ 9.5 bits

### Algorithm: Weighted Contrastive Learning

For a ranking [A > B > C > D > E > F], each pair provides signal:

```python
def update_taste_from_ranking(prefs, ranked_ids):
    n = len(ranked_ids)
    max_distance = n - 1

    positive_direction = zeros(512)
    negative_direction = zeros(512)

    for i in range(n):
        for j in range(i + 1, n):
            # i is ranked higher than j
            distance = j - i
            weight = distance / max_distance  # Normalize to [0, 1]

            # Higher-ranked item contributes to positive direction
            positive_direction += weight * embeddings[i]

            # Lower-ranked item contributes to negative direction
            negative_direction += weight * embeddings[j]

    # Normalize and update
    positive_direction = normalize(positive_direction)
    negative_direction = normalize(negative_direction)

    # Higher learning rate (0.4) due to richer signal
    prefs.taste_vector = 0.6 * prefs.taste_vector + 0.4 * positive_direction
```

### Pairwise Comparisons

With 6 items, we get `6 * 5 / 2 = 15` pairwise comparisons per round.

```
A > B, A > C, A > D, A > E, A > F  (5 comparisons)
B > C, B > D, B > E, B > F         (4 comparisons)
C > D, C > E, C > F                (3 comparisons)
D > E, D > F                       (2 comparisons)
E > F                              (1 comparison)
Total: 15 pairwise comparisons
```

### Selection Strategy

```
Rounds 1-2: EXPLORATION
    → 6 items from 6 different clusters (maximum coverage)

Rounds 3+: MIXED
    → 3 taste-aligned items (exploitation)
    → 3 items from underexplored clusters (exploration)
```

---

## AttributeTestEngine

Structured testing of ONE attribute at a time.

### Core Insight: Controlled Experiment

```
Traditional: Items vary on ALL dimensions
    → Hard to know WHY user chose an item

Attribute Isolation: Items vary on ONE dimension
    → Clear signal about that specific preference
```

### 6 Phases of Testing

```
Phase 1: Style Foundation
    - archetype (2 rounds)

Phase 2: Category Exploration
    - category (2 rounds)
    - logo_style (2 rounds)

Phase 3: Fit & Form
    - fit (2 rounds)
    - neckline (2 rounds)

Phase 4: Visual Elements
    - pattern_density (2 rounds)

Phase 5: Color Palette
    - color_family (2 rounds)
    - color_tone (2 rounds)

Phase 6: Material & Quality
    - material (1 round)
```

### Algorithm: Attribute Isolation Selection

```python
def select_attribute_isolated_items(prefs, available, test_attr):
    """
    Select 4 items that are visually similar but differ on test_attr.
    """
    # Get reference embedding (taste-aligned)
    reference = prefs.taste_vector or prefs.reference_embedding

    # Find visually similar items
    similar_items = find_similar_items(reference, available, k=100)

    # Group by test attribute value
    attr_groups = defaultdict(list)
    for item_id, similarity in similar_items:
        attr_value = get_attribute_value(item_id, test_attr)
        attr_groups[attr_value].append((item_id, similarity))

    # Select one item from each attribute value (prefer higher similarity)
    selected = []
    for attr_value in ATTRIBUTE_VALUES[test_attr]:
        if attr_value in attr_groups and len(selected) < 4:
            best_item = attr_groups[attr_value][0][0]
            selected.append(best_item)

    return selected
```

### Attribute Score Tracking

```python
def record_choice(prefs, winner_id, all_shown):
    current_attr = prefs.current_test_attribute

    winner_value = get_attribute_value(winner_id, current_attr)
    loser_values = [get_attribute_value(id, current_attr) for id in losers]

    # Winner gets positive score
    prefs.attribute_scores[current_attr][winner_value] += 1.0

    # Losers get negative scores
    for lv in loser_values:
        prefs.attribute_scores[current_attr][lv] -= 0.33
```

### Preference Locking

After 2 rounds on an attribute, lock in the preference:

```python
def maybe_lock_attribute(prefs):
    if prefs.rounds_on_current_attr >= ROUNDS_PER_ATTR:
        scores = prefs.attribute_scores[current_attr]
        best_value = max(scores.items(), key=lambda x: x[1])[0]

        # Calculate confidence
        sorted_scores = sorted(scores.values(), reverse=True)
        confidence = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]

        prefs.locked_attributes[current_attr] = best_value
        prefs.attribute_confidence[current_attr] = confidence
```

---

## Information Theory Analysis

### Bits Per Interaction

| Method | Choices | Bits | Formula |
|--------|---------|------|---------|
| Swipe | 2 | 1 | log₂(2) = 1 |
| Four-choice | 4 | 2 | log₂(4) = 2 |
| Rank 6 | 720 | 9.5 | log₂(6!) ≈ 9.49 |

### Convergence Analysis

Assuming ~50 bits needed for stable preferences:

| Method | Bits/Round | Rounds Needed |
|--------|------------|---------------|
| Swipe | 1 | 50 |
| Four-choice | 2 | 25 |
| Ranking | 9.5 | 6 |

### Information Efficiency

```
Ranking is 4-5x more efficient than four-choice
Ranking is 9.5x more efficient than swipe
```

---

## Comparison of Approaches

### When to Use Each Engine

**SwipeEngine (Binary)**
- Best for: Continuous discovery, cold start
- Pros: Simple UX, low cognitive load
- Cons: Slow convergence, less information per interaction

**FourChoiceEngine (Pick 1)**
- Best for: Quick preference learning, A/B testing
- Pros: More info than swipe, still simple
- Cons: Harder to distinguish nuanced preferences

**RankingEngine (Rank All)**
- Best for: Fastest convergence, nuanced preferences
- Pros: Maximum information, full preference ordering
- Cons: Higher cognitive load, more complex UX

**AttributeTestEngine (Structured)**
- Best for: Interpretable preferences, explainable AI
- Pros: Clear attribute-level preferences, structured output
- Cons: More rounds needed, less flexible

### Hybrid Approach

The system can combine engines:

```
1. Start with AttributeTest for structured preference discovery (12 rounds)
2. Continue with Ranking for rapid refinement (4 rounds)
3. Use Swipe for ongoing personalization
```

---

## Mathematical Summary

### Key Equations

**Cosine Similarity:**
```
sim(a, b) = (a · b) / (||a|| × ||b||)
```
With normalized vectors: `sim(a, b) = a · b`

**Taste Vector Update (EMA):**
```
taste_new = α × taste_old + (1-α) × embedding_new
taste_new = taste_new / ||taste_new||
```

**Bayesian Cluster Health:**
```
health = (likes + 1) / (total + 2)
```

**Information Content:**
```
I = log₂(number_of_possible_outcomes)
```

**Contrastive Weighting (Ranking):**
```
weight(i, j) = |rank_j - rank_i| / max_distance
```

---

## Implementation Files

| File | Engine | Purpose |
|------|--------|---------|
| `swipe_engine.py` | SwipeEngine | Base engine with binary feedback |
| `four_choice_engine.py` | FourChoiceEngine | Pick 1 of 4 with contrast |
| `ranking_engine.py` | RankingEngine | Rank 6 items |
| `attribute_test_engine.py` | AttributeTestEngine | Structured attribute testing |
| `swipe_server.py` | All | FastAPI server with endpoints |

---

## References

1. **FashionCLIP**: `patrickjohncyh/fashion-clip`
2. **Information Theory**: Shannon, C. (1948). A Mathematical Theory of Communication
3. **Contrastive Learning**: SimCLR, MoCo papers
4. **Bayesian Inference**: Beta distribution priors for binary outcomes
5. **K-Means Clustering**: Lloyd's algorithm for visual clustering
