"""
CanvasService — business logic for the POSE Canvas module.

Responsibilities:
- CRUD for user inspirations (upload / URL / Pinterest)
- Supabase Storage management for uploaded images
- Taste vector recomputation on every add/remove
- Style element aggregation mapped to feed query-param names
- Closest-product lookup for complete-the-fit
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from supabase import Client

from canvas.image_processor import (
    _ATTR_TO_FEED_PARAM,
    _STYLE_ATTR_KEYS,
    classify_style,
    encode_from_bytes,
    encode_from_url,
)
from canvas.models import (
    AttributeScore,
    DeleteInspirationResponse,
    InspirationResponse,
    InspirationSource,
    StyleElementsResponse,
)
from config.settings import get_settings
from core.logging import get_logger
from integrations.pinterest_style import (
    extract_pin_image_urls,
    get_pinterest_style_extractor,
)

logger = get_logger(__name__)


class CanvasService:
    """Stateless service — every public method receives the Supabase client."""

    # -----------------------------------------------------------------------
    # Inspiration CRUD
    # -----------------------------------------------------------------------

    def list_inspirations(
        self,
        user_id: str,
        supabase: Client,
    ) -> List[InspirationResponse]:
        result = (
            supabase.table("user_inspirations")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        rows = result.data or []
        return [_row_to_response(r) for r in rows]

    # -- Upload -----------------------------------------------------------

    def add_inspiration_upload(
        self,
        user_id: str,
        image_bytes: bytes,
        filename: str,
        supabase: Client,
    ) -> InspirationResponse:
        """Encode uploaded image, store in Supabase Storage, insert row."""
        settings = get_settings()
        self._check_quota(user_id, supabase)

        # 1. Encode
        embedding = encode_from_bytes(image_bytes)

        # 2. Upload to Supabase Storage
        ext = filename.rsplit(".", 1)[-1] if "." in filename else "jpg"
        storage_path = f"{user_id}/{uuid.uuid4().hex}.{ext}"
        bucket = settings.canvas_storage_bucket

        try:
            supabase.storage.from_(bucket).upload(
                path=storage_path,
                file=image_bytes,
                file_options={"content-type": _guess_content_type(ext)},
            )
            image_url = supabase.storage.from_(bucket).get_public_url(storage_path)
        except Exception:
            logger.exception("Supabase Storage upload failed")
            # Fallback: store without a hosted URL (shouldn't happen in prod)
            image_url = f"storage://{bucket}/{storage_path}"

        # 3. Classify style
        style_label, style_confidence, style_attrs = classify_style(
            embedding, supabase, nearest_k=settings.canvas_style_nearest_k,
        )

        # 4. Insert row
        row = self._insert_inspiration(
            supabase=supabase,
            user_id=user_id,
            source=InspirationSource.upload,
            image_url=image_url,
            embedding=embedding,
            style_label=style_label,
            style_confidence=style_confidence,
            style_attributes=style_attrs,
            original_url=None,
            title=filename,
            pinterest_pin_id=None,
        )

        # 5. Recompute taste vector
        self.recompute_taste_vector(user_id, supabase)

        return _row_to_response(row)

    # -- URL --------------------------------------------------------------

    def add_inspiration_url(
        self,
        user_id: str,
        url: str,
        title: Optional[str],
        supabase: Client,
    ) -> InspirationResponse:
        """Fetch image from URL, encode, and insert row."""
        settings = get_settings()
        self._check_quota(user_id, supabase)

        embedding = encode_from_url(url)

        style_label, style_confidence, style_attrs = classify_style(
            embedding, supabase, nearest_k=settings.canvas_style_nearest_k,
        )

        row = self._insert_inspiration(
            supabase=supabase,
            user_id=user_id,
            source=InspirationSource.url,
            image_url=url,
            embedding=embedding,
            style_label=style_label,
            style_confidence=style_confidence,
            style_attributes=style_attrs,
            original_url=url,
            title=title,
            pinterest_pin_id=None,
        )

        self.recompute_taste_vector(user_id, supabase)
        return _row_to_response(row)

    # -- Pinterest --------------------------------------------------------

    def add_inspiration_pinterest(
        self,
        user_id: str,
        supabase: Client,
        pin_ids: Optional[List[str]] = None,
        max_pins: Optional[int] = None,
    ) -> List[InspirationResponse]:
        """
        Sync Pinterest pins as inspirations.

        If *pin_ids* is provided, fetch those specific pins.
        Otherwise fall back to the user's saved board/section selection
        via PinterestClient.fetch_pins().
        """
        from integrations.pinterest_client import PinterestClient

        settings = get_settings()
        max_pins = max_pins or settings.pinterest_default_max_pins
        client = PinterestClient()

        # Fetch pins (respects saved selection when pin_ids is None)
        if pin_ids:
            # Fetch specific pins — we still go through the client to
            # handle token refresh, but limit to the requested IDs.
            all_pins = client.fetch_pins(user_id, max_pins=max_pins)
            pins = [p for p in all_pins if p.get("id") in set(pin_ids)]
        else:
            pins = client.fetch_pins(user_id, max_pins=max_pins)

        # Find which pins are already stored
        existing_pin_ids = self._existing_pinterest_pin_ids(user_id, supabase)

        added: List[InspirationResponse] = []
        extractor = get_pinterest_style_extractor()

        for pin in pins:
            pid = pin.get("id")
            if not pid or pid in existing_pin_ids:
                continue

            # Check quota per iteration (early exit if maxed)
            try:
                self._check_quota(user_id, supabase)
            except Exception:
                logger.warning("Inspiration quota reached during Pinterest sync",
                               user_id=user_id, added_so_far=len(added))
                break

            # Extract best image URL
            image_urls = extract_pin_image_urls([pin])
            if not image_urls:
                continue

            try:
                embedding = encode_from_url(image_urls[0])
            except Exception:
                logger.warning("Failed to encode Pinterest pin image",
                               pin_id=pid, url=image_urls[0])
                continue

            style_label, style_confidence, style_attrs = classify_style(
                embedding, supabase, nearest_k=settings.canvas_style_nearest_k,
            )

            row = self._insert_inspiration(
                supabase=supabase,
                user_id=user_id,
                source=InspirationSource.pinterest,
                image_url=image_urls[0],
                embedding=embedding,
                style_label=style_label,
                style_confidence=style_confidence,
                style_attributes=style_attrs,
                original_url=pin.get("link"),
                title=pin.get("title"),
                pinterest_pin_id=pid,
            )
            added.append(_row_to_response(row))

        # Recompute taste vector once after all pins processed
        if added:
            self.recompute_taste_vector(user_id, supabase)

        return added

    # -- Delete -----------------------------------------------------------

    def remove_inspiration(
        self,
        user_id: str,
        inspiration_id: str,
        supabase: Client,
    ) -> DeleteInspirationResponse:
        """Delete an inspiration and recompute the taste vector."""
        # Fetch row first (need image_url for Storage cleanup)
        result = (
            supabase.table("user_inspirations")
            .select("id, source, image_url")
            .eq("id", inspiration_id)
            .eq("user_id", user_id)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return DeleteInspirationResponse(
                deleted=False, taste_vector_updated=False, remaining_count=0,
            )

        row = rows[0]

        # Delete from Storage if it was an upload
        if row["source"] == InspirationSource.upload.value:
            self._delete_from_storage(row["image_url"], supabase)

        # Delete row
        supabase.table("user_inspirations").delete().eq("id", inspiration_id).execute()

        # Recompute
        tv_updated = self.recompute_taste_vector(user_id, supabase)

        remaining = (
            supabase.table("user_inspirations")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        remaining_count = remaining.count if remaining.count is not None else 0

        return DeleteInspirationResponse(
            deleted=True,
            taste_vector_updated=tv_updated,
            remaining_count=remaining_count,
        )

    # -----------------------------------------------------------------------
    # Taste vector
    # -----------------------------------------------------------------------

    def recompute_taste_vector(
        self,
        user_id: str,
        supabase: Client,
    ) -> bool:
        """
        Average all inspiration embeddings and update the user's taste vector
        via the ``update_user_taste_vector`` RPC.

        Returns True if the vector was updated, False if no inspirations exist.
        """
        result = (
            supabase.table("user_inspirations")
            .select("embedding")
            .eq("user_id", user_id)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return False

        embeddings = []
        for row in rows:
            raw = row.get("embedding")
            if raw is None:
                continue
            # pgvector returns embedding as a string "[0.01,-0.02,...]"
            if isinstance(raw, str):
                raw = raw.strip("[]")
                vec = np.fromstring(raw, sep=",", dtype=np.float64)
            elif isinstance(raw, list):
                vec = np.array(raw, dtype=np.float64)
            else:
                continue
            if vec.shape == (512,):
                embeddings.append(vec)

        if not embeddings:
            return False

        mean_vec = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm

        taste_list = mean_vec.astype("float64").tolist()

        try:
            supabase.rpc(
                "update_user_taste_vector",
                {"p_user_id": user_id, "p_taste_vector": taste_list},
            ).execute()
            logger.info(
                "Taste vector updated from canvas inspirations",
                user_id=user_id,
                n_inspirations=len(embeddings),
            )
            return True
        except Exception:
            logger.exception("update_user_taste_vector RPC failed")
            return False

    # -----------------------------------------------------------------------
    # Style elements
    # -----------------------------------------------------------------------

    def get_style_elements(
        self,
        user_id: str,
        supabase: Client,
        min_count: int = 2,
        min_confidence: float = 0.25,
    ) -> StyleElementsResponse:
        """
        Aggregate ``style_attributes`` across all inspirations and build
        a feed-param-ready ``suggested_filters`` dict.
        """
        result = (
            supabase.table("user_inspirations")
            .select("style_attributes")
            .eq("user_id", user_id)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return StyleElementsResponse(inspiration_count=0)

        # Merge all per-inspiration distributions
        merged: Dict[str, Counter] = {k: Counter() for k in _STYLE_ATTR_KEYS}

        for row in rows:
            attrs = row.get("style_attributes") or {}
            for key in _STYLE_ATTR_KEYS:
                dist = attrs.get(key)
                if not isinstance(dist, dict):
                    continue
                for value, score in dist.items():
                    # Each inspiration's contribution is weighted by its
                    # own normalised confidence for that value.
                    try:
                        merged[key][value] += float(score)
                    except (TypeError, ValueError):
                        pass

        n = len(rows)

        # Build raw_attributes + suggested_filters
        raw_attributes: Dict[str, List[AttributeScore]] = {}
        suggested_filters: Dict[str, List[str]] = {}

        for key in _STYLE_ATTR_KEYS:
            counter = merged[key]
            if not counter:
                continue

            total = sum(counter.values())
            scores: List[AttributeScore] = []
            for value, raw_count in counter.most_common():
                confidence = raw_count / total if total > 0 else 0.0
                # count = how many inspirations contributed this value
                # (approximate: raw_count can exceed n for array attrs)
                approx_count = min(round(raw_count), n)
                scores.append(AttributeScore(
                    value=value,
                    count=approx_count,
                    confidence=round(confidence, 3),
                ))

            raw_attributes[key] = scores

            # Filter into suggested_filters using thresholds
            feed_param = _ATTR_TO_FEED_PARAM.get(key)
            if feed_param:
                accepted = [
                    s.value for s in scores
                    if s.count >= min_count or s.confidence >= min_confidence
                ]
                if accepted:
                    suggested_filters[feed_param] = accepted

        return StyleElementsResponse(
            suggested_filters=suggested_filters,
            raw_attributes=raw_attributes,
            inspiration_count=n,
        )

    # -----------------------------------------------------------------------
    # Complete-the-fit helpers
    # -----------------------------------------------------------------------

    def _get_inspiration_embedding_str(
        self,
        inspiration_id: str,
        user_id: str,
        supabase: Client,
    ) -> Optional[str]:
        """Return the embedding of an inspiration as a pgvector-ready string, or None."""
        result = (
            supabase.table("user_inspirations")
            .select("embedding")
            .eq("id", inspiration_id)
            .eq("user_id", user_id)
            .execute()
        )
        rows = result.data or []
        if not rows:
            return None

        raw = rows[0].get("embedding")
        if raw is None:
            return None

        if isinstance(raw, str):
            raw = raw.strip("[]")
            vec = np.fromstring(raw, sep=",", dtype=np.float64)
        elif isinstance(raw, list):
            vec = np.array(raw, dtype=np.float64)
        else:
            return None

        return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"

    def _canvas_search(
        self,
        embedding_str: str,
        supabase: Client,
        count: int,
    ) -> List[Dict[str, Any]]:
        """
        Two-step search: (1) ``canvas_similar_search`` RPC to find nearest
        sku_ids via HNSW on ``image_embeddings`` (no products JOIN so the
        index is always used), then (2) batch-fetch product details.

        Falls back to ``match_products_with_hard_filters`` if the canvas
        RPC is not yet deployed.
        """
        # --- Step 1: HNSW nearest-neighbor on image_embeddings ----------
        try:
            nn_result = supabase.rpc(
                "canvas_similar_search",
                {"query_embedding": embedding_str, "match_count": count},
            ).execute()
        except Exception:
            # RPC not deployed yet — fall back to old approach
            logger.warning("canvas_similar_search RPC unavailable, falling back")
            try:
                result = supabase.rpc(
                    "match_products_with_hard_filters",
                    {"query_embedding": embedding_str, "match_count": count},
                ).execute()
                return result.data or []
            except Exception:
                logger.exception("Fallback RPC also failed")
                return []

        nn_rows = nn_result.data or []
        if not nn_rows:
            return []

        # Build similarity lookup and deduplicate sku_ids
        sim_by_sku: Dict[str, float] = {}
        for row in nn_rows:
            sid = str(row["sku_id"])
            if sid not in sim_by_sku:
                sim_by_sku[sid] = row["similarity"]

        sku_ids = list(sim_by_sku.keys())

        # --- Step 2: batch-fetch product details -----------------------
        # Supabase .in_() has a practical limit; chunk if needed.
        products: List[Dict[str, Any]] = []
        chunk_size = 200
        for i in range(0, len(sku_ids), chunk_size):
            chunk = sku_ids[i : i + chunk_size]
            try:
                pr = (
                    supabase.table("products")
                    .select(
                        "id,name,brand,category,broad_category,colors,materials,"
                        "price,original_price,fit,length,sleeve,neckline,"
                        "style_tags,primary_image_url,hero_image_url,in_stock"
                    )
                    .in_("id", chunk)
                    .eq("in_stock", True)
                    .execute()
                )
                products.extend(pr.data or [])
            except Exception:
                logger.exception("Product batch fetch failed")

        # --- Step 3: merge similarity + product details ----------------
        prod_by_id = {str(p["id"]): p for p in products}
        merged: List[Dict[str, Any]] = []
        for sid in sku_ids:
            prod = prod_by_id.get(sid)
            if not prod:
                continue
            prod["product_id"] = prod.pop("id")
            prod["similarity"] = sim_by_sku[sid]
            merged.append(prod)

        # Sort by similarity descending (HNSW order may differ slightly after JOIN)
        merged.sort(key=lambda x: x.get("similarity", 0), reverse=True)
        return merged

    def find_closest_product(
        self,
        inspiration_id: str,
        user_id: str,
        supabase: Client,
    ) -> Optional[Dict[str, Any]]:
        """
        Find the single closest real product to an inspiration's embedding.

        Uses the two-step ``_canvas_search`` approach so the HNSW index
        is always utilised, even for external (non-catalog) embeddings.
        """
        embedding_str = self._get_inspiration_embedding_str(
            inspiration_id, user_id, supabase,
        )
        if not embedding_str:
            return None

        results = self._canvas_search(embedding_str, supabase, count=5)
        return results[0] if results else None

    # Max items from the same brand in similar-items results
    _SIMILAR_MAX_PER_BRAND = 3

    def find_similar_products(
        self,
        inspiration_id: str,
        user_id: str,
        supabase: Client,
        count: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Find the *count* most similar real products to an inspiration's
        embedding via pgvector cosine similarity.

        Five-pass dedup (mirrors the search reranker + feed pipeline):

          1. **product_id** — removes multi-image rows for the same product.
          2. **SessionReranker._deduplicate** — sister-brand mapping,
             size-variant name normalisation, same image-URL dedup.
          3. **deduplicate_dicts** — image-hash regex dedup + exact
             name/brand (catches cross-brand clones sharing the same
             product photo URL pattern).
          4. **Fuzzy name** — ``SequenceMatcher`` ratio ≥ 0.80 within the
             same (canonical) brand.  Catches near-variants like
             "SoftActive Flare Leggings" vs "Active Flare Leggings".
          5. **Brand diversity cap** — max ``_SIMILAR_MAX_PER_BRAND``
             items per brand so results aren't dominated by one retailer.

        Over-fetches 5× to ensure enough unique products survive all passes.
        """
        from difflib import SequenceMatcher

        from recs.filter_utils import deduplicate_dicts
        from search.reranker import SessionReranker

        embedding_str = self._get_inspiration_embedding_str(
            inspiration_id, user_id, supabase,
        )
        if not embedding_str:
            return []

        rows = self._canvas_search(embedding_str, supabase, count=count * 5)

        # ---- Pass 1: product_id dedup (multiple images per product) ----
        seen_ids: set = set()
        id_unique: List[Dict[str, Any]] = []
        for row in rows:
            pid = row.get("product_id")
            if pid in seen_ids:
                continue
            seen_ids.add(pid)
            # Reranker reads "image_url"; RPC returns "primary_image_url"
            if "image_url" not in row:
                row["image_url"] = row.get("primary_image_url") or row.get("hero_image_url") or ""
            id_unique.append(row)

        # ---- Pass 2: reranker dedup (sister brands, size variants, image URL) ----
        reranker = SessionReranker()
        reranker_deduped = reranker._deduplicate(id_unique)

        # ---- Pass 3: image-hash + exact name/brand dedup ----
        hash_deduped = deduplicate_dicts(reranker_deduped)

        # ---- Pass 4: fuzzy same-brand name dedup ----
        fuzzy_deduped: List[Dict[str, Any]] = []
        accepted_by_brand: Dict[str, List[str]] = {}
        _normalize = SessionReranker._normalize_name
        _sister = SessionReranker._SISTER_BRANDS

        for item in hash_deduped:
            brand = (item.get("brand") or "").lower().strip()
            canon = _sister.get(brand, brand)
            name = _normalize(item.get("name") or "")

            if canon and name:
                is_dup = False
                for prev in accepted_by_brand.get(canon, []):
                    if SequenceMatcher(None, name, prev).ratio() >= 0.80:
                        is_dup = True
                        break
                if is_dup:
                    continue
                accepted_by_brand.setdefault(canon, []).append(name)

            fuzzy_deduped.append(item)

        # ---- Pass 5: brand diversity cap ----
        brand_counts: Dict[str, int] = {}
        final: List[Dict[str, Any]] = []
        max_brand = self._SIMILAR_MAX_PER_BRAND

        for item in fuzzy_deduped:
            brand = (item.get("brand") or "unknown").lower().strip()
            canon = _sister.get(brand, brand)
            if brand_counts.get(canon, 0) >= max_brand:
                continue
            final.append(item)
            brand_counts[canon] = brand_counts.get(canon, 0) + 1
            if len(final) >= count:
                break

        return final

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _check_quota(self, user_id: str, supabase: Client) -> None:
        settings = get_settings()
        result = (
            supabase.table("user_inspirations")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .execute()
        )
        current = result.count if result.count is not None else 0
        if current >= settings.canvas_max_inspirations:
            raise ValueError(
                f"Inspiration limit reached ({settings.canvas_max_inspirations}). "
                "Remove some inspirations before adding new ones."
            )

    def _insert_inspiration(
        self,
        supabase: Client,
        user_id: str,
        source: InspirationSource,
        image_url: str,
        embedding: np.ndarray,
        style_label: Optional[str],
        style_confidence: float,
        style_attributes: Dict[str, Any],
        original_url: Optional[str],
        title: Optional[str],
        pinterest_pin_id: Optional[str],
    ) -> Dict[str, Any]:
        embedding_str = "[" + ",".join(f"{float(v):.8f}" for v in embedding) + "]"

        payload: Dict[str, Any] = {
            "user_id": user_id,
            "source": source.value,
            "image_url": image_url,
            "embedding": embedding_str,
            "style_label": style_label,
            "style_confidence": style_confidence,
            "style_attributes": style_attributes,
            "original_url": original_url,
            "title": title,
            "pinterest_pin_id": pinterest_pin_id,
        }

        result = (
            supabase.table("user_inspirations")
            .insert(payload)
            .execute()
        )

        return result.data[0]

    def _existing_pinterest_pin_ids(
        self,
        user_id: str,
        supabase: Client,
    ) -> set:
        result = (
            supabase.table("user_inspirations")
            .select("pinterest_pin_id")
            .eq("user_id", user_id)
            .not_.is_("pinterest_pin_id", "null")
            .execute()
        )
        return {r["pinterest_pin_id"] for r in (result.data or [])}

    def _delete_from_storage(self, image_url: str, supabase: Client) -> None:
        """Best-effort delete of an uploaded image from Supabase Storage."""
        settings = get_settings()
        bucket = settings.canvas_storage_bucket
        # image_url looks like https://<project>.supabase.co/storage/v1/object/public/<bucket>/<path>
        # We need to extract the path after the bucket name.
        marker = f"/storage/v1/object/public/{bucket}/"
        if marker in image_url:
            path = image_url.split(marker, 1)[1]
            try:
                supabase.storage.from_(bucket).remove([path])
            except Exception:
                logger.warning("Failed to delete from Storage", path=path)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _row_to_response(row: Dict[str, Any]) -> InspirationResponse:
    """Convert a Supabase row dict into an ``InspirationResponse``."""
    return InspirationResponse(
        id=str(row["id"]),
        source=InspirationSource(row["source"]),
        image_url=row["image_url"],
        original_url=row.get("original_url"),
        title=row.get("title"),
        style_label=row.get("style_label"),
        style_confidence=row.get("style_confidence"),
        style_attributes=row.get("style_attributes") or {},
        pinterest_pin_id=row.get("pinterest_pin_id"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _guess_content_type(ext: str) -> str:
    return {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "webp": "image/webp",
        "gif": "image/gif",
    }.get(ext.lower(), "image/jpeg")


# Singleton
_canvas_service: Optional[CanvasService] = None


def get_canvas_service() -> CanvasService:
    global _canvas_service
    if _canvas_service is None:
        _canvas_service = CanvasService()
    return _canvas_service
