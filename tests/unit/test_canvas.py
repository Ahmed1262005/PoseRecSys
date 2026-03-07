"""
Unit tests for the POSE Canvas module.

Covers:
- Pydantic models (serialisation, validation, defaults)
- image_processor (_aggregate_attributes, _ATTR_TO_FEED_PARAM, classify_style)
- CanvasService (list, add_url, remove, recompute_taste_vector,
  get_style_elements, find_closest_product, quota, storage cleanup)
- route wiring (router registered, correct paths/methods)
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Test data factories
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 3, 5, 12, 0, 0, tzinfo=timezone.utc)


def _make_embedding(seed: int = 42) -> np.ndarray:
    """Deterministic 512-dim L2-normalised embedding."""
    rng = np.random.RandomState(seed)
    vec = rng.randn(512).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_embedding_str(seed: int = 42) -> str:
    """Embedding formatted as pgvector string."""
    vec = _make_embedding(seed)
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


_SENTINEL = object()


def _make_db_row(
    source: str = "url",
    style_label: str = "Boho",
    style_confidence: float = 0.6,
    style_attributes: Any = _SENTINEL,
    pinterest_pin_id: Optional[str] = None,
    **overrides: Any,
) -> Dict[str, Any]:
    """Fake user_inspirations DB row."""
    if style_attributes is _SENTINEL:
        style_attributes = {
            "style_tags": {"Boho": 0.5, "Romantic": 0.3, "Classic": 0.2},
            "pattern": {"floral": 0.6, "solid": 0.4},
            "color_family": {"Neutrals": 0.7, "Browns": 0.3},
        }
    row = {
        "id": str(uuid.uuid4()),
        "user_id": "user-123",
        "source": source,
        "image_url": "https://example.com/image.jpg",
        "original_url": "https://example.com/image.jpg",
        "title": "Test Inspiration",
        "embedding": _make_embedding_str(),
        "style_label": style_label,
        "style_confidence": style_confidence,
        "style_attributes": style_attributes,
        "pinterest_pin_id": pinterest_pin_id,
        "created_at": _NOW.isoformat(),
        "updated_at": _NOW.isoformat(),
    }
    row.update(overrides)
    return row


def _make_product_row(product_id: Optional[str] = None, **overrides) -> Dict[str, Any]:
    """Fake row from match_products_with_hard_filters RPC."""
    row = {
        "product_id": product_id or str(uuid.uuid4()),
        "name": "Boho Floral Maxi Dress",
        "brand": "TestBrand",
        "category": "dresses",
        "broad_category": "dresses",
        "colors": ["beige", "brown"],
        "materials": ["cotton"],
        "price": 59.99,
        "fit": "regular",
        "length": "maxi",
        "sleeve": "short",
        "neckline": "v-neck",
        "style_tags": ["Boho", "Romantic"],
        "primary_image_url": "https://example.com/product.jpg",
        "hero_image_url": "https://example.com/product.jpg",
        "similarity": 0.87,
    }
    row.update(overrides)
    return row


def _make_attr_row(**overrides) -> Dict[str, Any]:
    """Fake product_attributes row."""
    row = {
        "sku_id": str(uuid.uuid4()),
        "style_tags": ["Boho", "Romantic"],
        "pattern": "floral",
        "color_family": "Neutrals",
        "formality": "Casual",
        "occasions": ["casual", "vacation"],
        "silhouette": "A-Line",
        "fit_type": "regular",
        "sleeve_type": "short",
        "neckline": "v-neck",
    }
    row.update(overrides)
    return row


def _mock_supabase_chain() -> MagicMock:
    """
    Build a MagicMock that supports the fluent Supabase chaining pattern:
        supabase.table("x").select("y").eq("a","b").order("c").execute()
    Each chained method returns the same builder mock.
    """
    mock = MagicMock()
    builder = MagicMock()
    # Every chained method returns the builder itself
    for method in ("select", "eq", "in_", "order", "insert", "delete",
                   "not_", "is_", "limit"):
        getattr(builder, method).return_value = builder
    # .not_.is_(...) also needs to return builder
    builder.not_.is_.return_value = builder

    mock.table.return_value = builder
    # Default execute result
    result = MagicMock()
    result.data = []
    result.count = 0
    builder.execute.return_value = result

    # RPC default
    rpc_result = MagicMock()
    rpc_result.data = []
    rpc_exec = MagicMock()
    rpc_exec.execute.return_value = rpc_result
    mock.rpc.return_value = rpc_exec

    # Storage default
    storage_bucket = MagicMock()
    mock.storage.from_.return_value = storage_bucket

    return mock


# =========================================================================
# 1. Model tests
# =========================================================================

class TestModels:
    """Pydantic model validation and serialisation."""

    def test_inspiration_source_enum_values(self):
        from canvas.models import InspirationSource
        assert set(InspirationSource) == {
            InspirationSource.upload,
            InspirationSource.url,
            InspirationSource.camera,
            InspirationSource.pinterest,
        }

    def test_inspiration_response_from_dict(self):
        from canvas.models import InspirationResponse, InspirationSource
        data = _make_db_row()
        resp = InspirationResponse(
            id=data["id"],
            source=InspirationSource(data["source"]),
            image_url=data["image_url"],
            style_label=data["style_label"],
            style_attributes=data["style_attributes"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
        assert resp.source == InspirationSource.url
        assert resp.style_label == "Boho"

    def test_inspiration_response_defaults(self):
        from canvas.models import InspirationResponse, InspirationSource
        resp = InspirationResponse(
            id="abc",
            source=InspirationSource.upload,
            image_url="https://x.com/img.jpg",
            created_at=_NOW,
            updated_at=_NOW,
        )
        assert resp.original_url is None
        assert resp.title is None
        assert resp.style_label is None
        assert resp.style_confidence is None
        assert resp.style_attributes == {}
        assert resp.pinterest_pin_id is None

    def test_url_inspiration_request_validation(self):
        from canvas.models import UrlInspirationRequest
        req = UrlInspirationRequest(url="https://example.com/img.jpg")
        assert req.url == "https://example.com/img.jpg"
        assert req.title is None

    def test_url_inspiration_request_title_max_length(self):
        from canvas.models import UrlInspirationRequest
        # Should accept up to 500 chars
        req = UrlInspirationRequest(url="https://x.com/i.jpg", title="A" * 500)
        assert len(req.title) == 500

        # Should reject > 500 chars
        with pytest.raises(Exception):
            UrlInspirationRequest(url="https://x.com/i.jpg", title="A" * 501)

    def test_pinterest_sync_request_defaults(self):
        from canvas.models import PinterestSyncRequest
        req = PinterestSyncRequest()
        assert req.pin_ids is None
        assert req.max_pins is None

    def test_pinterest_sync_request_max_pins_validation(self):
        from canvas.models import PinterestSyncRequest
        with pytest.raises(Exception):
            PinterestSyncRequest(max_pins=0)
        with pytest.raises(Exception):
            PinterestSyncRequest(max_pins=201)
        req = PinterestSyncRequest(max_pins=100)
        assert req.max_pins == 100

    def test_delete_response_fields(self):
        from canvas.models import DeleteInspirationResponse
        resp = DeleteInspirationResponse(
            deleted=True, taste_vector_updated=True, remaining_count=5,
        )
        assert resp.deleted is True
        assert resp.remaining_count == 5
        assert resp.inspirations == []  # default empty

    def test_delete_response_with_surviving_inspirations(self):
        from canvas.models import DeleteInspirationResponse, InspirationResponse
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        surviving = InspirationResponse(
            id="surv-1", source="upload", image_url="https://example.com/a.jpg",
            style_label="Casual", style_confidence=0.8, style_attributes={},
            created_at=now, updated_at=now,
        )
        resp = DeleteInspirationResponse(
            deleted=True, taste_vector_updated=False, remaining_count=1,
            inspirations=[surviving],
        )
        assert len(resp.inspirations) == 1
        assert resp.inspirations[0].id == "surv-1"

    def test_attribute_score(self):
        from canvas.models import AttributeScore
        s = AttributeScore(value="Boho", count=8, confidence=0.72)
        assert s.value == "Boho"
        assert s.count == 8
        assert s.confidence == 0.72

    def test_style_elements_response_defaults(self):
        from canvas.models import StyleElementsResponse
        resp = StyleElementsResponse()
        assert resp.suggested_filters == {}
        assert resp.raw_attributes == {}
        assert resp.inspiration_count == 0

    def test_complete_fit_request_defaults(self):
        from canvas.models import CompleteFitFromInspirationRequest
        req = CompleteFitFromInspirationRequest()
        assert req.items_per_category == 4
        assert req.category is None
        assert req.offset == 0
        assert req.limit is None

    def test_complete_fit_request_validation(self):
        from canvas.models import CompleteFitFromInspirationRequest
        with pytest.raises(Exception):
            CompleteFitFromInspirationRequest(items_per_category=0)
        with pytest.raises(Exception):
            CompleteFitFromInspirationRequest(items_per_category=21)
        with pytest.raises(Exception):
            CompleteFitFromInspirationRequest(limit=101)

    def test_similar_products_response(self):
        from canvas.models import SimilarProductsResponse
        resp = SimilarProductsResponse(
            products=[{"product_id": "a"}, {"product_id": "b"}],
            count=2,
            total_available=10,
            offset=0,
            has_more=True,
            inspiration_id="ins-1",
        )
        assert resp.count == 2
        assert resp.total_available == 10
        assert resp.offset == 0
        assert resp.has_more is True

    def test_similar_products_response_last_page(self):
        from canvas.models import SimilarProductsResponse
        resp = SimilarProductsResponse(
            products=[{"product_id": "c"}],
            count=1,
            total_available=5,
            offset=4,
            has_more=False,
            inspiration_id="ins-1",
        )
        assert resp.has_more is False
        assert resp.offset == 4

    def test_complete_fit_response(self):
        from canvas.models import CompleteFitFromInspirationResponse
        resp = CompleteFitFromInspirationResponse(
            matched_product={"product_id": "abc", "name": "Dress"},
            outfit={"recommendations": {}, "status": "ok"},
        )
        assert resp.matched_product["product_id"] == "abc"
        assert resp.outfit["status"] == "ok"

    def test_inspiration_list_response(self):
        from canvas.models import InspirationListResponse, InspirationResponse, InspirationSource
        items = [
            InspirationResponse(
                id="1", source=InspirationSource.url,
                image_url="https://x.com/1.jpg",
                created_at=_NOW, updated_at=_NOW,
            ),
        ]
        resp = InspirationListResponse(inspirations=items, count=1)
        assert resp.count == 1
        assert len(resp.inspirations) == 1


# =========================================================================
# 2. Image processor tests
# =========================================================================

class TestAggregateAttributes:
    """Tests for _aggregate_attributes — pure logic, no mocks needed."""

    def test_empty_rows(self):
        from canvas.image_processor import _aggregate_attributes
        assert _aggregate_attributes([]) == {}

    def test_single_row_scalar_attributes(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [_make_attr_row()]
        dist = _aggregate_attributes(rows)

        # pattern is scalar -> "floral" should be 100%
        assert "pattern" in dist
        assert dist["pattern"]["floral"] == 1.0

        # color_family is scalar -> "Neutrals" should be 100%
        assert dist["color_family"]["Neutrals"] == 1.0

    def test_single_row_array_attributes(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [_make_attr_row(style_tags=["Boho", "Romantic"])]
        dist = _aggregate_attributes(rows)

        # 2 tags -> each gets 0.5
        assert "style_tags" in dist
        assert dist["style_tags"]["Boho"] == 0.5
        assert dist["style_tags"]["Romantic"] == 0.5

    def test_multiple_rows_aggregation(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [
            _make_attr_row(pattern="floral"),
            _make_attr_row(pattern="floral"),
            _make_attr_row(pattern="solid"),
        ]
        dist = _aggregate_attributes(rows)

        # floral=2, solid=1 -> floral=2/3, solid=1/3
        assert abs(dist["pattern"]["floral"] - 2 / 3) < 1e-6
        assert abs(dist["pattern"]["solid"] - 1 / 3) < 1e-6

    def test_none_values_skipped(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [_make_attr_row(pattern=None, color_family=None)]
        dist = _aggregate_attributes(rows)
        assert "pattern" not in dist
        assert "color_family" not in dist

    def test_na_values_skipped(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [_make_attr_row(pattern="N/A", formality="null")]
        dist = _aggregate_attributes(rows)
        assert "pattern" not in dist
        assert "formality" not in dist

    def test_mixed_array_with_none_items(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [_make_attr_row(style_tags=["Boho", None, "", "Classic"])]
        dist = _aggregate_attributes(rows)
        # None and "" should be skipped -> Boho=0.5, Classic=0.5
        assert len(dist["style_tags"]) == 2
        assert dist["style_tags"]["Boho"] == 0.5
        assert dist["style_tags"]["Classic"] == 0.5

    def test_distributions_sum_to_one(self):
        from canvas.image_processor import _aggregate_attributes
        rows = [
            _make_attr_row(
                style_tags=["Boho", "Romantic", "Classic"],
                pattern="floral",
                occasions=["casual", "vacation", "evening"],
            ),
            _make_attr_row(
                style_tags=["Boho", "Minimal"],
                pattern="solid",
                occasions=["casual", "office"],
            ),
        ]
        dist = _aggregate_attributes(rows)
        for key, d in dist.items():
            total = sum(d.values())
            assert abs(total - 1.0) < 1e-6, f"{key} distribution sums to {total}"


class TestAttrToFeedParam:
    """Verify the mapping covers all attribute keys."""

    def test_all_keys_mapped(self):
        from canvas.image_processor import _ATTR_TO_FEED_PARAM, _STYLE_ATTR_KEYS
        for key in _STYLE_ATTR_KEYS:
            assert key in _ATTR_TO_FEED_PARAM, f"Missing mapping for {key}"

    def test_feed_param_names_correct(self):
        from canvas.image_processor import _ATTR_TO_FEED_PARAM
        expected = {
            "style_tags": "include_style_tags",
            "pattern": "include_patterns",
            "color_family": "include_color_family",
            "formality": "include_formality",
            "occasions": "include_occasions",
            "silhouette": "include_silhouette",
            "fit_type": "include_fit",
            "sleeve_type": "include_sleeves",
            "neckline": "include_neckline",
        }
        assert _ATTR_TO_FEED_PARAM == expected


class TestClassifyStyle:
    """Tests for classify_style with mocked Supabase."""

    def test_returns_dominant_style_tag(self):
        from canvas.image_processor import classify_style

        mock_sb = _mock_supabase_chain()
        # RPC returns 2 products
        rpc_result = MagicMock()
        rpc_result.data = [
            _make_product_row(product_id="p1"),
            _make_product_row(product_id="p2"),
        ]
        mock_sb.rpc.return_value.execute.return_value = rpc_result

        # product_attributes query returns rows
        attr_result = MagicMock()
        attr_result.data = [
            _make_attr_row(sku_id="p1", style_tags=["Boho", "Romantic"]),
            _make_attr_row(sku_id="p2", style_tags=["Boho", "Classic"]),
        ]
        builder = mock_sb.table.return_value
        builder.execute.return_value = attr_result

        embedding = _make_embedding()
        label, confidence, distributions = classify_style(embedding, mock_sb, nearest_k=2)

        assert label == "Boho", f"Expected 'Boho', got '{label}'"
        assert confidence > 0
        assert "style_tags" in distributions

    def test_rpc_failure_returns_none(self):
        from canvas.image_processor import classify_style

        mock_sb = _mock_supabase_chain()
        mock_sb.rpc.side_effect = Exception("RPC timeout")

        embedding = _make_embedding()
        label, confidence, distributions = classify_style(embedding, mock_sb)

        assert label is None
        assert confidence == 0.0
        assert distributions == {}

    def test_empty_products_returns_none(self):
        from canvas.image_processor import classify_style

        mock_sb = _mock_supabase_chain()
        rpc_result = MagicMock()
        rpc_result.data = []
        mock_sb.rpc.return_value.execute.return_value = rpc_result

        embedding = _make_embedding()
        label, confidence, distributions = classify_style(embedding, mock_sb)

        assert label is None
        assert confidence == 0.0

    def test_no_attributes_returns_none(self):
        from canvas.image_processor import classify_style

        mock_sb = _mock_supabase_chain()
        rpc_result = MagicMock()
        rpc_result.data = [_make_product_row()]
        mock_sb.rpc.return_value.execute.return_value = rpc_result

        # product_attributes returns empty
        attr_result = MagicMock()
        attr_result.data = []
        mock_sb.table.return_value.execute.return_value = attr_result

        embedding = _make_embedding()
        label, confidence, distributions = classify_style(embedding, mock_sb)

        assert label is None


class TestEncoding:
    """Tests for encode_from_bytes and encode_from_url with mocked FashionCLIP."""

    @patch("canvas.image_processor.get_pinterest_style_extractor")
    def test_encode_from_bytes(self, mock_get_extractor):
        from canvas.image_processor import encode_from_bytes

        fake_emb = _make_embedding(seed=99)
        mock_extractor = MagicMock()
        mock_extractor.encode_image.return_value = fake_emb
        mock_get_extractor.return_value = mock_extractor

        # Minimal valid JPEG-like bytes (PIL won't care with mock)
        # We need real image bytes for PIL.Image.open — use a tiny PNG
        import io
        from PIL import Image
        img = Image.new("RGB", (2, 2), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = encode_from_bytes(png_bytes)
        assert result.shape == (512,)
        mock_extractor.encode_image.assert_called_once()

    @patch("canvas.image_processor.requests.get")
    @patch("canvas.image_processor.get_pinterest_style_extractor")
    def test_encode_from_url(self, mock_get_extractor, mock_requests_get):
        from canvas.image_processor import encode_from_url

        # Build a tiny PNG
        import io
        from PIL import Image
        img = Image.new("RGB", (2, 2), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="PNG")

        mock_response = MagicMock()
        mock_response.content = buf.getvalue()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_requests_get.return_value = mock_response

        fake_emb = _make_embedding(seed=77)
        mock_extractor = MagicMock()
        mock_extractor.encode_image.return_value = fake_emb
        mock_get_extractor.return_value = mock_extractor

        result = encode_from_url("https://example.com/test.jpg")
        assert result.shape == (512,)
        mock_requests_get.assert_called_once()

    @patch("canvas.image_processor.requests.get")
    def test_encode_from_url_http_error(self, mock_requests_get):
        from canvas.image_processor import encode_from_url
        import requests as req_lib

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = req_lib.HTTPError("404")
        mock_requests_get.return_value = mock_response

        with pytest.raises(req_lib.HTTPError):
            encode_from_url("https://example.com/missing.jpg")


# =========================================================================
# 3. Service tests
# =========================================================================

class TestCanvasServiceList:
    """CanvasService.list_inspirations"""

    def test_list_empty(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        result = svc.list_inspirations("user-123", mock_sb)
        assert result == []

    def test_list_returns_responses(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        rows = [_make_db_row(), _make_db_row()]
        exec_result = MagicMock()
        exec_result.data = rows
        mock_sb.table.return_value.execute.return_value = exec_result

        result = svc.list_inspirations("user-123", mock_sb)
        assert len(result) == 2
        assert result[0].source.value == "url"


class TestCanvasServiceAddUrl:
    """CanvasService.add_inspiration_url"""

    @patch("canvas.service.classify_style")
    @patch("canvas.service.encode_from_url")
    def test_add_url_success(self, mock_encode, mock_classify):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        mock_encode.return_value = _make_embedding()
        mock_classify.return_value = ("Boho", 0.6, {"style_tags": {"Boho": 0.6}})

        insert_row = _make_db_row(source="url")

        # The service calls .execute() multiple times:
        # 1. _check_quota (select count)
        # 2. _insert_inspiration (insert)
        # 3. recompute_taste_vector (select embeddings)
        # 4. recompute_taste_vector (rpc)
        # Use side_effect to return different results per call.
        quota_result = MagicMock()
        quota_result.data = []
        quota_result.count = 3

        insert_result = MagicMock()
        insert_result.data = [insert_row]

        # For recompute: select embeddings returns the same row's embedding
        emb_result = MagicMock()
        emb_result.data = [{"embedding": insert_row["embedding"]}]

        builder = mock_sb.table.return_value
        builder.execute.side_effect = [quota_result, insert_result, emb_result]

        # RPC for recompute
        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = MagicMock()
        mock_sb.rpc.return_value = rpc_exec

        result = svc.add_inspiration_url(
            user_id="user-123",
            url="https://example.com/dress.jpg",
            title="Cute dress",
            supabase=mock_sb,
        )
        assert result.source.value == "url"
        mock_encode.assert_called_once_with("https://example.com/dress.jpg")
        mock_classify.assert_called_once()

    @patch("canvas.service.encode_from_url")
    def test_add_url_quota_exceeded(self, mock_encode):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # Quota check: at limit
        quota_result = MagicMock()
        quota_result.count = 50  # default max
        mock_sb.table.return_value.execute.return_value = quota_result

        with pytest.raises(ValueError, match="Inspiration limit reached"):
            svc.add_inspiration_url("user-123", "https://x.com/i.jpg", None, mock_sb)

        mock_encode.assert_not_called()


class TestCanvasServiceRemove:
    """CanvasService.remove_inspiration"""

    def test_remove_not_found_is_idempotent(self):
        """DELETE for an already-gone item returns deleted=True (idempotent)."""
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # First execute: row lookup returns empty (already deleted)
        lookup_result = MagicMock()
        lookup_result.data = []
        # Second execute: list_inspirations returns empty
        list_result = MagicMock()
        list_result.data = []
        mock_sb.table.return_value.execute.side_effect = [
            lookup_result,
            list_result,
        ]

        result = svc.remove_inspiration("user-123", "nonexistent-id", mock_sb)
        assert result.deleted is True
        assert result.remaining_count == 0
        assert result.inspirations == []

    def test_remove_upload_triggers_storage_delete(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        bucket_url = "https://proj.supabase.co/storage/v1/object/public/inspirations/user-123/abc.jpg"
        row = {"id": "ins-1", "source": "upload", "image_url": bucket_url}
        lookup_result = MagicMock()
        lookup_result.data = [row]

        # After delete, list_inspirations returns empty
        list_result = MagicMock()
        list_result.data = []

        # execute calls: lookup, delete, list_inspirations
        mock_sb.table.return_value.execute.side_effect = [
            lookup_result,   # row lookup
            MagicMock(),     # delete execute
            list_result,     # list_inspirations
        ]

        result = svc.remove_inspiration("user-123", "ins-1", mock_sb)
        assert result.deleted is True
        assert result.remaining_count == 0
        # Verify storage.from_("inspirations").remove() was called
        mock_sb.storage.from_.assert_called_with("inspirations")

    def test_remove_url_no_storage_delete(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        row = {"id": "ins-2", "source": "url", "image_url": "https://example.com/img.jpg"}
        lookup_result = MagicMock()
        lookup_result.data = [row]

        list_result = MagicMock()
        list_result.data = []

        mock_sb.table.return_value.execute.side_effect = [
            lookup_result,   # row lookup
            MagicMock(),     # delete execute
            list_result,     # list_inspirations
        ]

        result = svc.remove_inspiration("user-123", "ins-2", mock_sb)
        assert result.deleted is True
        # storage.from_ should NOT be called for URL source
        mock_sb.storage.from_.assert_not_called()

    def test_remove_returns_surviving_inspirations(self):
        """DELETE response includes authoritative surviving list."""
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        row = {"id": "ins-del", "source": "url", "image_url": "https://example.com/del.jpg"}
        lookup_result = MagicMock()
        lookup_result.data = [row]

        # One surviving inspiration after deletion
        surviving_row = _make_db_row(
            id="ins-surv", source="upload",
            image_url="https://example.com/surv.jpg",
        )
        list_result = MagicMock()
        list_result.data = [surviving_row]

        mock_sb.table.return_value.execute.side_effect = [
            lookup_result,
            MagicMock(),
            list_result,
        ]

        result = svc.remove_inspiration("user-123", "ins-del", mock_sb)
        assert result.deleted is True
        assert result.remaining_count == 1
        assert len(result.inspirations) == 1
        assert result.inspirations[0].id == "ins-surv"


class TestCanvasServiceTasteVector:
    """CanvasService.recompute_taste_vector"""

    def test_no_inspirations_returns_false(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # Select returns empty
        exec_result = MagicMock()
        exec_result.data = []
        mock_sb.table.return_value.execute.return_value = exec_result

        assert svc.recompute_taste_vector("user-123", mock_sb) is False

    def test_single_embedding_normalised(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        emb_str = _make_embedding_str(seed=10)
        exec_result = MagicMock()
        exec_result.data = [{"embedding": emb_str}]
        mock_sb.table.return_value.execute.return_value = exec_result

        # RPC call should succeed
        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = MagicMock()
        mock_sb.rpc.return_value = rpc_exec

        result = svc.recompute_taste_vector("user-123", mock_sb)
        assert result is True
        mock_sb.rpc.assert_called_once_with(
            "update_user_taste_vector",
            {
                "p_user_id": "user-123",
                "p_taste_vector": pytest.approx(
                    _make_embedding(seed=10).astype("float64").tolist(), abs=1e-4,
                ),
            },
        )

    def test_multiple_embeddings_averaged(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        emb1 = _make_embedding(seed=1)
        emb2 = _make_embedding(seed=2)
        exec_result = MagicMock()
        exec_result.data = [
            {"embedding": "[" + ",".join(f"{v:.8f}" for v in emb1) + "]"},
            {"embedding": "[" + ",".join(f"{v:.8f}" for v in emb2) + "]"},
        ]
        mock_sb.table.return_value.execute.return_value = exec_result

        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = MagicMock()
        mock_sb.rpc.return_value = rpc_exec

        result = svc.recompute_taste_vector("user-123", mock_sb)
        assert result is True

        # Check that the taste vector is the normalised mean
        expected_mean = np.mean(np.stack([emb1, emb2]), axis=0)
        expected_mean = expected_mean / np.linalg.norm(expected_mean)

        call_args = mock_sb.rpc.call_args
        actual_vec = call_args[0][1]["p_taste_vector"]
        assert len(actual_vec) == 512
        np.testing.assert_allclose(actual_vec, expected_mean.tolist(), atol=1e-4)

    def test_rpc_failure_returns_false(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        exec_result = MagicMock()
        exec_result.data = [{"embedding": _make_embedding_str()}]
        mock_sb.table.return_value.execute.return_value = exec_result

        mock_sb.rpc.return_value.execute.side_effect = Exception("DB error")

        result = svc.recompute_taste_vector("user-123", mock_sb)
        assert result is False

    def test_invalid_embedding_shape_skipped(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # One valid, one invalid (wrong dimension)
        valid_emb = _make_embedding_str(seed=5)
        invalid_emb = "[" + ",".join("0.1" for _ in range(256)) + "]"  # 256-dim
        exec_result = MagicMock()
        exec_result.data = [
            {"embedding": valid_emb},
            {"embedding": invalid_emb},
        ]
        mock_sb.table.return_value.execute.return_value = exec_result

        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = MagicMock()
        mock_sb.rpc.return_value = rpc_exec

        result = svc.recompute_taste_vector("user-123", mock_sb)
        assert result is True
        # Should only use the valid embedding
        call_args = mock_sb.rpc.call_args
        actual_vec = call_args[0][1]["p_taste_vector"]
        assert len(actual_vec) == 512


class TestCanvasServiceStyleElements:
    """CanvasService.get_style_elements"""

    def test_no_inspirations(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        exec_result = MagicMock()
        exec_result.data = []
        mock_sb.table.return_value.execute.return_value = exec_result

        result = svc.get_style_elements("user-123", mock_sb)
        assert result.inspiration_count == 0
        assert result.suggested_filters == {}
        assert result.raw_attributes == {}

    def test_single_inspiration_style_elements(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        attrs = {
            "style_tags": {"Boho": 0.5, "Romantic": 0.3},
            "pattern": {"floral": 0.8},
            "color_family": {"Neutrals": 0.7},
        }
        exec_result = MagicMock()
        exec_result.data = [{"style_attributes": attrs}]
        mock_sb.table.return_value.execute.return_value = exec_result

        result = svc.get_style_elements("user-123", mock_sb, min_count=1, min_confidence=0.2)
        assert result.inspiration_count == 1
        # All values should appear in raw_attributes
        assert "style_tags" in result.raw_attributes
        assert "pattern" in result.raw_attributes

    def test_suggested_filters_use_feed_param_names(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        attrs = {
            "style_tags": {"Boho": 0.7, "Romantic": 0.3},
            "pattern": {"floral": 1.0},
        }
        exec_result = MagicMock()
        exec_result.data = [
            {"style_attributes": attrs},
            {"style_attributes": attrs},
            {"style_attributes": attrs},
        ]
        mock_sb.table.return_value.execute.return_value = exec_result

        result = svc.get_style_elements("user-123", mock_sb, min_count=2, min_confidence=0.25)

        # Keys should be feed param names
        for key in result.suggested_filters:
            assert key.startswith("include_"), f"Unexpected key: {key}"

    def test_threshold_filters_low_confidence(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # One inspiration with low-confidence attributes
        attrs = {
            "style_tags": {"Boho": 0.1, "Romantic": 0.05, "Classic": 0.85},
        }
        exec_result = MagicMock()
        exec_result.data = [{"style_attributes": attrs}]
        mock_sb.table.return_value.execute.return_value = exec_result

        # With min_confidence=0.5, only Classic should pass
        result = svc.get_style_elements("user-123", mock_sb, min_count=999, min_confidence=0.5)

        filters = result.suggested_filters.get("include_style_tags", [])
        assert "Classic" in filters
        assert "Boho" not in filters
        assert "Romantic" not in filters


class TestCanvasServiceClosestProduct:
    """CanvasService.find_closest_product"""

    def test_inspiration_not_found(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        exec_result = MagicMock()
        exec_result.data = []
        mock_sb.table.return_value.execute.return_value = exec_result

        result = svc.find_closest_product("ins-999", "user-123", mock_sb)
        assert result is None

    def test_closest_product_found(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        # Step 1: fetch embedding from user_inspirations
        emb_str = _make_embedding_str(seed=42)
        select_result = MagicMock()
        select_result.data = [{"embedding": emb_str}]

        # Step 2: canvas_similar_search RPC returns sku_ids
        nn_result = MagicMock()
        nn_result.data = [{"sku_id": "prod-abc", "similarity": 0.87}]

        # Step 3: batch-fetch product details from products table
        product_result = MagicMock()
        product_result.data = [{
            "id": "prod-abc",
            "name": "Boho Floral Maxi Dress",
            "brand": "TestBrand",
            "category": "dresses",
            "broad_category": "dresses",
            "colors": ["beige"],
            "materials": ["cotton"],
            "price": 59.99,
            "original_price": 79.99,
            "fit": "regular",
            "length": "maxi",
            "sleeve": "short",
            "neckline": "v-neck",
            "style_tags": ["Boho"],
            "primary_image_url": "https://example.com/product.jpg",
            "hero_image_url": None,
            "in_stock": True,
        }]

        # Wire up: first table call = embedding lookup, second = product fetch
        mock_sb.table.return_value.execute.side_effect = [
            select_result,   # user_inspirations embedding lookup
            product_result,  # products batch fetch
        ]

        # RPC returns canvas_similar_search results
        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = nn_result
        mock_sb.rpc.return_value = rpc_exec

        result = svc.find_closest_product("ins-1", "user-123", mock_sb)
        assert result is not None
        assert result["product_id"] == "prod-abc"
        assert result["similarity"] == 0.87

    def test_rpc_failure_returns_none(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        select_result = MagicMock()
        select_result.data = [{"embedding": _make_embedding_str()}]
        mock_sb.table.return_value.execute.return_value = select_result

        mock_sb.rpc.return_value.execute.side_effect = Exception("timeout")

        result = svc.find_closest_product("ins-1", "user-123", mock_sb)
        assert result is None


class TestCanvasServiceSimilarProducts:
    """CanvasService.find_similar_products with offset pagination."""

    def _setup_mock(self, n_products: int):
        """Create a mock with n_products returned by the HNSW RPC."""
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()

        emb_str = _make_embedding_str(seed=7)

        # Step 1: embedding lookup
        select_result = MagicMock()
        select_result.data = [{"embedding": emb_str}]

        # Step 2: HNSW RPC returns n sku_ids (distinct brands to survive dedup)
        brands = [
            "Brand_A", "Brand_B", "Brand_C", "Brand_D", "Brand_E",
            "Brand_F", "Brand_G", "Brand_H", "Brand_I", "Brand_J",
            "Brand_K", "Brand_L", "Brand_M", "Brand_N", "Brand_O",
        ]
        nn_rows = [
            {"sku_id": f"prod-{i:03d}", "similarity": round(0.90 - i * 0.005, 4)}
            for i in range(n_products)
        ]
        nn_result = MagicMock()
        nn_result.data = nn_rows

        # Step 3: products table returns matching rows
        product_rows = [
            {
                "id": f"prod-{i:03d}",
                "name": f"Product {i}",
                "brand": brands[i % len(brands)],
                "category": "tops",
                "broad_category": "tops",
                "colors": ["black"],
                "materials": ["cotton"],
                "price": 29.99 + i,
                "original_price": 39.99 + i,
                "fit": "regular",
                "length": "regular",
                "sleeve": "short",
                "neckline": "crew",
                "style_tags": ["Casual"],
                "primary_image_url": f"https://example.com/prod-{i:03d}.jpg",
                "hero_image_url": None,
                "in_stock": True,
            }
            for i in range(n_products)
        ]
        product_result = MagicMock()
        product_result.data = product_rows

        mock_sb.table.return_value.execute.side_effect = [
            select_result,   # embedding lookup
            product_result,  # products batch fetch
        ]

        rpc_exec = MagicMock()
        rpc_exec.execute.return_value = nn_result
        mock_sb.rpc.return_value = rpc_exec

        return svc, mock_sb

    def test_returns_tuple(self):
        svc, mock_sb = self._setup_mock(20)
        result = svc.find_similar_products("ins-1", "user-1", mock_sb, count=6)
        assert isinstance(result, tuple)
        products, total = result
        assert isinstance(products, list)
        assert isinstance(total, int)

    def test_default_offset_zero(self):
        svc, mock_sb = self._setup_mock(20)
        products, total = svc.find_similar_products(
            "ins-1", "user-1", mock_sb, count=6,
        )
        assert len(products) == 6
        assert total >= 6
        # First product should be the highest similarity
        assert products[0]["name"] == "Product 0"

    def test_offset_skips_items(self):
        svc, mock_sb = self._setup_mock(20)
        page0, total0 = svc.find_similar_products(
            "ins-1", "user-1", mock_sb, count=6, offset=0,
        )
        # Re-setup mock for second call (side_effect consumed)
        svc2, mock_sb2 = self._setup_mock(20)
        page1, total1 = svc2.find_similar_products(
            "ins-1", "user-1", mock_sb2, count=6, offset=6,
        )
        # Pages should not overlap
        ids_0 = {p["product_id"] for p in page0}
        ids_1 = {p["product_id"] for p in page1}
        assert ids_0.isdisjoint(ids_1), "Pages overlap"
        # Totals should be consistent
        assert total0 == total1

    def test_offset_beyond_total_returns_empty(self):
        svc, mock_sb = self._setup_mock(5)
        products, total = svc.find_similar_products(
            "ins-1", "user-1", mock_sb, count=6, offset=100,
        )
        assert products == []
        assert total <= 5

    def test_embedding_not_found_returns_empty(self):
        from canvas.service import CanvasService
        svc = CanvasService()
        mock_sb = _mock_supabase_chain()
        select_result = MagicMock()
        select_result.data = []
        mock_sb.table.return_value.execute.return_value = select_result

        products, total = svc.find_similar_products("ins-1", "user-1", mock_sb)
        assert products == []
        assert total == 0


class TestCanvasServiceHelpers:
    """Private helper tests."""

    def test_guess_content_type(self):
        from canvas.service import _guess_content_type
        assert _guess_content_type("jpg") == "image/jpeg"
        assert _guess_content_type("JPEG") == "image/jpeg"
        assert _guess_content_type("png") == "image/png"
        assert _guess_content_type("webp") == "image/webp"
        assert _guess_content_type("gif") == "image/gif"
        assert _guess_content_type("bmp") == "image/jpeg"  # fallback

    def test_row_to_response(self):
        from canvas.service import _row_to_response
        row = _make_db_row(source="pinterest", pinterest_pin_id="pin-123")
        resp = _row_to_response(row)
        assert resp.source.value == "pinterest"
        assert resp.pinterest_pin_id == "pin-123"
        assert isinstance(resp.style_attributes, dict)

    def test_row_to_response_null_attributes(self):
        from canvas.service import _row_to_response
        row = _make_db_row(style_attributes=None)
        resp = _row_to_response(row)
        assert resp.style_attributes == {}

    def test_singleton(self):
        from canvas.service import get_canvas_service
        s1 = get_canvas_service()
        s2 = get_canvas_service()
        assert s1 is s2


# =========================================================================
# 4. Route wiring tests
# =========================================================================

class TestRouteWiring:
    """Verify the router has the expected endpoints."""

    def test_router_prefix(self):
        from canvas.routes import router
        assert router.prefix == "/api/canvas"

    def test_route_count(self):
        from canvas.routes import router
        assert len(router.routes) == 8

    def test_expected_paths_and_methods(self):
        from canvas.routes import router
        route_map = {}
        for route in router.routes:
            methods = getattr(route, "methods", set())
            path = getattr(route, "path", "")
            for m in methods:
                route_map[f"{m} {path}"] = True

        expected = [
            "GET /api/canvas/inspirations",
            "POST /api/canvas/inspirations/upload",
            "POST /api/canvas/inspirations/url",
            "POST /api/canvas/inspirations/pinterest",
            "DELETE /api/canvas/inspirations/{inspiration_id}",
            "GET /api/canvas/inspirations/{inspiration_id}/similar",
            "GET /api/canvas/style-elements",
            "POST /api/canvas/inspirations/{inspiration_id}/complete-fit",
        ]
        for e in expected:
            assert e in route_map, f"Missing route: {e}"

    def test_all_routes_require_auth(self):
        """All canvas routes should have require_auth as a dependency."""
        from canvas.routes import router
        from core.auth import require_auth

        for route in router.routes:
            deps = getattr(route, "dependencies", []) or []
            endpoint = getattr(route, "endpoint", None)
            if endpoint is None:
                continue

            # Check if require_auth is in the function's dependencies
            # FastAPI stores deps in the endpoint's __wrapped__ or via params
            import inspect
            sig = inspect.signature(endpoint)
            has_auth = False
            for param in sig.parameters.values():
                if param.default is not inspect.Parameter.empty:
                    # Check if the Depends wraps require_auth
                    dep = param.default
                    if hasattr(dep, "dependency") and dep.dependency is require_auth:
                        has_auth = True
                        break
            assert has_auth, f"Route {endpoint.__name__} missing require_auth"

    def test_validate_image_upload_rejects_bad_type(self):
        from canvas.routes import _validate_image_upload
        from fastapi import HTTPException, UploadFile

        file = MagicMock(spec=UploadFile)
        file.content_type = "application/pdf"
        file.size = 1000

        with pytest.raises(HTTPException) as exc_info:
            _validate_image_upload(file)
        assert exc_info.value.status_code == 415

    def test_validate_image_upload_rejects_large_file(self):
        from canvas.routes import _validate_image_upload
        from fastapi import HTTPException, UploadFile

        file = MagicMock(spec=UploadFile)
        file.content_type = "image/jpeg"
        file.size = 11 * 1024 * 1024  # 11 MB

        with pytest.raises(HTTPException) as exc_info:
            _validate_image_upload(file)
        assert exc_info.value.status_code == 413

    def test_validate_image_upload_accepts_valid(self):
        from canvas.routes import _validate_image_upload
        from fastapi import UploadFile

        file = MagicMock(spec=UploadFile)
        file.content_type = "image/png"
        file.size = 500_000

        # Should not raise
        _validate_image_upload(file)

    def test_validate_image_upload_allows_unknown_type(self):
        """If content_type is None (unknown), don't reject."""
        from canvas.routes import _validate_image_upload
        from fastapi import UploadFile

        file = MagicMock(spec=UploadFile)
        file.content_type = None
        file.size = 500_000

        _validate_image_upload(file)


# =========================================================================
# 5. App integration test — router is registered
# =========================================================================

class TestAppRegistration:
    """Verify the canvas router is mounted in the FastAPI app."""

    def test_canvas_routes_in_app(self):
        from api.app import app

        paths = set()
        for route in app.routes:
            path = getattr(route, "path", "")
            if "/api/canvas" in path:
                paths.add(path)

        assert "/api/canvas/inspirations" in paths, (
            f"Canvas routes not found in app. Found paths: {sorted(paths)}"
        )

    def test_canvas_openapi_tag(self):
        """Canvas endpoints should appear under the 'Canvas' tag in OpenAPI."""
        from api.app import app

        schema = app.openapi()
        tags_used = set()
        for path_info in schema.get("paths", {}).values():
            for method_info in path_info.values():
                for tag in method_info.get("tags", []):
                    tags_used.add(tag)

        assert "Canvas" in tags_used
