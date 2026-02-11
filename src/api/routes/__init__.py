"""
Route modules for the API.

Each module exports a FastAPI APIRouter with endpoints
for a specific domain/feature.
"""

from api.routes import health
from api.routes import pinterest
from api.routes import women
from api.routes import unified

__all__ = ["health", "pinterest", "women", "unified"]
