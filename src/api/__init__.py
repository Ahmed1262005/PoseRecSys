"""
API module for FastAPI routes.

This module organizes all API endpoints by domain/feature.
Each route module defines a FastAPI APIRouter that can be
mounted on the main application.
"""

from api.routes import health, women, unified

__all__ = ["health", "women", "unified"]
