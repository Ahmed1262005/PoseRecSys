"""
Supabase JWT Authentication Module.

This module provides JWT token verification for Supabase Auth integration.
All authenticated endpoints should use the `require_auth` dependency.

Usage:
    from core.auth import require_auth, SupabaseUser
    
    @router.post("/protected")
    async def protected_endpoint(user: SupabaseUser = Depends(require_auth)):
        user_id = user.id  # Verified user ID from JWT
"""

from dataclasses import dataclass
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from config.settings import get_settings


# Security scheme for OpenAPI docs
security = HTTPBearer(
    scheme_name="Supabase JWT",
    description="JWT token from Supabase Auth. Get it after login via Supabase client.",
    auto_error=False,  # Don't auto-raise, we handle it ourselves
)


@dataclass
class SupabaseUser:
    """
    Authenticated user from Supabase JWT.
    
    Attributes:
        id: User's UUID (from 'sub' claim)
        email: User's email address
        phone: User's phone number
        role: Postgres role (usually 'authenticated')
        aal: Authentication Assurance Level ('aal1' or 'aal2' for MFA)
        session_id: Current session UUID
        is_anonymous: True if anonymous auth
        app_metadata: App-specific metadata (provider info, etc.)
        user_metadata: User profile metadata (name, avatar, etc.)
    """
    id: str
    email: Optional[str] = None
    phone: Optional[str] = None
    role: str = "authenticated"
    aal: str = "aal1"
    session_id: Optional[str] = None
    is_anonymous: bool = False
    app_metadata: Optional[dict] = None
    user_metadata: Optional[dict] = None


def verify_jwt(token: str) -> dict:
    """
    Verify and decode a Supabase JWT token.
    
    Args:
        token: The JWT token string (without 'Bearer ' prefix)
        
    Returns:
        Decoded JWT payload as dictionary
        
    Raises:
        HTTPException: If token is invalid, expired, or malformed
    """
    settings = get_settings()
    
    try:
        # Decode and verify the token
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
            options={
                "verify_exp": True,
                "verify_aud": True,
                "require": ["sub", "exp", "aud", "role"],
            }
        )
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def extract_user(payload: dict) -> SupabaseUser:
    """
    Extract SupabaseUser from verified JWT payload.
    
    Args:
        payload: Verified JWT payload dictionary
        
    Returns:
        SupabaseUser instance with user details
    """
    return SupabaseUser(
        id=payload["sub"],
        email=payload.get("email"),
        phone=payload.get("phone"),
        role=payload.get("role", "authenticated"),
        aal=payload.get("aal", "aal1"),
        session_id=payload.get("session_id"),
        is_anonymous=payload.get("is_anonymous", False),
        app_metadata=payload.get("app_metadata"),
        user_metadata=payload.get("user_metadata"),
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[SupabaseUser]:
    """
    FastAPI dependency to get the current authenticated user.
    
    Returns None if no valid token is provided (use require_auth for mandatory auth).
    
    Usage:
        @router.get("/maybe-protected")
        async def endpoint(user: Optional[SupabaseUser] = Depends(get_current_user)):
            if user:
                # Authenticated user
            else:
                # Anonymous access
    """
    if not credentials:
        return None
    
    payload = verify_jwt(credentials.credentials)
    return extract_user(payload)


async def require_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> SupabaseUser:
    """
    FastAPI dependency that requires authentication.
    
    Raises 401 if no valid token is provided.
    
    Usage:
        @router.post("/protected")
        async def endpoint(user: SupabaseUser = Depends(require_auth)):
            user_id = user.id  # Guaranteed to be present
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    payload = verify_jwt(credentials.credentials)
    return extract_user(payload)


# Convenience alias
CurrentUser = SupabaseUser
