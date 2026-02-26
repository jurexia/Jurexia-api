"""
Rate Limiter for Jurexia API
Sliding window rate limiter using in-memory dict (no Redis dependency).
Tiers: Free (10 req/min), Pro (30 req/min), Platinum (60 req/min)
"""
import time
from collections import defaultdict
from typing import Optional, Tuple
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

# Rate limits per subscription tier (requests per minute)
RATE_LIMITS = {
    "gratuito": 10,
    "pro_monthly": 30,
    "pro_annual": 30,
    "platinum_monthly": 60,
    "platinum_annual": 60,
    "ultra_secretarios": 40,
}
DEFAULT_RATE_LIMIT = 10  # Unknown/missing tier

# Paths that are rate-limited (only expensive endpoints)
RATE_LIMITED_PATHS = {
    "/chat",
    "/chat-sentencia",
    "/redactor-sentencia-chat",
    "/draft-sentencia-stream",
    "/analyze-expediente",
    "/generate-amparo-salud",
}

# Paths that are exempt from rate limiting
EXEMPT_PATHS = {
    "/health",
    "/api/wake",
    "/cache-status",
    "/quota/status",
}


# ═══════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW COUNTER
# ═══════════════════════════════════════════════════════════════════════════

class SlidingWindowCounter:
    """
    In-memory sliding window rate limiter.
    Tracks request timestamps per user and checks against window limits.
    
    Note: This resets on server restart. For production at scale, use Redis.
    For Jurexia's current traffic (<100 concurrent users), in-memory is fine.
    """
    
    def __init__(self):
        # user_key -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)
    
    def is_allowed(self, user_key: str, max_requests: int, window_seconds: int = 60) -> Tuple[bool, int, int]:
        """
        Check if a request is allowed for the given user.
        
        Returns:
            (allowed, remaining, retry_after_seconds)
        """
        now = time.time()
        window_start = now - window_seconds
        
        # Clean old entries
        self._requests[user_key] = [
            t for t in self._requests[user_key] if t > window_start
        ]
        
        current_count = len(self._requests[user_key])
        
        if current_count >= max_requests:
            # Calculate retry-after: time until the oldest request exits the window
            oldest = self._requests[user_key][0] if self._requests[user_key] else now
            retry_after = int(oldest + window_seconds - now) + 1
            return False, 0, max(retry_after, 1)
        
        # Record this request
        self._requests[user_key].append(now)
        remaining = max_requests - current_count - 1
        return True, remaining, 0
    
    def cleanup(self, max_age_seconds: int = 300):
        """Remove stale entries older than max_age_seconds."""
        cutoff = time.time() - max_age_seconds
        stale_keys = [
            key for key, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] < cutoff
        ]
        for key in stale_keys:
            del self._requests[key]


# Global instance
_rate_limiter = SlidingWindowCounter()


# ═══════════════════════════════════════════════════════════════════════════
# MIDDLEWARE
# ═══════════════════════════════════════════════════════════════════════════

def _extract_user_info(request: Request) -> Tuple[Optional[str], str]:
    """
    Extract user identifier and subscription tier from the request.
    Uses user_id from JSON body (set by frontend) or falls back to IP.
    
    Returns:
        (user_key, subscription_tier)
    """
    # Try to get user info from query params or headers
    # The /chat endpoint sends user_id in the JSON body, but we can't
    # read the body in middleware without consuming it. Use IP + any
    # available header as the rate limiting key.
    
    # Check for user email in a custom header (set by frontend)
    user_email = request.headers.get("X-User-Email")
    if user_email:
        return user_email.lower().strip(), "unknown"
    
    # Fall back to client IP
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    else:
        client_ip = request.client.host if request.client else "unknown"
    
    return f"ip:{client_ip}", "gratuito"


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware that enforces per-user rate limits on expensive endpoints.
    """
    
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        
        # Skip non-rate-limited paths
        if path not in RATE_LIMITED_PATHS:
            return await call_next(request)
        
        # Skip OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Extract user info
        user_key, tier = _extract_user_info(request)
        
        # Get rate limit for this tier
        max_requests = RATE_LIMITS.get(tier, DEFAULT_RATE_LIMIT)
        
        # Check rate limit
        allowed, remaining, retry_after = _rate_limiter.is_allowed(user_key, max_requests)
        
        if not allowed:
            print(f"⛔ RATE LIMIT: {user_key} exceeded {max_requests} req/min on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Demasiadas solicitudes. Por favor espera antes de enviar otra consulta.",
                    "retry_after": retry_after,
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(max_requests),
                    "X-RateLimit-Remaining": "0",
                },
            )
        
        # Process request and add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# ═══════════════════════════════════════════════════════════════════════════
# CLEANUP (call periodically to prevent memory leaks)
# ═══════════════════════════════════════════════════════════════════════════

def cleanup_rate_limiter():
    """Remove stale rate limiter entries. Call from a background task."""
    _rate_limiter.cleanup()
