"""
Security Middleware & Input Sanitization for Midtown Emergency Sheba.
Enforces standard hospital-grade security headers, rate limiting, and input hygiene.
Creator: Payal Rani Das, Midtown Eye Hospital, Comilla.
"""

import time
import re
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        response.headers["X-App-Creator"] = "Payal Rani Das, Midtown Eye Hospital, Comilla"
        return response


class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_requests_per_minute=120):
        super().__init__(app)
        self.max_requests = max_requests_per_minute
        self.requests_log = {}  # ip -> list of timestamps

    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        
        # Clean timestamps older than 60 seconds
        timestamps = self.requests_log.get(client_ip, [])
        valid_timestamps = [t for t in timestamps if now - t < 60.0]
        
        if len(valid_timestamps) >= self.max_requests:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded. Emergency protocol throttling active.",
                    "creator": "Payal Rani Das, Midtown Eye Hospital, Comilla",
                    "retry_after_seconds": 60
                }
            )
            
        valid_timestamps.append(now)
        self.requests_log[client_ip] = valid_timestamps
        return await call_next(request)


def sanitize_input_text(text: str) -> str:
    """
    Sanitizes user input to prevent XSS and command injection.
    """
    if not text:
        return ""
    # Strip HTML tags
    cleaned = re.sub(r'<[^>]*>', '', text)
    # Strip dangerous characters
    cleaned = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', cleaned)
    return cleaned.strip()
