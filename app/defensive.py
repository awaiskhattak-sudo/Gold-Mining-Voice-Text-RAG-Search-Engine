"""
Defensive Coding & Application Shield Layer
Prevents crashes, injection attacks, denial of service, and unhandled runtime exceptions.
"""

import re
import html
import time
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# Configure structured defensive logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DEFENSE] %(message)s")
logger = logging.getLogger("defensive_shield")


class DefensiveSecurityException(Exception):
    """Custom exception raised when security or defensive checks fail."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class InputSanitizer:
    """Sanitizes user input to prevent XSS, script injection, and control code crashes."""

    # Patterns for dangerous HTML / Script tags
    TAG_PATTERN = re.compile(r"<[^>]*?>")
    # Patterns for invisible / non-printable control characters (excluding standard whitespace)
    CONTROL_CHAR_PATTERN = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")
    # Excessive whitespace or repetition
    REPEAT_PATTERN = re.compile(r"(.)\1{25,}")  # 25+ repeated characters

    @classmethod
    def sanitize_text(cls, text: str) -> str:
        """Sanitizes query text while preserving meaningful punctuation and numbers."""
        if not text or not isinstance(text, str):
            return ""

        # 1. Strip null bytes and non-printable control characters
        cleaned = cls.CONTROL_CHAR_PATTERN.sub("", text)

        # 2. Strip HTML tags and escape HTML entities
        cleaned = cls.TAG_PATTERN.sub("", cleaned)
        cleaned = html.escape(cleaned, quote=False)

        # 3. Collapse multiple whitespace characters into single space
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # 4. Limit overly repetitive characters (e.g., 'aaaaaaaaaa...')
        cleaned = cls.REPEAT_PATTERN.sub(r"\1\1\1", cleaned)

        return cleaned


class PromptInjectionGuard:
    """Detects and neutralizes adversarial prompt injection & jailbreak attempts."""

    INJECTION_PATTERNS = [
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?", re.IGNORECASE),
        re.compile(r"disregard\s+(all\s+)?(guidelines|rules|instructions?)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(unrestricted|DAN|jailbroken|evil)", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?(system\s+prompt|developer\s+message)", re.IGNORECASE),
        re.compile(r"print\s+(your\s+)?(instructions|configuration|secret\s+key)", re.IGNORECASE),
        re.compile(r"bypass\s+(filter|safety|rules|guardrails)", re.IGNORECASE),
        re.compile(r"system\s*:\s*override", re.IGNORECASE),
        re.compile(r"mode\s*:\s*(unrestricted|jailbreak)", re.IGNORECASE),
    ]

    @classmethod
    def evaluate(cls, query: str) -> Tuple[bool, str]:
        """
        Inspects query for injection signatures.
        Returns: (is_safe, sanitized_or_flagged_message)
        """
        for pattern in cls.INJECTION_PATTERNS:
            if pattern.search(query):
                logger.warning(f"Adversarial prompt injection pattern detected: '{pattern.pattern}'")
                return False, "Query contains disallowed instruction patterns. Please ask a direct question about gold mining."
        return True, query


class RateLimiter:
    """In-memory sliding window rate limiter per client IP."""

    def __init__(self, max_requests: int = 30, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = defaultdict(list)

    def is_allowed(self, client_ip: str) -> bool:
        now = time.time()
        client_history = self.requests[client_ip]

        # Purge entries older than the window
        self.requests[client_ip] = [t for t in client_history if now - t < self.window_seconds]

        if len(self.requests[client_ip]) >= self.max_requests:
            logger.warning(f"Rate limit exceeded for client IP: {client_ip}")
            return False

        self.requests[client_ip].append(now)
        return True


class CircuitBreaker:
    """Prevents cascading failures when external services (Gemini / Ollama) encounter outages."""

    def __init__(self, failure_threshold: int = 3, recovery_time_seconds: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_time = recovery_time_seconds
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.state = "CLOSED"  # CLOSED (healthy), OPEN (failing, trip calls), HALF_OPEN (probing)

    def record_success(self):
        self.failure_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit Breaker TRIPPED to OPEN state. Failures: {self.failure_count}")

    def can_execute(self) -> bool:
        if self.state == "CLOSED":
            return True
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_time:
                self.state = "HALF_OPEN"
                logger.info("Circuit Breaker probe: transitioning to HALF_OPEN")
                return True
            return False
        if self.state == "HALF_OPEN":
            return True
        return True


class DefensiveMiddleware(BaseHTTPMiddleware):
    """
    FastAPI Global Middleware:
    1. Intercepts all unhandled exceptions and prevents server crash.
    2. Sanitizes headers (removes fingerprinting Server tokens).
    3. Enforces client IP rate limits.
    """

    def __init__(self, app, rate_limiter: Optional[RateLimiter] = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter or RateLimiter()

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"

        # Rate limit API requests
        if request.url.path.startswith("/api/"):
            if not self.rate_limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={
                        "error": "Rate limit exceeded. Please wait a moment before sending more queries.",
                        "status": "rate_limited"
                    }
                )

        try:
            response = await call_next(request)
            # Security headers
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Server"] = "AuriSearch-Protected"
            return response

        except DefensiveSecurityException as dse:
            logger.warning(f"Security intercept: {dse.message}")
            return JSONResponse(
                status_code=dse.status_code,
                content={"error": dse.message, "status": "security_violation"}
            )

        except Exception as exc:
            # Catch-all defensive shield: never reveal raw tracebacks to users
            logger.error(f"Defensive Middleware caught unhandled error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "An internal operational safeguard handled an unexpected event. The server remains stable.",
                    "status": "recovered"
                }
            )


def validate_user_query(raw_query: str, min_len: int = 2, max_len: int = 800) -> str:
    """
    Full defensive validation pipeline for a query.
    Raises DefensiveSecurityException if query is invalid.
    """
    if not raw_query or not isinstance(raw_query, str):
        raise DefensiveSecurityException("Query cannot be empty or non-string.", status_code=400)

    # 1. Sanitize
    cleaned = InputSanitizer.sanitize_text(raw_query)

    # 2. Length check
    if len(cleaned) < min_len:
        raise DefensiveSecurityException(f"Query is too short (min {min_len} characters required).", status_code=400)
    if len(cleaned) > max_len:
        raise DefensiveSecurityException(f"Query exceeds maximum allowed length of {max_len} characters.", status_code=400)

    # 3. Prompt Injection check
    is_safe, message = PromptInjectionGuard.evaluate(cleaned)
    if not is_safe:
        raise DefensiveSecurityException(message, status_code=400)

    return cleaned
