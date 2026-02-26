"""
Input Sanitizer for Jurexia API
Protects against XSS, SQL injection, and prompt injection attacks.
"""
import re
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# XSS SANITIZATION
# ═══════════════════════════════════════════════════════════════════════════

# HTML tags that should be stripped
_XSS_PATTERNS = [
    re.compile(r'<script[^>]*>.*?</script>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<iframe[^>]*>.*?</iframe>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<object[^>]*>.*?</object>', re.IGNORECASE | re.DOTALL),
    re.compile(r'<embed[^>]*>', re.IGNORECASE),
    re.compile(r'<link[^>]*>', re.IGNORECASE),
    re.compile(r'on\w+\s*=\s*["\'][^"\']*["\']', re.IGNORECASE),  # Event handlers
    re.compile(r'javascript\s*:', re.IGNORECASE),  # javascript: URLs
    re.compile(r'data\s*:\s*text/html', re.IGNORECASE),  # data: URLs
]


def sanitize_xss(text: str) -> str:
    """Remove potential XSS vectors from text while preserving legal content."""
    result = text
    for pattern in _XSS_PATTERNS:
        result = pattern.sub('', result)
    return result


# ═══════════════════════════════════════════════════════════════════════════
# SQL INJECTION DETECTION
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that indicate SQL injection attempts
_SQL_INJECTION_PATTERNS = [
    re.compile(r";\s*DROP\s+", re.IGNORECASE),
    re.compile(r";\s*DELETE\s+FROM\s+", re.IGNORECASE),
    re.compile(r";\s*TRUNCATE\s+", re.IGNORECASE),
    re.compile(r";\s*ALTER\s+TABLE\s+", re.IGNORECASE),
    re.compile(r";\s*UPDATE\s+\w+\s+SET\s+", re.IGNORECASE),
    re.compile(r";\s*INSERT\s+INTO\s+", re.IGNORECASE),
    re.compile(r"UNION\s+ALL\s+SELECT", re.IGNORECASE),
    re.compile(r"UNION\s+SELECT", re.IGNORECASE),
    re.compile(r"'\s*OR\s+'1'\s*=\s*'1", re.IGNORECASE),
    re.compile(r"'\s*OR\s+1\s*=\s*1", re.IGNORECASE),
    re.compile(r"--\s*$", re.MULTILINE),  # SQL comment at end of line
    re.compile(r"/\*.*?\*/", re.DOTALL),  # Block comments
    re.compile(r"xp_cmdshell", re.IGNORECASE),
    re.compile(r"EXEC\s+master", re.IGNORECASE),
]


def detect_sql_injection(text: str) -> bool:
    """
    Check if text contains potential SQL injection patterns.
    Note: Uses conservative patterns to avoid false positives on legal text.
    """
    for pattern in _SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# PROMPT INJECTION GUARD
# ═══════════════════════════════════════════════════════════════════════════

# Patterns that indicate prompt injection attempts
_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"ignore\s+(all\s+)?above\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"forget\s+(all\s+)?previous", re.IGNORECASE),
    re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"system\s+prompt\s*:", re.IGNORECASE),
    re.compile(r"<\|system\|>", re.IGNORECASE),
    re.compile(r"<\|assistant\|>", re.IGNORECASE),
    re.compile(r"<\|user\|>", re.IGNORECASE),
    re.compile(r"```system", re.IGNORECASE),
    re.compile(r"OVERRIDE:\s*", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"DAN\s+mode", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:a\s+)?(?:different|new)\s+(?:AI|assistant|model)", re.IGNORECASE),
    re.compile(r"pretend\s+(?:you\s+are|to\s+be)\s+(?:a\s+)?(?:different|new)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt", re.IGNORECASE),
    re.compile(r"show\s+(?:me\s+)?(?:your|the)\s+(?:system\s+)?prompt", re.IGNORECASE),
]


def detect_prompt_injection(text: str) -> bool:
    """
    Check if text contains prompt injection attempts.
    Returns True if a potential injection is detected.
    """
    for pattern in _PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════
# COMBINED SANITIZER
# ═══════════════════════════════════════════════════════════════════════════

def sanitize_input(text: str) -> Tuple[str, Optional[str]]:
    """
    Apply all sanitization checks to input text.
    
    Returns:
        (sanitized_text, rejection_reason)
        If rejection_reason is not None, the input should be rejected.
    """
    if not text or not text.strip():
        return text, None
    
    # Check for SQL injection (reject)
    if detect_sql_injection(text):
        return "", "Entrada rechazada: se detectó un patrón potencialmente peligroso."
    
    # Check for prompt injection (reject)
    if detect_prompt_injection(text):
        return "", "Tu consulta contiene instrucciones que no puedo procesar. Por favor reformula tu pregunta legal."
    
    # XSS sanitization (clean, don't reject)
    cleaned = sanitize_xss(text)
    
    # Length limit (prevent abuse with extremely long inputs)
    MAX_INPUT_LENGTH = 15000  # 15K characters is generous for a legal query
    if len(cleaned) > MAX_INPUT_LENGTH:
        cleaned = cleaned[:MAX_INPUT_LENGTH]
    
    return cleaned, None
