"""
Iurexia Legal Cache Manager — ON-DEMAND Strategy (v6 — Feb 2026)
================================================================
Creates a Gemini cached context (~950K tokens of Mexican legal texts)
ONLY when a user explicitly activates "Genio Jurídico" — NOT at server startup.

Corpus: 12 files (CPEUM, CCF, CPF, LFT, Amparo, CCom, LGTOC + 5 DDHH treaties)
Limit: Gemini 3 Flash Preview = 1,048,576 tokens (1M) HARD LIMIT
Actual: ~950K tokens → fits with ~97K margin for queries/responses

SAFETY LOCKS (9 total):
  1. Orphan Cleanup — deletes ALL existing caches before creating
  2. asyncio.Lock — prevents concurrent creation (race conditions)
  3. Double-check — re-verifies inside the lock
  4. TTL 8 min — auto-expires in Google if unused
  5. Frontend Timer — UI disables after 8 min inactivity
  6. Kill Switch — /genio/kill endpoint for emergencies
  7. Token Limit — ABORTS if corpus exceeds MAX_CACHE_TOKENS
  8. Budget Guard — max 10 cache creates per day
  9. Startup Cleanup — lifespan only DELETES, NEVER creates

Usage:
    cache_name = await get_or_create_cache()
"""

import os
import time
import asyncio
import logging
import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iurexia.cache")

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_MODEL = os.getenv("CACHE_MODEL", "publishers/google/models/gemini-3-flash-preview")
CACHE_CORPUS_DIR = os.getenv("CACHE_CORPUS_DIR", "cache_corpus")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "8"))  # 8 minutes as requested by user
CACHE_DISPLAY_NAME = "iurexia-legal-corpus-v6"

# ── Safety Limits ────────────────────────────────────────────────────────────
MAX_CACHE_TOKENS = 950_000       # HARD SAFETY: abort if corpus exceeds this
MAX_DAILY_CREATES = int(os.getenv("MAX_DAILY_CREATES", "10"))  # Budget guard

# GCP Credentials — defaults to API Key mode for Render compatibility
# (SA key creation blocked by org policy iam.disableServiceAccountKeyCreation)
GCP_PROJECT = os.getenv("GCP_PROJECT", "iurexia-v")
GCP_LOCATION = os.getenv("GCP_LOCATION", "us-central1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Fallback if no SA credentials
USE_VERTEX = os.getenv("USE_VERTEX", "true").lower() == "true"  # True for Render with SA via entrypoint.sh

# ── Global State ─────────────────────────────────────────────────────────────
_cache_name: Optional[str] = None
_cache_created_at: float = 0
_cache_lock = None  # Will be initialized in get_or_create_cache
_gemini_client = None
_last_error: Optional[str] = None  # Last cache error for diagnostics
_daily_create_count: int = 0
_daily_create_date: str = ""  # YYYY-MM-DD format


def _load_corpus_texts() -> list[str]:
    """Load all TXT files from the corpus directory, sorted by name.
    
    SAFETY LOCK #7: Validates total tokens < MAX_CACHE_TOKENS.
    Returns empty list if exceeded (prevents oversized cache creation).
    """
    global _last_error
    corpus_dir = Path(CACHE_CORPUS_DIR)
    if not corpus_dir.exists():
        _last_error = f"Corpus dir not found: {corpus_dir.resolve()}"
        logger.error(f"🚨 {_last_error}")
        return []
    
    texts = []
    files = sorted(corpus_dir.glob("*.txt"))
    
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            texts.append(text)
            tokens_est = len(text) // 4
            logger.info(f"  Loaded {f.name}: {len(text):,} chars (~{tokens_est:,} tokens)")
        except Exception as e:
            logger.error(f"  Failed to load {f.name}: {e}")
    
    total_chars = sum(len(t) for t in texts)
    total_tokens = total_chars // 4
    logger.info(f"  Total corpus: {len(texts)} files, {total_chars:,} chars (~{total_tokens:,} tokens)")
    
    # SAFETY LOCK #7: Token limit validation
    if total_tokens > MAX_CACHE_TOKENS:
        _last_error = f"Corpus too big: {total_tokens:,} > {MAX_CACHE_TOKENS:,} tokens"
        logger.error(
            f"🚨 CORPUS TOO BIG: {total_tokens:,} tokens > MAX {MAX_CACHE_TOKENS:,}. "
            f"ABORTING cache creation to prevent API error/cost overrun."
        )
        return []  # Empty = cache won't be created
    
    return texts


async def _cleanup_orphan_caches():
    """Delete ALL existing caches with our display name before creating a new one.
    
    This prevents the critical bug where each Render deploy/restart
    creates a new cache without deleting the old one, leading to
    N caches running simultaneously at $0.97/hour EACH.
    """
    try:
        from google import genai
        client = get_gemini_client()
        
        caches = list(client.caches.list())
        orphans = [c for c in caches if getattr(c, 'display_name', '') == CACHE_DISPLAY_NAME]
        
        if orphans:
            logger.warning(f"Found {len(orphans)} orphan cache(s) — deleting...")
            for cache in orphans:
                try:
                    client.caches.delete(name=cache.name)
                    logger.info(f"  Deleted orphan: {cache.name}")
                except Exception as e:
                    logger.error(f"  Failed to delete orphan {cache.name}: {e}")
    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")


def _check_daily_budget() -> bool:
    """SAFETY LOCK #8: Budget guard — max N cache creates per calendar day.
    
    Prevents runaway costs if something triggers repeated cache creation.
    Resets at midnight (server time).
    """
    global _daily_create_count, _daily_create_date
    
    today = datetime.date.today().isoformat()
    
    if _daily_create_date != today:
        _daily_create_count = 0
        _daily_create_date = today
    
    if _daily_create_count >= MAX_DAILY_CREATES:
        logger.error(
            f"🚨 BUDGET GUARD: {_daily_create_count} cache creates today "
            f"(max={MAX_DAILY_CREATES}). BLOCKING new cache creation."
        )
        return False
    
    return True


async def _create_cache() -> Optional[str]:
    """Create a new Gemini cache with the legal corpus.
    
    ALWAYS cleans up orphans first to prevent cache multiplication.
    Validates token count and daily budget before creating.
    """
    global _cache_name, _cache_created_at, _daily_create_count, _last_error
    
    try:
        # SAFETY LOCK #8: Check daily budget
        if not _check_daily_budget():
            _last_error = f"Daily cache creation limit reached ({MAX_DAILY_CREATES})"
            return None
        
        # Step 1: Clean up any orphan caches
        await _cleanup_orphan_caches()
        
        from google import genai
        from google.genai import types as gtypes
        
        client = get_gemini_client()
        
        # Step 2: Load corpus (includes SAFETY LOCK #7: token validation)
        texts = _load_corpus_texts()
        if not texts:
            if not _last_error:
                _last_error = f"No corpus texts loaded from {CACHE_CORPUS_DIR} (dir exists: {Path(CACHE_CORPUS_DIR).exists()}, cwd: {Path.cwd()})"
            logger.error(f"No corpus texts loaded — cache disabled. {_last_error}")
            return None
        
        # Step 3: Build cache contents
        contents = [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=t) for t in texts]
            )
        ]
        
        ttl_seconds = CACHE_TTL_MINUTES * 60
        
        system_instruction = (
            "Eres Iurexia, un asistente jurídico mexicano de élite. "
            "Tienes acceso directo al texto íntegro de las siguientes leyes y tratados "
            "internacionales ratificados por México. Cuando el usuario haga una consulta legal, "
            "cita TEXTUALMENTE los artículos relevantes con su número exacto y ley de origen. "
            "Nunca inventes contenido legal. Si un artículo no está en tu contexto, dilo explícitamente."
        )
        
        logger.info(f"Creating Gemini cache ON-DEMAND: model={CACHE_MODEL}, ttl={CACHE_TTL_MINUTES}m")
        
        cache = await client.aio.caches.create(
            model=CACHE_MODEL,
            config=gtypes.CreateCachedContentConfig(
                display_name=CACHE_DISPLAY_NAME,
                system_instruction=system_instruction,
                contents=contents,
                ttl=f"{ttl_seconds}s",
                tools=[gtypes.Tool(google_search=gtypes.GoogleSearch())],
            )
        )
        
        _cache_name = cache.name
        _cache_created_at = time.time()
        _daily_create_count += 1
        
        logger.info(
            f"Cache created: {_cache_name} (TTL={CACHE_TTL_MINUTES}m, "
            f"daily_count={_daily_create_count}/{MAX_DAILY_CREATES}, cost=~$0.09/h)"
        )
        return _cache_name
        
    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
        logger.error(f"Cache creation failed: {_last_error}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _is_cache_valid() -> bool:
    """Check if the current cache is still within its TTL."""
    if not _cache_name:
        return False
    elapsed = time.time() - _cache_created_at
    ttl_seconds = CACHE_TTL_MINUTES * 60
    return elapsed < (ttl_seconds * 0.98)  # 98% of TTL (more aggressive retention)


async def _refresh_cache_ttl(cache_name: str):
    """Background task to extend the TTL of an active cache in Vertex AI."""
    try:
        from google.genai import types as gtypes
        client = get_gemini_client()
        ttl_seconds = CACHE_TTL_MINUTES * 60
        
        await client.aio.caches.update(
            name=cache_name,
            config=gtypes.UpdateCachedContentConfig(
                ttl=f"{ttl_seconds}s"
            )
        )
        logger.info(f"  Cache TTL refreshed for: {cache_name}")
    except Exception as e:
        logger.error(f"  Failed to refresh cache TTL: {e}")

async def get_or_create_cache() -> Optional[str]:
    """Get existing cache or create one on-demand.
    
    This is the MAIN entry point. Called from chat/sentencia endpoints.
    Updates _cache_created_at to implement an inactivity-based timeout.
    """
    global _cache_lock, _cache_created_at
    
    # Fast path: cache exists and is valid
    if _is_cache_valid():
        # Update local timestamp for inactivity timeout logic
        _cache_created_at = time.time()
        # Optionally update Vertex AI TTL in background (don't wait)
        asyncio.create_task(_refresh_cache_ttl(_cache_name))
        return _cache_name
    
    # Ensure lock exists
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    
    async with _cache_lock:
        # Double-check
        if _is_cache_valid():
            _cache_created_at = time.time()
            asyncio.create_task(_refresh_cache_ttl(_cache_name))
            return _cache_name
        
        return await _create_cache()

def get_cache_name() -> Optional[str]:
    """Get current active cache name."""
    global _cache_created_at
    if _is_cache_valid():
        # Update timestamp on hit (Sync version used by some paths)
        _cache_created_at = time.time()
        return _cache_name
    return None


async def get_cache_name_async() -> Optional[str]:
    """Get or create cache lazily (Async version)."""
    return await get_or_create_cache()


def get_gemini_client():
    """Get or create a shared Gemini client instance."""
    global _gemini_client
    
    if _gemini_client is None:
        from google import genai
        
        if USE_VERTEX:
            logger.info(f"Initializing Gemini Client via VERTEX AI (Project: {GCP_PROJECT})")
            _gemini_client = genai.Client(
                vertexai=True,
                project=GCP_PROJECT,
                location=GCP_LOCATION
            )
        else:
            logger.info("Initializing Gemini Client via AI STUDIO (shared key)")
            _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
    return _gemini_client


def get_cache_model() -> str:
    """Return the model to use for cached requests."""
    return CACHE_MODEL


def get_cache_status() -> dict:
    """Return cache status for diagnostics."""
    elapsed = time.time() - _cache_created_at if _cache_created_at else 0
    ttl_seconds = CACHE_TTL_MINUTES * 60
    
    return {
        "cache_name": _cache_name,
        "cache_model": CACHE_MODEL,
        "cache_available": _is_cache_valid(),
        "cache_age_minutes": round(elapsed / 60, 1) if elapsed else 0,
        "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "cache_remaining_minutes": round(max(0, ttl_seconds - elapsed) / 60, 1) if elapsed else 0,
        "corpus_dir": CACHE_CORPUS_DIR,
        "strategy": "on-demand",
        "est_hourly_cost": "$0.09" if _is_cache_valid() else "$0.00",
        "last_error": _last_error,
        "daily_creates": f"{_daily_create_count}/{MAX_DAILY_CREATES}",
        "max_tokens_limit": MAX_CACHE_TOKENS,
    }


async def cleanup_on_startup():
    """SAFETY LOCK #9: Called from app lifespan — ONLY deletes, NEVER creates.
    
    This ensures that server restarts/deploys don't leave orphan caches
    running at $0.97/hour each, without creating new ones automatically.
    """
    logger.info("Startup cleanup: deleting any orphan caches...")
    await _cleanup_orphan_caches()
    logger.info("Startup cleanup complete — NO cache created (on-demand only)")


async def delete_all_caches():
    """Emergency kill switch: delete ALL caches."""
    global _cache_name, _cache_created_at
    
    await _cleanup_orphan_caches()
    _cache_name = None
    _cache_created_at = 0
    logger.info("All caches deleted via kill switch")
