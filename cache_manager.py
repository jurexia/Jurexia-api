"""
Iurexia Legal Cache Manager — ON-DEMAND Strategy
=================================================
Creates a Gemini cached context (~968K tokens of Mexican legal texts)
ONLY when a user actually makes a query — NOT at server startup.

TTL is 1 hour to minimize storage costs ($0.97/hour).

CRITICAL FIX: Ensures only ONE cache exists at a time.
Before creating, deletes any orphaned caches from previous deploys.

Usage:
    # In your endpoint — lazy initialization:
    cache_name = await get_or_create_cache()
    # Returns cache name if available, None otherwise
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iurexia.cache")

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_MODEL = os.getenv("CACHE_MODEL", "models/gemini-3-flash-preview")
CACHE_CORPUS_DIR = os.getenv("CACHE_CORPUS_DIR", "/app/cache_corpus")
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "1"))  # 1 hour default (was 24!)
CACHE_DISPLAY_NAME = "iurexia-legal-corpus-v1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Global State ─────────────────────────────────────────────────────────────
_cache_name: Optional[str] = None
_cache_created_at: float = 0
_cache_lock = asyncio.Lock() if asyncio.get_event_loop().is_running() else None
_gemini_client = None


def _load_corpus_texts() -> list[str]:
    """Load all TXT files from the corpus directory, sorted by name."""
    corpus_dir = Path(CACHE_CORPUS_DIR)
    if not corpus_dir.exists():
        logger.warning(f"Cache corpus dir not found: {corpus_dir}")
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
    
    return texts


async def _cleanup_orphan_caches():
    """Delete ALL existing caches with our display name before creating a new one.
    
    This prevents the critical bug where each Render deploy/restart
    creates a new cache without deleting the old one, leading to
    N caches running simultaneously at $0.97/hour EACH.
    """
    if not GEMINI_API_KEY:
        return
    
    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
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


async def _create_cache() -> Optional[str]:
    """Create a new Gemini cache with the legal corpus.
    
    ALWAYS cleans up orphans first to prevent cache multiplication.
    """
    global _cache_name, _cache_created_at
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — cache disabled")
        return None
    
    try:
        # Step 1: Clean up any orphan caches
        await _cleanup_orphan_caches()
        
        from google import genai
        from google.genai import types as gtypes
        
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Step 2: Load corpus
        texts = _load_corpus_texts()
        if not texts:
            logger.error("No corpus texts loaded — cache disabled")
            return None
        
        # Step 3: Build cache contents
        contents = [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=t) for t in texts]
            )
        ]
        
        ttl_seconds = CACHE_TTL_HOURS * 3600
        
        system_instruction = (
            "Eres Iurexia, un asistente jurídico mexicano de élite. "
            "Tienes acceso directo al texto íntegro de las siguientes leyes y tratados "
            "internacionales ratificados por México. Cuando el usuario haga una consulta legal, "
            "cita TEXTUALMENTE los artículos relevantes con su número exacto y ley de origen. "
            "Nunca inventes contenido legal. Si un artículo no está en tu contexto, dilo explícitamente."
        )
        
        logger.info(f"Creating Gemini cache ON-DEMAND: model={CACHE_MODEL}, ttl={CACHE_TTL_HOURS}h")
        
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
        
        hourly_cost = 0.97  # ~968K tokens at $1/M/hour
        logger.info(f"Cache created: {_cache_name} (TTL={CACHE_TTL_HOURS}h, cost=~${hourly_cost:.2f}/h)")
        return _cache_name
        
    except Exception as e:
        logger.error(f"Cache creation failed: {e}")
        return None


def _is_cache_valid() -> bool:
    """Check if the current cache is still within its TTL."""
    if not _cache_name:
        return False
    elapsed = time.time() - _cache_created_at
    ttl_seconds = CACHE_TTL_HOURS * 3600
    return elapsed < (ttl_seconds * 0.9)  # 90% of TTL


async def get_or_create_cache() -> Optional[str]:
    """Get existing cache or create one on-demand.
    
    This is the MAIN entry point. Called from chat/sentencia endpoints.
    Thread-safe via asyncio.Lock to prevent duplicate cache creation.
    """
    global _cache_lock
    
    # Fast path: cache exists and is valid
    if _is_cache_valid():
        return _cache_name
    
    # Ensure lock exists (handles import-time vs runtime)
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()
    
    async with _cache_lock:
        # Double-check after acquiring lock
        if _is_cache_valid():
            return _cache_name
        
        return await _create_cache()


def get_cache_name() -> Optional[str]:
    """Get the current cache name (Sync version). Returns None if no active cache."""
    if _is_cache_valid():
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
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    
    return _gemini_client


def get_cache_model() -> str:
    """Return the model to use for cached requests."""
    return CACHE_MODEL


def get_cache_status() -> dict:
    """Return cache status for diagnostics."""
    elapsed = time.time() - _cache_created_at if _cache_created_at else 0
    ttl_seconds = CACHE_TTL_HOURS * 3600
    
    return {
        "cache_name": _cache_name,
        "cache_model": CACHE_MODEL,
        "cache_available": _is_cache_valid(),
        "cache_age_minutes": round(elapsed / 60, 1) if elapsed else 0,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "cache_remaining_minutes": round(max(0, ttl_seconds - elapsed) / 60, 1) if elapsed else 0,
        "corpus_dir": CACHE_CORPUS_DIR,
        "strategy": "on-demand",
        "est_hourly_cost": "$0.97" if _is_cache_valid() else "$0.00",
    }


async def delete_all_caches():
    """Emergency kill switch: delete ALL caches."""
    global _cache_name, _cache_created_at
    
    await _cleanup_orphan_caches()
    _cache_name = None
    _cache_created_at = 0
    logger.info("All caches deleted via kill switch")
