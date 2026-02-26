"""
Iurexia Legal Cache Manager — Gemini Context Caching
====================================================
Manages the lifecycle of a Gemini cached context containing
12 ultra-clean Mexican legal texts (~1.05M tokens).

Usage:
    from cache_manager import get_cache_name, get_gemini_client

    # In your endpoint:
    cache_name = get_cache_name()
    client = get_gemini_client()
    response = client.models.generate_content(
        model=CACHE_MODEL,
        contents=[user_query],
        config=types.GenerateContentConfig(
            cached_content=cache_name,
            ...
        )
    )
"""

import os
import time
import threading
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("iurexia.cache")

# ── Configuration ────────────────────────────────────────────────────────────
CACHE_MODEL = os.getenv("CACHE_MODEL", "models/gemini-2.5-flash")
CACHE_CORPUS_DIR = os.getenv("CACHE_CORPUS_DIR", "/app/cache_corpus")
CACHE_TTL_HOURS = int(os.getenv("CACHE_TTL_HOURS", "24"))
CACHE_DISPLAY_NAME = "iurexia-legal-corpus-v1"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Global State ─────────────────────────────────────────────────────────────
_cache_name: Optional[str] = None
_cache_created_at: float = 0
_cache_lock = threading.Lock()
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
            logger.info(f"  📄 Loaded {f.name}: {len(text):,} chars (~{tokens_est:,} tokens)")
        except Exception as e:
            logger.error(f"  ❌ Failed to load {f.name}: {e}")
    
    total_chars = sum(len(t) for t in texts)
    total_tokens = total_chars // 4
    logger.info(f"  📚 Total corpus: {len(texts)} files, {total_chars:,} chars (~{total_tokens:,} tokens)")
    
    return texts


async def _create_cache_async() -> Optional[str]:
    """Create a new Gemini cache with the legal corpus (Async)."""
    global _cache_name, _cache_created_at
    
    if not GEMINI_API_KEY:
        logger.error("GEMINI_API_KEY not set — cache disabled")
        return None
    
    try:
        from google import genai
        from google.genai import types as gtypes
        
        # Use aio for non-blocking IO
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Load corpus (CPU bound, slightly blocking but okay for now)
        texts = _load_corpus_texts()
        if not texts:
            logger.error("No corpus texts loaded — cache disabled")
            return None
        
        # Build contents as Parts
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
        
        logger.info(f"🔄 Creating Gemini cache (ASYNC): model={CACHE_MODEL}, ttl={CACHE_TTL_HOURS}h")
        
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
        
        logger.info(f"✅ Cache created (ASYNC): {_cache_name}")
        return _cache_name
        
    except Exception as e:
        logger.error(f"❌ Async Cache creation failed: {e}")
        return None


async def _refresh_if_needed_async():
    """Refresh cache asynchronously if needed."""
    global _cache_name
    
    if not _cache_name:
        return
    
    elapsed = time.time() - _cache_created_at
    ttl_seconds = CACHE_TTL_HOURS * 3600
    if elapsed > (ttl_seconds * 0.8):
        await _create_cache_async()


async def initialize_cache():
    """Initialize the Gemini cache. Optimized for async startup."""
    # Note: Using a lock here is tricky for async, but since this is called
    # usually at startup via ensure_future, it should be fine.
    if _cache_name is not None:
        return _cache_name
    
    return await _create_cache_async()


def get_cache_name() -> Optional[str]:
    """Get the current cache name (Sync version)."""
    # Still keep sync version for simplicity where needed
    return _cache_name


async def get_cache_name_async() -> Optional[str]:
    """Get the current cache name (Async version)."""
    await _refresh_if_needed_async()
    return _cache_name


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
        "cache_available": _cache_name is not None,
        "cache_age_hours": round(elapsed / 3600, 2) if elapsed else 0,
        "cache_ttl_hours": CACHE_TTL_HOURS,
        "cache_remaining_hours": round((ttl_seconds - elapsed) / 3600, 2) if elapsed else 0,
        "corpus_dir": CACHE_CORPUS_DIR,
    }
