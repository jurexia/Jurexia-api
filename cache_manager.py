"""
Iurexia Legal Cache Manager — ON-DEMAND Strategy (v8 — Feb 2026)
================================================================
Creates a Gemini cached context (~950K tokens of Mexican legal texts)
ONLY when a user explicitly activates "Genio Jurídico" — NOT at server startup.

│ REGLA PERMANENTE DEL MODELO │
│ Modelo: gemini-3-flash-preview                              │
│ Input token limit: 1,048,576 (fuente: ai.google.dev/docs)  │
│ NO usar gemini-2.0 — está OBSOLETO                          │

API: Gemini AI Studio (google-genai SDK) via GEMINI_API_KEY
Model: gemini-3-flash-preview
Corpus: 12 files in cache_corpus/ — CPEUM, CCF, CPF, LFT, Amparo, CCom, LGTOC + 5 DDHH treaties
Hard limit: 1,048,576 tokens (límite oficial del modelo según Google)

SAFETY LOCKS (9 total):
  1. Orphan Cleanup — deletes ALL existing caches before creating a new one
  2. asyncio.Lock — prevents concurrent creation (race conditions)
  3. Double-check — re-verifies inside the lock
  4. TTL 8 min — auto-expires in Gemini if unused
  5. Frontend Timer — UI disables button after 8 min inactivity
  6. Kill Switch — /genio/kill endpoint for emergencies
  7. Token Limit — ABORTS if corpus exceeds MAX_CACHE_TOKENS
  8. Budget Guard — max 10 cache creates per day
  9. Startup Cleanup — lifespan ONLY DELETES, NEVER creates

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

# ══ MODELO OFICIAL ══════════════════════════════════════════════════════════════
# SIEMPRE: gemini-3-flash-preview  (input limit 1,048,576 — fuente: ai.google.dev)
# NUNCA:   gemini-2.0-flash ni ninguna variante 2.x — OBSOLETOS
# ═══════════════════════════════════════════════════════════════════════════════
CACHE_MODEL = os.getenv("CACHE_MODEL", "gemini-3-flash-preview")
CACHE_CORPUS_DIR = os.getenv("CACHE_CORPUS_DIR", "cache_corpus")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "8"))
CACHE_DISPLAY_NAME = "iurexia-legal-corpus-v7"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Safety Limits ────────────────────────────────────────────────────────────
MAX_CACHE_TOKENS = 1_048_576     # Límite oficial de gemini-3-flash-preview (fuente: ai.google.dev)
MAX_DAILY_CREATES = int(os.getenv("MAX_DAILY_CREATES", "10"))  # Budget guard

# ── Global State ─────────────────────────────────────────────────────────────
_cache_name: Optional[str] = None
_cache_created_at: float = 0
_cache_lock = None  # Initialized lazily in get_or_create_cache
_gemini_client = None
_last_error: Optional[str] = None
_daily_create_count: int = 0
_daily_create_date: str = ""  # YYYY-MM-DD


def get_gemini_client():
    """Get or create the shared Gemini AI Studio client (singleton)."""
    global _gemini_client
    if _gemini_client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set — cannot create Gemini client")
        from google import genai
        logger.info("Initializing Gemini Client via AI Studio (GEMINI_API_KEY)")
        _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client


def _load_corpus_texts() -> list[str]:
    """Load all TXT files from the corpus directory, sorted by name.

    SAFETY LOCK #7: Validates total tokens < MAX_CACHE_TOKENS.
    Returns empty list if exceeded (prevents oversized cache creation).
    """
    global _last_error
    corpus_dir = Path(CACHE_CORPUS_DIR)
    if not corpus_dir.exists():
        _last_error = f"Corpus dir not found: {corpus_dir.resolve()} (cwd: {Path.cwd()})"
        logger.error(f"🚨 {_last_error}")
        return []

    texts = []
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        _last_error = f"No .txt files found in {corpus_dir.resolve()}"
        logger.error(f"🚨 {_last_error}")
        return []

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
        _last_error = f"Corpus too big: {total_tokens:,} > {MAX_CACHE_TOKENS:,} tokens — ABORTING"
        logger.error(f"🚨 {_last_error}")
        return []

    return texts


async def _find_existing_cache_remote() -> Optional[any]:
    """Check Gemini API for any active cache with our display name (Sync for Pro/Multi-instance)."""
    try:
        client = get_gemini_client()
        # caches.list() es un iterador, lo convertimos a lista
        caches = list(client.caches.list())
        for cache in caches:
            if getattr(cache, "display_name", "") == CACHE_DISPLAY_NAME:
                # Verificar que no esté vencido (Gemini lo borra solo, pero por si acaso)
                return cache
    except Exception as e:
        logger.error(f"Error searching remote cache: {e}")
    return None

async def _cleanup_orphan_caches(exclude_name: str = None):
    """SAFETY LOCK #1: Delete orphan caches EXCEPT the one we might be using."""
    try:
        client = get_gemini_client()
        caches = list(client.caches.list())
        orphans = [c for c in caches if getattr(c, "display_name", "") == CACHE_DISPLAY_NAME and c.name != exclude_name]

        if orphans:
            logger.warning(f"Found {len(orphans)} orphan cache(s) — cleaning up...")
            for cache in orphans:
                try:
                    client.caches.delete(name=cache.name)
                    logger.info(f"  Deleted orphan: {cache.name}")
                except Exception as e:
                    logger.error(f"  Failed to delete orphan {cache.name}: {e}")
    except Exception as e:
        logger.error(f"Orphan cleanup failed: {e}")


def _check_daily_budget() -> bool:
    """SAFETY LOCK #8: Budget guard — max N cache creates per calendar day."""
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
    """Create a new Gemini context cache with the legal corpus.

    Always cleans up orphans first. Validates token count and daily budget.
    """
    global _cache_name, _cache_created_at, _daily_create_count, _last_error

    try:
        # SAFETY LOCK #8
        if not _check_daily_budget():
            _last_error = f"Daily cache creation limit reached ({MAX_DAILY_CREATES}/día)"
            return None

        # SAFETY LOCK #1: Clean up orphans first
        await _cleanup_orphan_caches()

        from google.genai import types as gtypes

        client = get_gemini_client()

        # SAFETY LOCK #7: Load and validate corpus
        texts = _load_corpus_texts()
        if not texts:
            if not _last_error:
                _last_error = f"No corpus texts loaded from '{CACHE_CORPUS_DIR}'"
            logger.error(f"Cannot create cache: {_last_error}")
            return None

        # Build cache contents (user turn with all legal texts)
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

        logger.info(
            f"Creating Gemini context cache ON-DEMAND: model={CACHE_MODEL}, "
            f"ttl={CACHE_TTL_MINUTES}m, files={len(texts)}"
        )

        # Create the cache — NO tools (not supported in cached content)
        cache = await client.aio.caches.create(
            model=CACHE_MODEL,
            config=gtypes.CreateCachedContentConfig(
                display_name=CACHE_DISPLAY_NAME,
                system_instruction=system_instruction,
                contents=contents,
                ttl=f"{ttl_seconds}s",
            )
        )

        _cache_name = cache.name
        _cache_created_at = time.time()
        _daily_create_count += 1
        _last_error = None  # Clear any previous error on success

        logger.info(
            f"✅ Cache created: {_cache_name} "
            f"(TTL={CACHE_TTL_MINUTES}m, daily={_daily_create_count}/{MAX_DAILY_CREATES})"
        )
        return _cache_name

    except Exception as e:
        _last_error = f"{type(e).__name__}: {e}"
        logger.error(f"❌ Cache creation failed: {_last_error}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _is_cache_valid() -> bool:
    """Check if the current cache is still within its TTL."""
    if not _cache_name:
        return False
    elapsed = time.time() - _cache_created_at
    ttl_seconds = CACHE_TTL_MINUTES * 60
    return elapsed < (ttl_seconds * 0.98)  # 98% of TTL for safety margin


async def _refresh_cache_ttl(cache_name: str):
    """Background task to extend the TTL of an active cache."""
    try:
        from google.genai import types as gtypes
        client = get_gemini_client()
        ttl_seconds = CACHE_TTL_MINUTES * 60
        await client.aio.caches.update(
            name=cache_name,
            config=gtypes.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
        )
        logger.info(f"  Cache TTL refreshed: {cache_name}")
    except Exception as e:
        logger.error(f"  Failed to refresh cache TTL: {e}")


async def get_or_create_cache() -> Optional[str]:
    """Main entry point: get existing cache (local or remote) or create one.

    Uses a 3-tier check:
    1. Local memory (Fastest)
    2. Gemini API Discovery (Multi-instance sync)
    3. New Creation (Last resort)
    """
    global _cache_lock, _cache_name, _cache_created_at

    # 1. Tier: Local Valid Cache
    if _is_cache_valid():
        _cache_created_at = time.time()
        asyncio.create_task(_refresh_cache_ttl(_cache_name))
        return _cache_name

    # Ensure lock for creation/discovery
    if _cache_lock is None:
        _cache_lock = asyncio.Lock()

    async with _cache_lock:
        # Re-check local after lock
        if _is_cache_valid():
            return _cache_name

        # 2. Tier: Remote Discovery (For Render multi-instance support)
        remote_cache = await _find_existing_cache_remote()
        if remote_cache:
            _cache_name = remote_cache.name
            # Si lo acabamos de encontrar, estimamos su creación como 'ahora' 
            # para que _is_cache_valid() no lo mate de inmediato, 
            # y dejamos que el refresh extienda su vida.
            _cache_created_at = time.time()
            asyncio.create_task(_refresh_cache_ttl(_cache_name))
            logger.info(f"🔄 Adopted remote cache from another instance: {_cache_name}")
            return _cache_name

        # 3. Tier: Create New
        return await _create_cache()


def get_cache_name() -> Optional[str]:
    """Get current active cache name (sync)."""
    global _cache_created_at
    if _is_cache_valid():
        _cache_created_at = time.time()
        return _cache_name
    return None


async def get_cache_name_async() -> Optional[str]:
    """Get or create cache lazily (async)."""
    return await get_or_create_cache()


def get_cache_model() -> str:
    """Return the model to use for cached requests."""
    return CACHE_MODEL


def get_cache_status() -> dict:
    """Return cache status for diagnostics and the /genio/status endpoint."""
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

    Ensures server restarts don't leave orphan caches running at cost.
    """
    logger.info("Startup cleanup: checking for orphan caches...")
    await _cleanup_orphan_caches()
    logger.info("Startup cleanup complete — no cache created (on-demand only)")


async def delete_all_caches():
    """Emergency kill switch: delete ALL caches immediately and reset state."""
    global _cache_name, _cache_created_at, _last_error
    try:
        await _cleanup_orphan_caches()
        logger.info("✅ All Gemini caches deleted via kill switch.")
    except Exception as e:
        logger.error(f"Error during manual cache deletion: {e}")
    
    # RESET TOTAL: Esto asegura que la siguiente consulta no intente usar un ID muerto
    _cache_name = None
    _cache_created_at = 0
    _last_error = "Cache desactivado manualmente por el usuario."
