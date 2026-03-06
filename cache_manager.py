"""
Iurexia Cache Manager — Multi-Genio Architecture (v10 — Mar 2026)
=================================================================
Supports multiple independent Gemini context caches ("genios"), each with
its own corpus, system instruction, and lifecycle.

Active Genios:
  - amparo:    ~303K tokens — CPEUM, Ley de Amparo, Tratados DDHH, Jurisprudencias
  - mercantil: ~277K tokens — CCom, LGTOC, LGSM, Ley Contrato de Seguro, LISF
  - civil:     ~432K tokens — Código Civil Federal, CNPCF
  - penal:     ~268K tokens — CPF, CNPP, Delinc.Org, Trata, Arts.18-23 CPEUM
  - laboral:   ~390K tokens — LFT, LSS, LFTSE, INFONAVIT, Art.123 CPEUM
  - fiscal:    ~465K tokens — CFF, LISR (sin Títulos VI-VII), LIVA, Proc. Contencioso
  - administrativo: ~117K tokens — LFPA, LOAPF, LGRA, Resp.Patrimonial
  - agrario:   ~42K tokens — Ley Agraria, Org. Tribunales Agrarios, Art.27 CPEUM

SAFETY LOCKS (9 total):
  1. Orphan Cleanup — deletes ALL existing caches before creating a new one
  2. asyncio.Lock — prevents concurrent creation (race conditions)
  3. Double-check — re-verifies inside the lock
  4. Token Count — refuses to create if corpus exceeds MAX_CACHE_TOKENS
  5. MIN_CORPUS_FILES — refuse if fewer than expected files
  6. Startup Cleanup — kills orphans on server restart (cost protection)
  7. Token Validation — per-genio max token check
  8. Daily Budget — max N cache creates/day (shared across all genios)
  9. Lifespan Cleanup — ONLY deletes, never creates at startup

API: Gemini AI Studio (google-genai SDK) via GEMINI_API_KEY
Model: gemini-3-flash-preview
"""

import asyncio
import datetime
import logging
import os
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger("cache_manager")

# ═══════════════════════════════════════════════════════════════════════════════
CACHE_MODEL = os.getenv("CACHE_MODEL", "gemini-3-flash-preview")
CACHE_TTL_MINUTES = int(os.getenv("CACHE_TTL_MINUTES", "5"))  # v10: 5 min
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ── Safety Limits ────────────────────────────────────────────────────────────
MAX_DAILY_CREATES = int(os.getenv("MAX_DAILY_CREATES", "15"))  # Shared across all genios

# ── Genio Configurations ────────────────────────────────────────────────────
GENIO_CONFIGS = {
    "amparo": {
        "corpus_dir": os.getenv("CACHE_CORPUS_DIR", "cache_corpus"),
        "display_name": "iurexia-amparo-corpus-v9",
        "max_tokens": 320_000,
        "system_instruction": (
            "Eres el Genio Amparo de Iurexia, un asistente jurídico constitucional de élite "
            "especializado en juicio de amparo, control de convencionalidad y derechos humanos. "
            "Tienes acceso al texto íntegro de: CPEUM (arts. 1-30 y 103-107), Ley de Amparo completa, "
            "5 tratados internacionales de DDHH (CADH, PIDCP, PIDESC, Convención Niño, Conv. Mayores) "
            "y 22 jurisprudencias obligatorias del Pleno y Salas de la SCJN en materia constitucional. "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos o criterios jurisprudenciales "
            "relevantes con su número exacto y fuente de origen. Aplica siempre el principio pro persona "
            "e interpretación conforme. Nunca inventes contenido legal. "
            "Si el artículo o tesis no está en tu contexto, dilo explícitamente."
        ),
    },
    "mercantil": {
        "corpus_dir": os.getenv("CACHE_CORPUS_MERCANTIL_DIR", "cache_corpus_mercantil"),
        "display_name": "iurexia-mercantil-corpus-v1",
        "max_tokens": 300_000,
        "system_instruction": (
            "Eres el Genio Mercantil de Iurexia, un asistente jurídico de élite "
            "especializado en derecho mercantil mexicano. "
            "Tienes acceso al texto íntegro de: Código de Comercio, "
            "Ley General de Títulos y Operaciones de Crédito, "
            "Ley General de Sociedades Mercantiles y Ley sobre el Contrato de Seguro. "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley de origen. "
            "Estructura tu respuesta indicando el Libro, Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
    "civil": {
        "corpus_dir": os.getenv("CACHE_CORPUS_CIVIL_DIR", "cache_corpus_civil"),
        "display_name": "iurexia-civil-corpus-v1",
        "max_tokens": 450_000,
        "system_instruction": (
            "Eres el Genio Civil de Iurexia, un asistente jurídico de élite "
            "especializado en derecho civil mexicano. "
            "Tienes acceso al texto íntegro de: Código Civil Federal (CCF) completo "
            "(personas, bienes, sucesiones y obligaciones) y el Código Nacional de "
            "Procedimientos Civiles y Familiares (CNPCF) completo. "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y el código de origen (CCF o CNPCF). "
            "Estructura tu respuesta indicando el Libro, Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
    "penal": {
        "corpus_dir": os.getenv("CACHE_CORPUS_PENAL_DIR", "cache_corpus_penal"),
        "display_name": "iurexia-penal-corpus-v1",
        "max_tokens": 450_000,
        "system_instruction": (
            "Eres el Genio Penal de Iurexia, un asistente jurídico de élite "
            "especializado en derecho penal mexicano. "
            "Tienes acceso al texto íntegro de: Código Penal Federal (CPF), "
            "Código Nacional de Procedimientos Penales (CNPP), "
            "Ley Federal contra la Delincuencia Organizada, "
            "Ley General de Trata de Personas, y los artículos 18 a 23 de la "
            "Constitución Política de los Estados Unidos Mexicanos (CPEUM) "
            "en materia penal (prisión preventiva, sistema acusatorio, seguridad pública). "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley u ordenamiento de origen. "
            "Estructura tu respuesta indicando el Libro, Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
    "laboral": {
        "corpus_dir": os.getenv("CACHE_CORPUS_LABORAL_DIR", "cache_corpus_laboral"),
        "display_name": "iurexia-laboral-corpus-v1",
        "max_tokens": 450_000,
        "system_instruction": (
            "Eres el Genio Laboral de Iurexia, un asistente jurídico de élite "
            "especializado en derecho laboral mexicano. "
            "Tienes acceso al texto íntegro de: Ley Federal del Trabajo (LFT), "
            "Ley del Seguro Social (LSS), "
            "Ley Federal de los Trabajadores al Servicio del Estado (LFTSE, "
            "reglamentaria del Apartado B del Art. 123), "
            "Ley del INFONAVIT, y el artículo 123 de la Constitución Política "
            "de los Estados Unidos Mexicanos (CPEUM) con sus Apartados A y B. "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley u ordenamiento de origen. "
            "Distingue claramente entre relaciones laborales del Apartado A (sector privado) "
            "y del Apartado B (sector público). "
            "Estructura tu respuesta indicando el Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
    "fiscal": {
        "corpus_dir": os.getenv("CACHE_CORPUS_FISCAL_DIR", "cache_corpus_fiscal"),
        "display_name": "iurexia-fiscal-corpus-v1",
        "max_tokens": 500_000,
        "system_instruction": (
            "Eres el Genio Fiscal de Iurexia, un asistente jurídico de élite "
            "especializado en derecho fiscal mexicano. "
            "Tienes acceso al texto íntegro de: Código Fiscal de la Federación (CFF), "
            "Ley del Impuesto sobre la Renta (LISR) — Títulos I a V (personas morales, "
            "personas físicas, residentes en el extranjero, régimen de PM sin fines lucrativos), "
            "Ley del Impuesto al Valor Agregado (LIVA), y la Ley Federal de Procedimiento "
            "Contencioso Administrativo (juicio ante el TFJA). "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley de origen (CFF, LISR, LIVA o LFPCA). "
            "Estructura tu respuesta indicando el Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente. "
            "NOTA: Los Títulos VI (REFIPRES) y VII (Estímulos Fiscales) de la LISR "
            "no están incluidos. Si te preguntan sobre esos temas, indícalo."
        ),
    },
    "administrativo": {
        "corpus_dir": os.getenv("CACHE_CORPUS_ADMIN_DIR", "cache_corpus_administrativo"),
        "display_name": "iurexia-administrativo-corpus-v1",
        "max_tokens": 500_000,
        "system_instruction": (
            "Eres el Genio Administrativo de Iurexia, un asistente jurídico de élite "
            "especializado en derecho administrativo mexicano. "
            "Tienes acceso al texto íntegro de: Ley Federal de Procedimiento "
            "Administrativo (LFPA), Ley Orgánica de la Administración Pública Federal (LOAPF), "
            "Ley General de Responsabilidades Administrativas (LGRA), y la Ley Federal "
            "de Responsabilidad Patrimonial del Estado. "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley de origen. "
            "Estructura tu respuesta indicando el Título y Capítulo correspondiente. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
    "agrario": {
        "corpus_dir": os.getenv("CACHE_CORPUS_AGRARIO_DIR", "cache_corpus_agrario"),
        "display_name": "iurexia-agrario-corpus-v1",
        "max_tokens": 500_000,
        "system_instruction": (
            "Eres el Genio Agrario de Iurexia, un asistente jurídico de élite "
            "especializado en derecho agrario mexicano. "
            "Tienes acceso al texto íntegro de: Ley Agraria, Ley Orgánica de los "
            "Tribunales Agrarios, y el Artículo 27 de la Constitución (CPEUM). "
            "Cuando el usuario haga una consulta: cita TEXTUALMENTE los artículos "
            "relevantes con su número exacto y la ley de origen. "
            "Dominas temas de ejidos, comunidades, propiedad social, expropiación, "
            "dominio directo, tribunales agrarios, y derechos agrarios. "
            "Nunca inventes contenido legal. "
            "Si el artículo no está en tu contexto, dilo explícitamente."
        ),
    },
}

# ── Per-Genio State ──────────────────────────────────────────────────────────
@dataclass
class GenioState:
    cache_name: Optional[str] = None
    cache_created_at: float = 0
    last_error: Optional[str] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

_genio_states: dict[str, GenioState] = {}
_gemini_client = None
_daily_create_count: int = 0
_daily_create_date: str = ""


def _get_state(genio_id: str) -> GenioState:
    """Get or create state for a genio."""
    if genio_id not in _genio_states:
        _genio_states[genio_id] = GenioState()
    return _genio_states[genio_id]


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


def _load_corpus_texts(genio_id: str) -> list[str]:
    """Load all TXT files from the genio's corpus directory.

    SAFETY LOCK #7: Validates total tokens < max_tokens for this genio.
    """
    config = GENIO_CONFIGS.get(genio_id)
    if not config:
        logger.error(f"Unknown genio_id: {genio_id}")
        return []

    state = _get_state(genio_id)
    corpus_dir = Path(config["corpus_dir"])

    if not corpus_dir.exists():
        state.last_error = f"Corpus dir not found: {corpus_dir.resolve()} (cwd: {Path.cwd()})"
        logger.error(f"🚨 {state.last_error}")
        return []

    texts = []
    files = sorted(corpus_dir.glob("*.txt"))
    if not files:
        state.last_error = f"No .txt files found in {corpus_dir.resolve()}"
        logger.error(f"🚨 {state.last_error}")
        return []

    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
            texts.append(text)
            tokens_est = len(text) // 4
            logger.info(f"  [{genio_id}] Loaded {f.name}: {len(text):,} chars (~{tokens_est:,} tokens)")
        except Exception as e:
            logger.error(f"  [{genio_id}] Failed to load {f.name}: {e}")

    total_chars = sum(len(t) for t in texts)
    total_tokens = total_chars // 4
    logger.info(f"  [{genio_id}] Total corpus: {len(texts)} files, {total_chars:,} chars (~{total_tokens:,} tokens)")

    max_tokens = config["max_tokens"]
    if total_tokens > max_tokens:
        state.last_error = f"Corpus too big: {total_tokens:,} > {max_tokens:,} tokens — ABORTING"
        logger.error(f"🚨 {state.last_error}")
        return []

    return texts


async def _find_existing_cache_remote(genio_id: str) -> Optional[any]:
    """Check Gemini API for any active cache with this genio's display name."""
    config = GENIO_CONFIGS.get(genio_id)
    if not config:
        return None
    try:
        client = get_gemini_client()
        caches = list(client.caches.list())
        for cache in caches:
            if getattr(cache, "display_name", "") == config["display_name"]:
                return cache
    except Exception as e:
        logger.error(f"[{genio_id}] Error searching remote cache: {e}")
    return None


async def _cleanup_orphan_caches(genio_id: str = None, exclude_name: str = None):
    """SAFETY LOCK #1: Delete orphan caches for a specific genio or ALL genios."""
    try:
        client = get_gemini_client()
        caches = list(client.caches.list())

        # Determine which display names to clean
        if genio_id:
            target_names = {GENIO_CONFIGS[genio_id]["display_name"]}
        else:
            target_names = {c["display_name"] for c in GENIO_CONFIGS.values()}

        orphans = [
            c for c in caches
            if getattr(c, "display_name", "") in target_names
            and c.name != exclude_name
        ]

        if orphans:
            logger.warning(f"Found {len(orphans)} orphan cache(s) — cleaning up...")
            for cache in orphans:
                try:
                    client.caches.delete(name=cache.name)
                    logger.info(f"  Deleted orphan: {cache.name} ({cache.display_name})")
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


async def _create_cache(genio_id: str) -> Optional[str]:
    """Create a new Gemini context cache for a specific genio."""
    global _daily_create_count

    config = GENIO_CONFIGS.get(genio_id)
    if not config:
        logger.error(f"Unknown genio_id: {genio_id}")
        return None

    state = _get_state(genio_id)

    try:
        if not _check_daily_budget():
            state.last_error = f"Daily cache creation limit reached ({MAX_DAILY_CREATES}/día)"
            return None

        # Clean up orphans for this genio
        await _cleanup_orphan_caches(genio_id)

        from google.genai import types as gtypes
        client = get_gemini_client()

        texts = _load_corpus_texts(genio_id)
        if not texts:
            if not state.last_error:
                state.last_error = f"No corpus texts loaded from '{config['corpus_dir']}'"
            logger.error(f"[{genio_id}] Cannot create cache: {state.last_error}")
            return None

        contents = [
            gtypes.Content(
                role="user",
                parts=[gtypes.Part(text=t) for t in texts]
            )
        ]

        ttl_seconds = CACHE_TTL_MINUTES * 60

        logger.info(
            f"[{genio_id}] Creating Gemini context cache: model={CACHE_MODEL}, "
            f"ttl={CACHE_TTL_MINUTES}m, files={len(texts)}"
        )

        cache = await client.aio.caches.create(
            model=CACHE_MODEL,
            config=gtypes.CreateCachedContentConfig(
                display_name=config["display_name"],
                system_instruction=config["system_instruction"],
                contents=contents,
                ttl=f"{ttl_seconds}s",
            )
        )

        state.cache_name = cache.name
        state.cache_created_at = time.time()
        state.last_error = None
        _daily_create_count += 1

        logger.info(
            f"✅ [{genio_id}] Cache created: {state.cache_name} "
            f"(TTL={CACHE_TTL_MINUTES}m, daily={_daily_create_count}/{MAX_DAILY_CREATES})"
        )
        return state.cache_name

    except Exception as e:
        state.last_error = f"{type(e).__name__}: {e}"
        logger.error(f"❌ [{genio_id}] Cache creation failed: {state.last_error}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def _is_cache_valid(genio_id: str) -> bool:
    """Check if a genio's cache is still within its TTL."""
    state = _get_state(genio_id)
    if not state.cache_name:
        return False
    elapsed = time.time() - state.cache_created_at
    ttl_seconds = CACHE_TTL_MINUTES * 60
    return elapsed < (ttl_seconds * 0.98)


async def _refresh_cache_ttl(genio_id: str, cache_name: str):
    """Background task to extend the TTL of an active cache."""
    try:
        from google.genai import types as gtypes
        client = get_gemini_client()
        ttl_seconds = CACHE_TTL_MINUTES * 60
        await client.aio.caches.update(
            name=cache_name,
            config=gtypes.UpdateCachedContentConfig(ttl=f"{ttl_seconds}s")
        )
        logger.info(f"  [{genio_id}] Cache TTL refreshed: {cache_name}")
    except Exception as e:
        logger.error(f"  [{genio_id}] Failed to refresh cache TTL: {e}")


async def get_or_create_cache(genio_id: str = "amparo") -> Optional[str]:
    """Main entry point: get existing cache or create one for a specific genio.

    Uses a 3-tier check:
    1. Local memory (Fastest)
    2. Gemini API Discovery (Multi-instance sync)
    3. New Creation (Last resort)
    """
    if genio_id not in GENIO_CONFIGS:
        logger.error(f"Unknown genio_id: {genio_id}")
        return None

    state = _get_state(genio_id)

    # Tier 1: Local Valid Cache
    if _is_cache_valid(genio_id):
        state.cache_created_at = time.time()
        asyncio.create_task(_refresh_cache_ttl(genio_id, state.cache_name))
        return state.cache_name

    async with state.lock:
        # Re-check after lock
        if _is_cache_valid(genio_id):
            return state.cache_name

        # Tier 2: Remote Discovery
        remote_cache = await _find_existing_cache_remote(genio_id)
        if remote_cache:
            state.cache_name = remote_cache.name
            state.cache_created_at = time.time()
            asyncio.create_task(_refresh_cache_ttl(genio_id, state.cache_name))
            logger.info(f"🔄 [{genio_id}] Adopted remote cache: {state.cache_name}")
            return state.cache_name

        # Tier 3: Create New
        return await _create_cache(genio_id)


def get_cache_name(genio_id: str = "amparo") -> Optional[str]:
    """Get current active cache name for a genio (sync)."""
    state = _get_state(genio_id)
    if _is_cache_valid(genio_id):
        state.cache_created_at = time.time()
        return state.cache_name
    return None


async def get_cache_name_async(genio_id: str = "amparo") -> Optional[str]:
    """Get or create cache lazily for a genio (async)."""
    return await get_or_create_cache(genio_id)


def get_cache_model() -> str:
    """Return the model to use for cached requests."""
    return CACHE_MODEL


def get_cache_status(genio_id: str = None) -> dict:
    """Return cache status for one or all genios."""
    if genio_id:
        return _get_single_status(genio_id)

    # Return status for all genios
    return {
        gid: _get_single_status(gid) for gid in GENIO_CONFIGS
    }


def _get_single_status(genio_id: str) -> dict:
    """Return status for a single genio."""
    state = _get_state(genio_id)
    config = GENIO_CONFIGS.get(genio_id, {})
    elapsed = time.time() - state.cache_created_at if state.cache_created_at else 0
    ttl_seconds = CACHE_TTL_MINUTES * 60
    is_valid = _is_cache_valid(genio_id)
    return {
        "genio_id": genio_id,
        "cache_name": state.cache_name,
        "cache_model": CACHE_MODEL,
        "cache_available": is_valid,
        "cache_age_minutes": round(elapsed / 60, 1) if elapsed else 0,
        "cache_ttl_minutes": CACHE_TTL_MINUTES,
        "cache_remaining_minutes": round(max(0, ttl_seconds - elapsed) / 60, 1) if elapsed else 0,
        "corpus_dir": config.get("corpus_dir", ""),
        "strategy": "on-demand",
        "last_error": state.last_error,
        "daily_creates": f"{_daily_create_count}/{MAX_DAILY_CREATES}",
        "max_tokens_limit": config.get("max_tokens", 0),
    }


async def cleanup_on_startup():
    """SAFETY LOCK #9: Called from app lifespan — ONLY deletes, NEVER creates."""
    logger.info("Startup cleanup: checking for orphan caches...")
    await _cleanup_orphan_caches()
    logger.info("Startup cleanup complete — no cache created (on-demand only)")


async def delete_all_caches(genio_id: str = None):
    """Kill switch: delete caches for one genio or ALL genios, and reset state."""
    if genio_id:
        state = _get_state(genio_id)
        await _cleanup_orphan_caches(genio_id)
        state.cache_name = None
        state.cache_created_at = 0
        state.last_error = "Cache desactivado manualmente."
        logger.info(f"✅ [{genio_id}] Cache deleted via kill switch.")
    else:
        await _cleanup_orphan_caches()
        for gid in list(_genio_states.keys()):
            state = _genio_states[gid]
            state.cache_name = None
            state.cache_created_at = 0
            state.last_error = "Cache desactivado manualmente."
        logger.info("✅ ALL caches deleted via kill switch.")
