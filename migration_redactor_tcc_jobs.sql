-- ──────────────────────────────────────────────────────────────────────
-- Migración: redactor_tcc_jobs
-- Persistencia del job store del Redactor TCC v4 entre /analyze y /finalize.
-- Sustituye al dict in-memory que se perdía en cada redeploy de Render.
-- ──────────────────────────────────────────────────────────────────────
--
-- Cómo correr esto:
--   1) Supabase Dashboard → SQL Editor → New query
--   2) Pegar este archivo completo y RUN
--   3) Verificar que la tabla aparezca en Table Editor
--
-- Idempotente: se puede correr varias veces sin romper nada.

create table if not exists public.redactor_tcc_jobs (
    job_id      text primary key,
    pass0       jsonb       not null,
    pass2       jsonb       not null,
    caso_meta   jsonb       not null,
    created_at  timestamptz not null default now(),
    expires_at  timestamptz not null
);

comment on table public.redactor_tcc_jobs is
    'Estado del Redactor TCC v4 entre POST /analyze (escribe) y POST /finalize (lee+borra). TTL=1h. Se inserta tras Pass 2 con el plan filtrado para que el secretario pueda revisarlo en la UI.';

-- Índice para GC eficiente
create index if not exists idx_redactor_tcc_jobs_expires
    on public.redactor_tcc_jobs (expires_at);

-- ──────────────────────────────────────────────────────────────────────
-- Política RLS: solo el service-role key puede tocar esta tabla.
-- (El backend usa SUPABASE_SERVICE_ROLE_KEY; no se expone al cliente.)
-- ──────────────────────────────────────────────────────────────────────

alter table public.redactor_tcc_jobs enable row level security;

-- Revocar acceso al rol anon/authenticated (cliente)
revoke all on public.redactor_tcc_jobs from anon, authenticated;

-- service_role conserva acceso completo por bypass de RLS, no necesita policy.

-- ──────────────────────────────────────────────────────────────────────
-- Función de limpieza opcional (puedes llamarla desde un cron de Supabase)
-- ──────────────────────────────────────────────────────────────────────

create or replace function public.redactor_tcc_jobs_gc()
returns integer
language plpgsql
security definer
as $$
declare
    n_deleted integer;
begin
    delete from public.redactor_tcc_jobs
    where expires_at < now();
    get diagnostics n_deleted = row_count;
    return n_deleted;
end;
$$;

comment on function public.redactor_tcc_jobs_gc is
    'Borra jobs expirados. Llamar desde pg_cron o desde el backend periódicamente.';

-- ──────────────────────────────────────────────────────────────────────
-- Verificación
-- ──────────────────────────────────────────────────────────────────────
-- select tablename, schemaname from pg_tables where tablename='redactor_tcc_jobs';
-- select count(*) from public.redactor_tcc_jobs;
