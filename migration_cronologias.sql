-- ════════════════════════════════════════════════════════════════════
-- Cronología del expediente: el trabajo y su resultado (12-oct-2026)
--
-- La API corre con tres workers y la petición que sube el expediente no es
-- la que pregunta por el progreso: lo que tiene que sobrevivir entre una y
-- otra vive aquí, no en la memoria de un proceso.
--
-- Escribe sólo la API (service_role). El dueño puede leer sus filas.
-- `latido_at` lo renueva el trabajo mientras corre: una fila «procesando»
-- con el latido viejo es un trabajo que murió con su worker (despliegue,
-- reciclado por --limit-max-requests), y quien la lea la da por fallida y
-- devuelve las consultas.
-- ════════════════════════════════════════════════════════════════════

create table if not exists public.cronologias (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references auth.users(id) on delete cascade,
    estado      text not null default 'en_cola'
                check (estado in ('en_cola', 'procesando', 'lista', 'error')),
    titulo      text,
    carpeta_id  text,
    documentos  jsonb not null default '[]'::jsonb,
    paginas     integer not null default 0,
    consultas   integer not null default 0,
    cobro_id    uuid references public.consultas_cobros(id),
    progreso    jsonb not null default '{}'::jsonb,
    resultado   jsonb,
    error       text,
    latido_at   timestamptz not null default now(),
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create index if not exists cronologias_user_idx
    on public.cronologias (user_id, created_at desc);

alter table public.cronologias enable row level security;

drop policy if exists "cronologias: cada quien ve las suyas" on public.cronologias;
create policy "cronologias: cada quien ve las suyas"
    on public.cronologias for select to authenticated
    using (user_id = auth.uid());
