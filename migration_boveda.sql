-- ═══ LA BÓVEDA (oct-2026) ══════════════════════════════════════════════════
-- Cientos de documentos por carpeta, indexados por usuario y carpeta, para
-- preguntarles en conjunto (chat) y en tabla (filas = documentos, columnas =
-- preguntas). Sólo Platinum; cobra consultas en proporción con
-- `consumir_consultas` (migration_cobro_consultas.sql, que va antes).
--
-- La carpeta ES el expediente que ya existe (`expedientes`): la Bóveda es una
-- gaveta más de la carpeta, no un mundo aparte. Borrar la carpeta borra su
-- Bóveda por cascada, y el disparador deja en `boveda_purgas` lo que vive
-- fuera de Postgres (vectores en Qdrant y archivos en Storage) para que el
-- trabajador lo borre también.
--
-- Todo se escribe desde la API con la clave de servicio. El usuario LEE lo
-- suyo y puede BORRAR lo suyo; no inserta ni edita directo.

-- ── El cubo ────────────────────────────────────────────────────────────────
-- Privado y sin políticas para el usuario: se sube con URL firmada que da la
-- API (después de comprobar plan y cuotas) y se lee con URL firmada también.
-- Así un archivo ya cotizado no se puede cambiar por otro más grande.
insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values ('boveda', 'boveda', false, 52428800,
        array['application/pdf', 'application/gzip',
              'application/vnd.openxmlformats-officedocument.wordprocessingml.document'])
on conflict (id) do nothing;


-- ── Tandas: cada cobro (ingesta o corrida de tabla) ─────────────────────────
create table if not exists public.boveda_tandas (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  expediente_id  uuid not null references public.expedientes(id) on delete cascade,
  tipo           text not null check (tipo in ('ingesta', 'tabla')),
  tabla_id       uuid,
  unidades       integer not null default 0,     -- páginas (ingesta) o celdas (tabla)
  consultas      integer not null default 0,     -- lo cobrado
  movimiento_id  uuid references public.consultas_movimientos(id) on delete set null,
  devueltas      integer not null default 0,
  estado         text not null default 'en_curso' check (estado in ('en_curso', 'terminada')),
  creado_en      timestamptz not null default now(),
  terminado_en   timestamptz
);
create index if not exists boveda_tandas_usuario on public.boveda_tandas (user_id, creado_en desc);


-- ── Documentos ─────────────────────────────────────────────────────────────
-- subiendo → subido → revisando → revisado → en_cola → procesando → listo | error
--   subiendo:   la API dio la URL firmada; el navegador está subiendo
--   subido:     el navegador avisó que terminó; falta contar páginas
--   revisado:   páginas contadas (y cuáles piden OCR); espera la confirmación del cobro
--   en_cola:    cobrado; espera trabajador
create table if not exists public.boveda_documentos (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  expediente_id  uuid not null references public.expedientes(id) on delete cascade,
  nombre         text not null,
  mime_type      text,
  tamano         bigint,
  storage_path   text not null,
  huella         text,
  paginas        integer,
  paginas_ocr    integer,
  caracteres     integer,
  fragmentos     integer,
  estado         text not null default 'subiendo'
                 check (estado in ('subiendo', 'subido', 'revisando', 'revisado',
                                   'en_cola', 'procesando', 'listo', 'error')),
  error          text,
  intentos       smallint not null default 0,
  tomado_en      timestamptz,
  tomado_por     text,
  tanda_id       uuid references public.boveda_tandas(id) on delete set null,
  creado_en      timestamptz not null default now(),
  actualizado_en timestamptz not null default now()
);
create index if not exists boveda_documentos_carpeta on public.boveda_documentos (expediente_id, creado_en);
create index if not exists boveda_documentos_usuario on public.boveda_documentos (user_id);
create index if not exists boveda_documentos_pendientes on public.boveda_documentos (estado, creado_en)
  where estado in ('subido', 'revisando', 'en_cola', 'procesando');
-- El mismo archivo dos veces en la misma carpeta no se cobra dos veces.
create unique index if not exists boveda_documentos_sin_duplicado
  on public.boveda_documentos (expediente_id, huella) where huella is not null;


-- ── Tablas de revisión ─────────────────────────────────────────────────────
-- columnas: [{"id": "c1", "titulo": "Partes", "pregunta": "¿Quiénes son las partes?", "tipo": "texto"}]
create table if not exists public.boveda_tablas (
  id             uuid primary key default gen_random_uuid(),
  user_id        uuid not null references auth.users(id) on delete cascade,
  expediente_id  uuid not null references public.expedientes(id) on delete cascade,
  titulo         text not null default 'Tabla de revisión',
  columnas       jsonb not null default '[]'::jsonb,
  creado_en      timestamptz not null default now(),
  actualizado_en timestamptz not null default now()
);
create index if not exists boveda_tablas_carpeta on public.boveda_tablas (expediente_id);

alter table public.boveda_tandas
  drop constraint if exists boveda_tandas_tabla_fk,
  add constraint boveda_tandas_tabla_fk foreign key (tabla_id)
      references public.boveda_tablas(id) on delete set null;

create table if not exists public.boveda_celdas (
  id             uuid primary key default gen_random_uuid(),
  tabla_id       uuid not null references public.boveda_tablas(id) on delete cascade,
  documento_id   uuid not null references public.boveda_documentos(id) on delete cascade,
  columna_id     text not null,
  user_id        uuid not null references auth.users(id) on delete cascade,
  estado         text not null default 'en_cola'
                 check (estado in ('en_cola', 'procesando', 'lista', 'error')),
  respuesta      jsonb,
  tanda_id       uuid references public.boveda_tandas(id) on delete set null,
  tokens_entrada integer,
  tokens_salida  integer,
  costo_usd      numeric(10, 6),
  intentos       smallint not null default 0,
  tomado_en      timestamptz,
  actualizado_en timestamptz not null default now(),
  unique (tabla_id, documento_id, columna_id)
);
create index if not exists boveda_celdas_pendientes on public.boveda_celdas (estado, actualizado_en)
  where estado in ('en_cola', 'procesando');


-- ── Purgas: lo que vive fuera de Postgres ──────────────────────────────────
create table if not exists public.boveda_purgas (
  id             bigserial primary key,
  user_id        uuid not null,
  expediente_id  uuid,
  documento_id   uuid,
  storage_path   text,
  creado_en      timestamptz not null default now(),
  hecho_en       timestamptz,
  error          text
);
create index if not exists boveda_purgas_pendientes on public.boveda_purgas (creado_en) where hecho_en is null;

create or replace function public.boveda_documento_borrado()
returns trigger
language plpgsql
security definer
set search_path to 'public'
as $function$
begin
  insert into public.boveda_purgas (user_id, expediente_id, documento_id, storage_path)
  values (old.user_id, old.expediente_id, old.id, old.storage_path);
  return old;
end;
$function$;

drop trigger if exists boveda_documento_borrado on public.boveda_documentos;
create trigger boveda_documento_borrado
  after delete on public.boveda_documentos
  for each row execute function public.boveda_documento_borrado();


-- ── RLS ────────────────────────────────────────────────────────────────────
alter table public.boveda_tandas     enable row level security;
alter table public.boveda_documentos enable row level security;
alter table public.boveda_tablas     enable row level security;
alter table public.boveda_celdas     enable row level security;
alter table public.boveda_purgas     enable row level security;   -- sin políticas: sólo servicio

drop policy if exists "Tandas propias" on public.boveda_tandas;
create policy "Tandas propias" on public.boveda_tandas for select using (auth.uid() = user_id);

drop policy if exists "Documentos de bóveda propios" on public.boveda_documentos;
create policy "Documentos de bóveda propios" on public.boveda_documentos for select using (auth.uid() = user_id);
drop policy if exists "Borrar documentos de bóveda propios" on public.boveda_documentos;
create policy "Borrar documentos de bóveda propios" on public.boveda_documentos for delete using (auth.uid() = user_id);

drop policy if exists "Tablas propias" on public.boveda_tablas;
create policy "Tablas propias" on public.boveda_tablas for select using (auth.uid() = user_id);
drop policy if exists "Borrar tablas propias" on public.boveda_tablas;
create policy "Borrar tablas propias" on public.boveda_tablas for delete using (auth.uid() = user_id);

drop policy if exists "Celdas propias" on public.boveda_celdas;
create policy "Celdas propias" on public.boveda_celdas for select using (auth.uid() = user_id);


-- ── La cola: tomar trabajo sin pisarse (3 procesos de uvicorn) ─────────────
-- Primero lo que sólo hay que CONTAR (rápido, y destraba la cotización);
-- luego lo cobrado. Entre usuarios, el que menos tiene en proceso va antes:
-- quinientos documentos de uno no dejan esperando los cinco de otro.
-- Un trabajo tomado hace más de 15 minutos se da por huérfano (el proceso
-- murió en un despliegue) y se vuelve a tomar; al tercer intento, error.
create or replace function public.boveda_tomar_documento(p_trabajador text)
returns setof public.boveda_documentos
language plpgsql
security definer
set search_path to 'public'
as $function$
declare
  v_id uuid;
begin
  -- Huérfanos agotados: error, para que la tanda pueda cerrarse.
  update public.boveda_documentos
     set estado = 'error', error = coalesce(error, 'no se pudo procesar tras 3 intentos'),
         actualizado_en = now()
   where estado in ('revisando', 'procesando')
     and tomado_en < now() - interval '15 minutes'
     and intentos >= 3;

  select d.id into v_id
    from public.boveda_documentos d
   where d.estado in ('subido', 'en_cola')
      or (d.estado in ('revisando', 'procesando') and d.tomado_en < now() - interval '15 minutes')
   order by case when d.estado in ('subido', 'revisando') then 0 else 1 end,
            (select count(*) from public.boveda_documentos x
              where x.user_id = d.user_id and x.estado = 'procesando'),
            d.creado_en
   limit 1
   for update skip locked;

  if v_id is null then
    return;
  end if;

  return query
  update public.boveda_documentos
     set estado = case when estado in ('subido', 'revisando') then 'revisando' else 'procesando' end,
         intentos = intentos + 1,
         tomado_en = now(),
         tomado_por = p_trabajador,
         actualizado_en = now()
   where id = v_id
  returning *;
end;
$function$;

-- Una FILA de la tabla (todas sus celdas en cola) va junta: el documento se
-- lee una vez para todas las preguntas.
create or replace function public.boveda_tomar_fila(p_trabajador text)
returns table (tabla_id uuid, documento_id uuid, columnas text[])
language plpgsql
security definer
set search_path to 'public'
as $function$
declare
  v_tabla uuid;
  v_doc   uuid;
begin
  update public.boveda_celdas c
     set estado = 'error',
         respuesta = jsonb_build_object('error', 'no se pudo responder tras 3 intentos'),
         actualizado_en = now()
   where c.estado = 'procesando' and c.tomado_en < now() - interval '10 minutes' and c.intentos >= 3;

  select c.tabla_id, c.documento_id into v_tabla, v_doc
    from public.boveda_celdas c
   where c.estado = 'en_cola'
      or (c.estado = 'procesando' and c.tomado_en < now() - interval '10 minutes')
   order by c.actualizado_en
   limit 1
   for update skip locked;

  if v_tabla is null then
    return;
  end if;

  return query
  with tomadas as (
    update public.boveda_celdas c
       set estado = 'procesando', intentos = c.intentos + 1, tomado_en = now(), actualizado_en = now()
     where c.id in (
       select x.id from public.boveda_celdas x
        where x.tabla_id = v_tabla and x.documento_id = v_doc
          and (x.estado = 'en_cola' or (x.estado = 'procesando' and x.tomado_en < now() - interval '10 minutes'))
          for update skip locked)
    returning c.columna_id
  )
  select v_tabla, v_doc, array_agg(t.columna_id) from tomadas t having count(*) > 0;
end;
$function$;

revoke all on function public.boveda_tomar_documento(text) from public, anon, authenticated;
revoke all on function public.boveda_tomar_fila(text) from public, anon, authenticated;
revoke all on function public.boveda_documento_borrado() from public, anon, authenticated;
grant execute on function public.boveda_tomar_documento(text) to service_role;
grant execute on function public.boveda_tomar_fila(text) to service_role;
