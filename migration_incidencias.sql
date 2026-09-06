-- ═══════════════════════════════════════════════════════════════════════
-- EL LIBRO DE INCIDENCIAS                                    6-sep-2026
-- ═══════════════════════════════════════════════════════════════════════
--
-- POR QUÉ EXISTE
-- --------------
-- Las quejas entran por tres puertas distintas y ninguna se hablaba con las
-- otras: los reportes de la plataforma (`user_feedback`), las correcciones
-- que hace un abogado cuando le discute una respuesta (`correcciones_usuario`)
-- y el correo a soporte@. Cada puerta tenía su propio aviso y ninguna tenía
-- seguimiento, así que la cola creció hasta 182 pendientes con el más viejo
-- del 4 de mayo — cuatro meses.
--
-- Esta tabla es el sitio donde las tres desembocan para recibir el MISMO
-- trato: triaje, verificación, corrección y cierre. Una incidencia no es una
-- queja: es una queja ya clasificada y en camino a alguna parte.
--
-- LA REGLA QUE GOBIERNA TODO
-- --------------------------
--   Los datos se corrigen solos; el comportamiento no.
--
-- Indexar una norma que faltaba, corregir la vigencia de un documento o
-- reindexar un texto mal troceado se aplica sin preguntar. Tocar un prompt,
-- un umbral, el código o el cobro exige visto bueno humano. `requiere_vb`
-- es esa frontera hecha columna: mientras esté en true y sin `aprobada_at`,
-- nada se aplica.
--
-- LO QUE APRENDIÓ EL TRIAJE EN FRÍO (los 182 de la cola)
-- ------------------------------------------------------
--   41  buzón equivocado — consultas jurídicas escritas en la caja de reportes
--   52  soporte          — se contesta, no se arregla
--   48  defecto          — fallo reproducible
--   17  encuesta         — la salida de cancelación cayendo aquí por error
--   17  calidad          — la respuesta llegó, pero estaba mal
--    7  mejora           — petición de función
--
-- Y el dato que decide el diseño: las reglas por patrón aciertan ~76%. El
-- resto necesita al modelo. Un `historial` escrito «hostorial» se escapa;
-- tres quejas de artículos que no dicen la palabra «artículos» se escapan.
-- Por eso `triaje_por` distingue quién clasificó: si el modelo cambia mucho
-- lo que dictó la regla, la regla está mal y hay que verlo.

create table if not exists public.incidencias (
    id              uuid primary key default gen_random_uuid(),

    -- ── De dónde vino ───────────────────────────────────────────────────
    origen          text not null check (origen in ('reporte','correccion','correo')),
    origen_id       text not null,
    folio           text,
    user_email      text,
    user_id         uuid,
    texto           text not null,
    -- Para las correcciones: la respuesta que el abogado señaló como mala.
    -- Sin esto la queja no se puede verificar, sólo creer.
    contexto        text,
    creado_at       timestamptz not null default now(),

    -- ── Triaje ──────────────────────────────────────────────────────────
    familia         text check (familia in
                      ('defecto','calidad','soporte','mejora','encuesta','buzon-equivocado')),
    clase           text,
    triaje_por      text check (triaje_por in ('regla','modelo','humano')),
    confianza       numeric(3,2),

    -- ── Vida de la incidencia ───────────────────────────────────────────
    -- nueva → triada → verificando → {confirmada | no_reproducible}
    --   confirmada → {corregida | espera_vb} → cerrada
    estado          text not null default 'nueva' check (estado in
                      ('nueva','triada','verificando','confirmada','no_reproducible',
                       'corregida','espera_vb','cerrada','descartada')),

    -- Qué se intentó para reproducirla y qué pasó. Se guarda entero: es lo
    -- único que distingue «lo arreglé» de «me lo creí».
    verificacion    jsonb,
    diagnostico     text,
    correccion      text,

    -- ── La frontera ─────────────────────────────────────────────────────
    requiere_vb     boolean not null default true,
    aprobada_at     timestamptz,
    aprobada_por    text,

    -- ── Agrupación ──────────────────────────────────────────────────────
    -- Siete reportes de tres personas describían el mismo fallo de artículos.
    -- Se arregla una vez y se cierra a los siete: la madre lleva el trabajo,
    -- las hijas apuntan a ella y reciben su propio correo de cierre.
    madre_id        uuid references public.incidencias(id) on delete set null,

    -- ── Correos ─────────────────────────────────────────────────────────
    aviso_at        timestamptz,   -- el que te llega a ti
    cierre_at       timestamptz,   -- el que le llega al usuario

    actualizado_at  timestamptz not null default now()
);

-- Una incidencia por cada cosa que entró. El cron se dispara varias veces al
-- día y sin esto duplicaría la cola entera en cada vuelta.
create unique index if not exists incidencias_origen_uk
    on public.incidencias (origen, origen_id);

create index if not exists incidencias_estado_ix on public.incidencias (estado, creado_at desc);
create index if not exists incidencias_familia_ix on public.incidencias (familia, clase);
create index if not exists incidencias_pendiente_vb_ix
    on public.incidencias (aprobada_at) where estado = 'espera_vb';

create or replace function public.incidencias_touch() returns trigger
language plpgsql as $$
begin new.actualizado_at = now(); return new; end $$;

drop trigger if exists incidencias_touch_tg on public.incidencias;
create trigger incidencias_touch_tg before update on public.incidencias
    for each row execute function public.incidencias_touch();

-- ── Cerradura ───────────────────────────────────────────────────────────
-- La tabla lleva correos de usuarios y diagnósticos internos. Nadie la lee
-- desde el navegador: sólo el cron, que entra con service_role y se salta
-- RLS por diseño. RLS activo SIN políticas = nadie pasa, que es lo correcto
-- aquí. (Lección del 5-sep: revocar a `anon` no sirve si el permiso viene
-- de PUBLIC — por eso se revoca a PUBLIC.)
alter table public.incidencias enable row level security;
revoke all on public.incidencias from public, anon, authenticated;
grant all on public.incidencias to service_role;
