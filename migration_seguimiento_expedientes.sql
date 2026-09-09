-- ──────────────────────────────────────────────────────────────────────
-- Migración: seguimiento de expedientes ante órganos jurisdiccionales
-- Vigilancia diaria de acuerdos en PJF, Ciudad de México y Querétaro.
-- ──────────────────────────────────────────────────────────────────────
--
-- Cómo correr esto:
--   1) Supabase Dashboard → SQL Editor → New query
--   2) Pegar este archivo completo y RUN
--   3) Verificar que las tablas aparezcan en Table Editor
--
-- Idempotente: se puede correr varias veces sin romper nada.
--
-- QUÉ NO TOCA: `public.expedientes` es la carpeta inteligente del abogado
-- —su archivo del despacho— y se queda como está. Esto es otra cosa: una
-- vigilancia de lo que hace el juzgado. Se vinculan por
-- `seg_expedientes_seguidos.expediente_id`, que es opcional, y no se funden.
--
-- LAS TRES REGLAS QUE ESTE ESQUEMA TIENE QUE SOSTENER:
--   1. Se revisa una vez al día, a las 9:10 hora de la Ciudad de México.
--   2. Sólo se escribe al abogado cuando hay una actuación NUEVA.
--   3. Si no se pudo revisar, hay que decirlo.
-- La tercera es la que manda en el diseño: obliga a que cada expediente deje
-- rastro de CADA día, tanto si se pudo como si no (`seg_revisiones`). Sin esa
-- fila, «no se pudo revisar» sería una opinión; con ella es un hecho que se
-- puede consultar.


-- ── 1. Catálogo de órganos jurisdiccionales ───────────────────────────
create table if not exists public.seg_organos (
    id               bigint generated always as identity primary key,
    jurisdiccion     text not null check (jurisdiccion in ('PJF','CDMX','QRO')),
    clave_externa    text not null,
    nombre           text not null,
    circuito_id      int,
    circuito_ordinal int,
    entidad          text,
    distrito         text,
    materia          text,
    familia          text,
    activo           boolean not null default true,
    vigencia_hasta   date,
    metadatos        jsonb not null default '{}',
    actualizado_en   timestamptz not null default now()
);

comment on column public.seg_organos.clave_externa is
    'La clave con la que el portal identifica al órgano. PJF: el CatOrganismoId que va en VerCaptura.aspx?organismo=… (p. ej. 293). CDMX: la clave del juzgado en el boletín. QRO: ciudad|juz|num.';
comment on column public.seg_organos.circuito_id is
    'Id INTERNO del circuito en el sistema del PJF (1..109, no correlativo). NO es el ordinal romano: el ordinal XXII de Querétaro tiene id interno 53. Confundirlos rompe la consulta en 21 de los 32 circuitos.';
comment on column public.seg_organos.vigencia_hasta is
    'Órganos extintos por la reforma judicial. El PJF los rotula en el propio nombre, "(… - 30/09/2024)". Se conservan porque un expediente antiguo sigue apuntando a ellos.';

create unique index if not exists ux_seg_organos_clave
    on public.seg_organos (jurisdiccion, clave_externa);
create index if not exists ix_seg_organos_busqueda
    on public.seg_organos using gin (to_tsvector('spanish', nombre));
create index if not exists ix_seg_organos_navegacion
    on public.seg_organos (jurisdiccion, circuito_id) where activo;


-- ── 2. Tipos de asunto ────────────────────────────────────────────────
-- Varían por familia de órgano: un Colegiado conoce de Amparo Directo y un
-- Juzgado de Distrito de Amparo Indirecto. Ofrecer la lista equivocada en el
-- alta es la primera forma de que el abogado dé de alta un expediente que
-- nunca va a encontrarse.
create table if not exists public.seg_tipos_asunto (
    id            bigint generated always as identity primary key,
    jurisdiccion  text not null,
    familia       text not null,
    clave_externa text not null,
    nombre        text not null,
    orden         int,
    unique (jurisdiccion, familia, clave_externa)
);


-- ── 3. El expediente seguido ──────────────────────────────────────────
create table if not exists public.seg_expedientes_seguidos (
    id            uuid primary key default gen_random_uuid(),
    user_id       uuid not null references auth.users(id) on delete cascade,
    expediente_id uuid,
    organo_id     bigint not null references public.seg_organos(id),
    numero        text not null,
    anio          int,
    tipo_asunto_clave        text,
    tipo_procedimiento_clave text,
    neun          text,
    alias         text not null,
    modo          text not null default 'automatico'
                  check (modo in ('automatico','asistido')),
    estado        text not null default 'activo'
                  check (estado in ('activo','pausado','archivado','requiere_atencion')),
    correo_aviso  text,
    linea_base_en timestamptz,
    ultima_revision_ok  timestamptz,
    ultima_actuacion_en date,
    fallos_consecutivos int not null default 0,
    escalado_en   timestamptz,
    creado_en     timestamptz not null default now()
);

comment on column public.seg_expedientes_seguidos.numero is
    'El número tal cual lo pide el portal: "71/2026". No se normaliza ni se parte, porque es lo que se manda en la petición.';
comment on column public.seg_expedientes_seguidos.linea_base_en is
    'Momento del alta. Todo lo anterior se guarda marcado como línea base y NO genera correo: sin esto, el primer día el abogado recibiría tres años de acuerdos de golpe y se daría de baja.';
comment on column public.seg_expedientes_seguidos.modo is
    'automatico: lo consulta el cron. asistido: el portal exige un reto que resuelve el abogado (hoy, Querétaro). El PJF entero puede pasar a asistido si el canario detecta que le pusieron CAPTCHA.';

-- El tipo de asunto forma parte de la identidad: un mismo "71/2026" puede
-- existir en el mismo órgano como Amparo Indirecto y como Causa Penal.
-- El coalesce es imprescindible: `null` no colisiona consigo mismo en un
-- índice único, así que sin él el mismo expediente se podría dar de alta
-- veinte veces.
create unique index if not exists ux_seg_seguidos_unico
    on public.seg_expedientes_seguidos
       (user_id, organo_id, numero, coalesce(tipo_asunto_clave,''));
create index if not exists ix_seg_seguidos_barrido
    on public.seg_expedientes_seguidos (modo, organo_id) where estado = 'activo';
create index if not exists ix_seg_seguidos_usuario
    on public.seg_expedientes_seguidos (user_id, estado);
create index if not exists ix_seg_seguidos_carpeta
    on public.seg_expedientes_seguidos (expediente_id) where expediente_id is not null;


-- ── 4. Actuaciones detectadas ─────────────────────────────────────────
create table if not exists public.seg_actuaciones (
    id             uuid primary key default gen_random_uuid(),
    seguimiento_id uuid not null references public.seg_expedientes_seguidos(id)
                        on delete cascade,
    user_id        uuid not null,
    huella_clave   char(64) not null,
    huella_texto   char(64) not null,
    simhash        bigint,
    cuaderno       text,
    fecha_auto     date,
    fecha_publicacion date,
    orden_en_lista int,
    titulo         text,
    resumen        text not null,
    texto_completo text,
    url_fuente     text,
    origen         text not null check (origen in
                     ('pjf_vercaptura','pjf_veracuerdo','cdmx_boletin','qro_asistido')),
    version        int not null default 1,
    reemplaza_a    uuid references public.seg_actuaciones(id),
    es_linea_base  boolean not null default false,
    detectada_en   timestamptz not null default now(),
    avisada_en     timestamptz
);

comment on column public.seg_actuaciones.huella_clave is
    'Identidad del acuerdo: sha256 de jurisdicción|órgano|número|neun|cuaderno|fecha_auto|resumen[:300]. Lleva fecha_auto y no fecha_publicacion porque la del auto es del juez y no cambia. Lleva cuaderno porque el principal y el incidente de suspensión pueden tener autos el mismo día.';
comment on column public.seg_actuaciones.orden_en_lista is
    'INFORMATIVO. Nunca entra en la identidad: el juzgado lo renumera al intercalar un acuerdo atrasado, y si formara parte de la huella, un día cualquiera todos los acuerdos parecerían nuevos.';
comment on column public.seg_actuaciones.simhash is
    'Desempate de reediciones. Si la huella_clave no existe pero hay una actuación del mismo cuaderno y fecha con distancia de Hamming <= 6 sobre 64, no es nueva: es la misma con la cabecera corregida.';
comment on column public.seg_actuaciones.es_linea_base is
    'Histórico traído en el alta. Nunca genera correo.';

create unique index if not exists ux_seg_actuaciones_identidad
    on public.seg_actuaciones (seguimiento_id, huella_clave, version);
create index if not exists ix_seg_actuaciones_cronologia
    on public.seg_actuaciones (seguimiento_id, fecha_auto desc, version desc);
create index if not exists ix_seg_actuaciones_pendientes
    on public.seg_actuaciones (seguimiento_id)
    where avisada_en is null and es_linea_base = false;
create index if not exists ix_seg_actuaciones_desempate
    on public.seg_actuaciones (seguimiento_id, cuaderno, fecha_auto);


-- ── 5. Bitácora de revisiones ─────────────────────────────────────────
-- El fundamento de la regla 3. Una fila por expediente, día e intento.
create table if not exists public.seg_revisiones (
    id             bigint generated always as identity primary key,
    corrida_id     uuid not null,
    seguimiento_id uuid not null references public.seg_expedientes_seguidos(id)
                        on delete cascade,
    user_id        uuid not null,
    fecha_local    date not null,
    intento        smallint not null default 1,
    iniciada_en    timestamptz not null default now(),
    terminada_en   timestamptz,
    duracion_ms    int,
    resultado      text not null check (resultado in (
                     'ok_sin_novedad','ok_con_novedad','inhabil','omitida_pausado',
                     'fallo_red','fallo_http','fallo_formato','fallo_no_encontrado',
                     'fallo_captcha','pendiente_abogado')),
    http_status    int,
    bytes          int,
    hash_respuesta char(64),
    n_actuaciones_vistas int,
    detalle        text,
    evidencia_ruta text
);

comment on column public.seg_revisiones.fecha_local is
    'Día natural en America/Mexico_City, no en UTC. Es el día del que habla el correo.';
comment on column public.seg_revisiones.hash_respuesta is
    'sha256 del cuerpo. Detecta el caso perverso del portal que responde 200 con contenido congelado durante días: mismo hash muchos días seguidos es sospecha, no tranquilidad.';
comment on column public.seg_revisiones.n_actuaciones_vistas is
    'Cuántas filas trajo la tabla. Si son MENOS de las que ya teníamos guardadas, el resultado no es ok_sin_novedad sino fallo_formato: ausencia de filas no es ausencia de novedad.';

create unique index if not exists ux_seg_revisiones_dia
    on public.seg_revisiones (seguimiento_id, fecha_local, intento);
create index if not exists ix_seg_revisiones_parte
    on public.seg_revisiones (fecha_local, resultado);
create index if not exists ix_seg_revisiones_corrida
    on public.seg_revisiones (corrida_id);


-- ── 6. Corridas del cron ──────────────────────────────────────────────
create table if not exists public.seg_corridas (
    id           uuid primary key default gen_random_uuid(),
    fecha_local  date not null,
    pase         smallint not null,
    disparo      text not null check (disparo in ('cron','manual','reintento')),
    iniciada_en  timestamptz not null default now(),
    terminada_en timestamptz,
    n_total      int,
    n_ok         int,
    n_novedad    int,
    n_fallo      int,
    n_pendiente  int,
    nota         text
);

comment on column public.seg_corridas.pase is
    '1 = 9:10, 2 = 9:40, 3 = 10:20, 4 = cierre de las 11:00. El índice único hace que un cron disparado dos veces no duplique la corrida.';

create unique index if not exists ux_seg_corridas_pase
    on public.seg_corridas (fecha_local, pase);


-- ── 7. Documentos fuente compartidos ──────────────────────────────────
-- El boletín de CDMX es uno para toda la cartera: se descarga una vez al día
-- y todos los expedientes de CDMX se resuelven contra él sin volver a la red.
create table if not exists public.seg_documentos_fuente (
    id             uuid primary key default gen_random_uuid(),
    jurisdiccion   text not null,
    fecha_local    date not null,
    id_externo     text,
    url            text not null,
    sha256         char(64),
    bytes          bigint,
    paginas        int,
    con_capa_texto boolean,
    estado         text not null default 'pendiente'
                   check (estado in ('pendiente','descargado','indexado','fallo')),
    ruta_storage   text,
    descargado_en  timestamptz,
    indexado_en    timestamptz
);

create unique index if not exists ux_seg_docs_dia
    on public.seg_documentos_fuente (jurisdiccion, fecha_local);


-- ── 8. Avisos enviados ────────────────────────────────────────────────
create table if not exists public.seg_avisos (
    id             uuid primary key default gen_random_uuid(),
    user_id        uuid not null,
    seguimiento_id uuid references public.seg_expedientes_seguidos(id) on delete cascade,
    tipo           text not null check (tipo in
                     ('actuacion','no_se_pudo','ultimo_aviso','escalado_david',
                      'recordatorio_asistido')),
    fecha_local    date not null,
    clave_idem     text not null,
    destinatario   text not null,
    asunto         text not null,
    resend_id      text,
    estado         text not null default 'encolado'
                   check (estado in ('encolado','enviado','rebotado','fallo')),
    creado_en      timestamptz not null default now(),
    enviado_en     timestamptz
);

comment on column public.seg_avisos.clave_idem is
    'Clave determinista, p. ej. actuacion:<seguimiento>:<huella_clave>:<version>. Si el worker se reinicia a mitad del envío, el correo no sale dos veces.';

create unique index if not exists ux_seg_avisos_idem
    on public.seg_avisos (clave_idem);
create index if not exists ix_seg_avisos_usuario
    on public.seg_avisos (user_id, fecha_local desc);


-- ── 9. Días inhábiles ─────────────────────────────────────────────────
create table if not exists public.seg_dias_inhabiles (
    jurisdiccion text not null,
    fecha        date not null,
    motivo       text,
    fuente       text,
    primary key (jurisdiccion, fecha)
);

comment on table public.seg_dias_inhabiles is
    'En día inhábil no se consulta y no se escribe, pero SÍ se deja fila en seg_revisiones con resultado=inhabil: cuesta poco y hace que el silencio de ese día tenga papel. No se pierde nada, porque el portal devuelve el histórico completo y lo publicado en el hueco aparece el primer día hábil siguiente con su fecha real.';


-- ── 10. Retos asistidos ───────────────────────────────────────────────
-- Donde el portal exige un reto, Iurexia llega hasta el borde y la persona lo
-- cruza. Aquí sólo vive el enlace preparado y el resultado que trajo.
create table if not exists public.seg_retos_captcha (
    id              uuid primary key default gen_random_uuid(),
    seguimiento_id  uuid not null references public.seg_expedientes_seguidos(id)
                         on delete cascade,
    user_id         uuid not null,
    fecha_local     date not null,
    token           text not null unique,
    url_preparada   text not null,
    estado          text not null default 'abierto'
                    check (estado in ('abierto','resuelto','caducado','cancelado')),
    creado_en       timestamptz not null default now(),
    expira_en       timestamptz not null,
    resuelto_en     timestamptz,
    bytes_recibidos int
);

create index if not exists ix_seg_retos_abiertos
    on public.seg_retos_captcha (user_id, fecha_local) where estado = 'abierto';


-- ══════════════════════════════════════════════════════════════════════
-- RLS
-- ══════════════════════════════════════════════════════════════════════
-- Lo del abogado es del abogado. El backend (service_role) omite RLS por
-- diseño y escribe todo; el frontend usa la anon key con la sesión del
-- abogado y sólo ve lo suyo.

alter table public.seg_organos              enable row level security;
alter table public.seg_tipos_asunto         enable row level security;
alter table public.seg_expedientes_seguidos enable row level security;
alter table public.seg_actuaciones          enable row level security;
alter table public.seg_revisiones           enable row level security;
alter table public.seg_retos_captcha        enable row level security;
alter table public.seg_avisos               enable row level security;
alter table public.seg_corridas             enable row level security;
alter table public.seg_documentos_fuente    enable row level security;
alter table public.seg_dias_inhabiles       enable row level security;

-- Catálogos: lectura para cualquier sesión autenticada, escritura sólo backend.
drop policy if exists seg_organos_lectura on public.seg_organos;
create policy seg_organos_lectura on public.seg_organos
    for select to authenticated using (true);

drop policy if exists seg_tipos_lectura on public.seg_tipos_asunto;
create policy seg_tipos_lectura on public.seg_tipos_asunto
    for select to authenticated using (true);

drop policy if exists seg_inhabiles_lectura on public.seg_dias_inhabiles;
create policy seg_inhabiles_lectura on public.seg_dias_inhabiles
    for select to authenticated using (true);

-- El seguimiento es del abogado: lo crea, lo edita y lo da de baja él.
drop policy if exists seg_seguidos_propios on public.seg_expedientes_seguidos;
create policy seg_seguidos_propios on public.seg_expedientes_seguidos
    for all to authenticated
    using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- Contenido y bitácora: el usuario LEE, el backend ESCRIBE.
drop policy if exists seg_actuaciones_lectura on public.seg_actuaciones;
create policy seg_actuaciones_lectura on public.seg_actuaciones
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists seg_revisiones_lectura on public.seg_revisiones;
create policy seg_revisiones_lectura on public.seg_revisiones
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists seg_avisos_lectura on public.seg_avisos;
create policy seg_avisos_lectura on public.seg_avisos
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists seg_retos_lectura on public.seg_retos_captcha;
create policy seg_retos_lectura on public.seg_retos_captcha
    for select to authenticated using (auth.uid() = user_id);

drop policy if exists seg_retos_resolver on public.seg_retos_captcha;
create policy seg_retos_resolver on public.seg_retos_captcha
    for update to authenticated
    using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- Corridas y documentos fuente son infraestructura, no son del usuario.
revoke all on public.seg_corridas           from anon, authenticated;
revoke all on public.seg_documentos_fuente  from anon, authenticated;
