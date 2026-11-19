-- EL CRÉDITO DE CLAUDE, CICLO POR CICLO (18-nov-2026)
--
-- El plan de Anthropic da 200 USD al mes, que se renuevan el día 17 a las
-- 15:00 (hora de México). chat_claude.py anota aquí cada respuesta que
-- escribe Claude (tipo «respuesta», con su costo estimado) y el primer «sin
-- saldo» de cada ciclo (tipo «agotado»). La vista claude_ciclos lo resume:
-- cuándo empezó a contestar Claude, cuándo se acabó el crédito, cuántas
-- respuestas dio y cuánto costaron. El costo es el estimado con la tarifa de
-- Haiku 5.5, no la factura.
--
-- Sólo la clave de servicio la lee y la escribe: RLS encendido, sin políticas.

create table if not exists public.claude_uso (
    id          bigserial primary key,
    creado_at   timestamptz not null default now(),
    ciclo       date not null,                  -- el día 17 en que empezó el ciclo
    tipo        text not null check (tipo in ('respuesta', 'agotado')),
    ruta        text,                           -- buscar | basico | pro
    esfuerzo    text,
    fin         text,                           -- stop_reason de Claude
    usd         numeric(10, 5),
    entrada     integer,                        -- tokens de entrada (con caché)
    salida      integer,
    segundos    numeric(7, 1),
    peticion    text
);

create index if not exists claude_uso_ciclo_tipo on public.claude_uso (ciclo, tipo, creado_at);

alter table public.claude_uso enable row level security;
revoke all on public.claude_uso from anon, authenticated;

create or replace view public.claude_ciclos with (security_invoker = true) as
select
    ciclo,
    min(creado_at) filter (where tipo = 'respuesta')             as primera_respuesta,
    max(creado_at) filter (where tipo = 'respuesta')             as ultima_respuesta,
    min(creado_at) filter (where tipo = 'agotado')               as agotado,
    count(*)       filter (where tipo = 'respuesta')             as respuestas,
    round(coalesce(sum(usd) filter (where tipo = 'respuesta'), 0), 2) as usd_estimado
from public.claude_uso
group by ciclo
order by ciclo desc;

revoke all on public.claude_ciclos from anon, authenticated;
