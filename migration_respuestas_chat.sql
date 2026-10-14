-- ════════════════════════════════════════════════════════════════════
-- respuestas_chat: la copia que la API guarda de cada respuesta del chat
-- Escrita el 14-oct-2026. NO APLICADA: espera el visto bueno de David.
--
-- POR QUÉ
--
-- El historial del chat sólo lo guardaba el navegador, al terminar el
-- stream. Roberto perdió el 13/14-oct tres respuestas que el servidor
-- terminó sin error (23:07, 23:14 y 05:05 UTC): su navegador las leyó
-- hasta el final y no las guardó, y la consulta siguiente borró esas
-- conversaciones por vacías. Ver respuesta_guardada.py.
--
-- CÓMO SE USA
--
--   · La API (service_role) abre la fila al empezar el stream y la cierra
--     al terminar con la respuesta tal como la guardaría el navegador.
--   · El navegador la RECLAMA antes de guardar en `messages`:
--         update respuestas_chat set guardada_at = now()
--          where peticion_id = $1 and guardada_at is null
--      returning peticion_id;
--     Sólo uno gana; sólo quien gana inserta los mensajes. Así la pestaña
--     que terminó bien y la recuperación nunca guardan dos veces lo mismo.
--   · Al abrir el chat (y al volver a la pestaña) el navegador recupera lo
--     terminado hace más de un minuto que nadie reclamó.
--
-- SEGURIDAD
--
--   · Sin funciones nuevas: todo con RLS. El dueño lo pone la API con el
--     usuario DEL TOKEN, nunca el del cuerpo.
--   · `authenticated` sólo lee sus filas y sólo puede escribir UNA columna,
--     `guardada_at`, de sus filas. `anon`, nada.
--   · `conversation_id` va sin llave foránea a propósito: la conversación
--     puede no existir ya (la borró la limpieza de vacías) y la recuperación
--     la vuelve a crear con ese mismo id.
-- ════════════════════════════════════════════════════════════════════

create table if not exists public.respuestas_chat (
    peticion_id     uuid primary key,
    user_id         uuid not null references auth.users(id) on delete cascade,
    conversation_id uuid,
    expediente_id   uuid,
    estado_mx       text,
    pregunta        text not null default '',
    respuesta       text,
    estado          text not null default 'en_curso'
                    check (estado in ('en_curso', 'terminada', 'interrumpida')),
    creada_at       timestamptz not null default now(),
    terminada_at    timestamptz,
    guardada_at     timestamptz
);

comment on table public.respuestas_chat is
    'Copia de cada respuesta del chat que escribe la API (respuesta_guardada.py); el navegador la reclama (guardada_at) y la recupera si no alcanzó a guardarla.';

-- Lo único que se consulta: lo pendiente de un usuario, lo más nuevo primero.
create index if not exists respuestas_chat_pendientes
    on public.respuestas_chat (user_id, creada_at desc)
    where guardada_at is null;

alter table public.respuestas_chat enable row level security;

revoke all on table public.respuestas_chat from anon, authenticated;
grant select on table public.respuestas_chat to authenticated;
grant update (guardada_at) on table public.respuestas_chat to authenticated;

drop policy if exists "Cada quien lee sus respuestas" on public.respuestas_chat;
create policy "Cada quien lee sus respuestas" on public.respuestas_chat
    for select to authenticated
    using ((select auth.uid()) = user_id);

drop policy if exists "Cada quien reclama sus respuestas" on public.respuestas_chat;
create policy "Cada quien reclama sus respuestas" on public.respuestas_chat
    for update to authenticated
    using ((select auth.uid()) = user_id)
    with check ((select auth.uid()) = user_id);

-- Lo de más de 30 días ya no se recupera (el navegador sólo mira 7): fuera,
-- cada madrugada (10:15 UTC = 4:15 en CDMX).
select cron.schedule(
    'purgar-respuestas-chat',
    '15 10 * * *',
    $$delete from public.respuestas_chat where creada_at < now() - interval '30 days'$$
);
