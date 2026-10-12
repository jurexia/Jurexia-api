-- ═══ COBRO PROPORCIONAL DE CONSULTAS (oct-2026) ═══════════════════════════
-- David: «esto consumirá las consultas en proporción, sino el usuario con
-- expedientes completos rebasaría el costo en servicios que lo que paga por su
-- plan». La Bóveda y la Cronología cobran VARIAS consultas de una vez, y del
-- MISMO contador que el chat (`queries_used` / `consultas_recargadas`): no hay
-- monedero aparte.
--
-- `consume_query` no sirve para eso: cobra de una en una, y llamarla N veces
-- puede quedarse a medias (cobra 7 de 10 y se acaba el saldo). Aquí el cobro
-- es todo o nada, dentro de un FOR UPDATE.
--
-- Cada cobro deja un MOVIMIENTO con lo que salió del plan y lo que salió de la
-- recarga, para que la devolución regrese cada consulta a su sitio y nunca
-- devuelva más de lo cobrado (idempotente aunque el aviso de fallo llegue dos
-- veces).
--
-- Sólo el servidor (service_role) llama estas funciones; el usuario ve sus
-- movimientos, no los escribe.

create table if not exists public.consultas_movimientos (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references auth.users(id) on delete cascade,
  unidades     integer not null check (unidades > 0),
  de_plan      integer not null default 0 check (de_plan >= 0),
  de_recarga   integer not null default 0 check (de_recarga >= 0),
  motivo       text not null,
  devuelto_plan    integer not null default 0,
  devuelto_recarga integer not null default 0,
  creado_en    timestamptz not null default now(),
  devuelto_en  timestamptz
);
create index if not exists consultas_movimientos_usuario
  on public.consultas_movimientos (user_id, creado_en desc);

alter table public.consultas_movimientos enable row level security;
drop policy if exists "Movimientos propios, sólo lectura" on public.consultas_movimientos;
create policy "Movimientos propios, sólo lectura" on public.consultas_movimientos
  for select using (auth.uid() = user_id);


-- Cuántas consultas puede gastar hoy, con las mismas puertas que el chat.
create or replace function public.consultas_disponibles(p_user_id uuid)
returns jsonb
language plpgsql
stable
security definer
set search_path to 'public', 'auth'
as $function$
declare
  f public.user_profiles%rowtype;
  v_exigir boolean;
begin
  if exists (select 1 from public.blocked_users b where b.user_id = p_user_id) then
    return jsonb_build_object('puede', false, 'motivo', 'cuenta_bloqueada', 'disponibles', 0);
  end if;
  select * into f from public.user_profiles where id = p_user_id;
  if not found then
    return jsonb_build_object('puede', false, 'motivo', 'user_not_found', 'disponibles', 0);
  end if;
  if f.suspendido_at is not null then
    return jsonb_build_object('puede', false, 'motivo', 'suscripcion_suspendida', 'disponibles', 0);
  end if;
  select exigir_correo_verificado into v_exigir from public.ajustes_seguridad where id = true;
  if coalesce(v_exigir, false) and not public.correo_verificado(p_user_id) then
    return jsonb_build_object('puede', false, 'motivo', 'email_no_verificado', 'disponibles', 0);
  end if;
  return jsonb_build_object(
    'puede', true,
    'ilimitado', f.queries_limit = -1,
    'del_plan', case when f.queries_limit = -1 then null
                     else greatest(0, f.queries_limit - coalesce(f.queries_used, 0)) end,
    'recargadas', coalesce(f.consultas_recargadas, 0),
    'disponibles', case when f.queries_limit = -1 then 2147483647
                        else greatest(0, f.queries_limit - coalesce(f.queries_used, 0))
                             + coalesce(f.consultas_recargadas, 0) end,
    'subscription_type', f.subscription_type);
end;
$function$;


-- Cobra p_unidades de golpe: primero del plan, luego de la recarga. Todo o nada.
create or replace function public.consumir_consultas(p_user_id uuid, p_unidades integer, p_motivo text)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'auth'
as $function$
declare
  v_used    integer;
  v_limit   integer;
  v_extra   integer;
  v_susp    timestamptz;
  v_sub     text;
  v_exigir  boolean;
  v_libres  integer;
  v_plan    integer;
  v_rec     integer;
  v_mov     uuid;
begin
  if auth.uid() is not null and auth.uid() <> p_user_id then
    raise exception 'no autorizado: esa cuota no es tuya' using errcode = '42501';
  end if;
  if p_unidades is null or p_unidades <= 0 then
    raise exception 'unidades inválidas: %', p_unidades using errcode = '22023';
  end if;

  if exists (select 1 from public.blocked_users b where b.user_id = p_user_id) then
    return jsonb_build_object('allowed', false, 'error', 'cuenta_bloqueada', 'disponibles', 0);
  end if;

  select queries_used, queries_limit, coalesce(consultas_recargadas, 0), suspendido_at, subscription_type
    into v_used, v_limit, v_extra, v_susp, v_sub
    from public.user_profiles where id = p_user_id for update;
  if not found then
    return jsonb_build_object('allowed', false, 'error', 'user_not_found', 'disponibles', 0);
  end if;
  if v_susp is not null then
    return jsonb_build_object('allowed', false, 'error', 'suscripcion_suspendida', 'disponibles', 0);
  end if;
  select exigir_correo_verificado into v_exigir from public.ajustes_seguridad where id = true;
  if coalesce(v_exigir, false) and not public.correo_verificado(p_user_id) then
    return jsonb_build_object('allowed', false, 'error', 'email_no_verificado', 'disponibles', 0);
  end if;

  if v_limit = -1 then
    v_plan := p_unidades;
    v_rec := 0;
  else
    v_libres := greatest(0, v_limit - coalesce(v_used, 0));
    if v_libres + v_extra < p_unidades then
      return jsonb_build_object('allowed', false, 'error', 'sin_saldo',
        'disponibles', v_libres + v_extra, 'requeridas', p_unidades,
        'subscription_type', v_sub);
    end if;
    v_plan := least(v_libres, p_unidades);
    v_rec := p_unidades - v_plan;
  end if;

  update public.user_profiles
     set queries_used = coalesce(queries_used, 0) + v_plan,
         consultas_recargadas = coalesce(consultas_recargadas, 0) - v_rec,
         updated_at = now(),
         last_query_at = now()
   where id = p_user_id;

  insert into public.consultas_movimientos (user_id, unidades, de_plan, de_recarga, motivo)
  values (p_user_id, p_unidades, v_plan, v_rec, left(coalesce(p_motivo, ''), 300))
  returning id into v_mov;

  return jsonb_build_object('allowed', true, 'movimiento_id', v_mov,
    'cobradas', p_unidades, 'de_plan', v_plan, 'de_recarga', v_rec,
    'disponibles', case when v_limit = -1 then 2147483647
                        else greatest(0, v_limit - coalesce(v_used, 0) - v_plan) + v_extra - v_rec end,
    'subscription_type', v_sub);
end;
$function$;


-- Devuelve hasta p_unidades (todas si es nulo) de un movimiento, cada una a su
-- origen: primero a la recarga (lo comprado vale más que lo del plan, que se
-- reinicia cada mes) y luego al plan. Nunca más de lo que se cobró.
create or replace function public.devolver_consultas(p_movimiento_id uuid, p_unidades integer default null)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'auth'
as $function$
declare
  m        public.consultas_movimientos%rowtype;
  v_quedan integer;
  v_pedido integer;
  v_rec    integer;
  v_plan   integer;
begin
  select * into m from public.consultas_movimientos where id = p_movimiento_id for update;
  if not found then
    return jsonb_build_object('devueltas', 0, 'error', 'movimiento_no_encontrado');
  end if;
  if auth.uid() is not null and auth.uid() <> m.user_id then
    raise exception 'no autorizado' using errcode = '42501';
  end if;

  v_quedan := (m.de_plan - m.devuelto_plan) + (m.de_recarga - m.devuelto_recarga);
  v_pedido := least(v_quedan, coalesce(p_unidades, v_quedan));
  if v_pedido <= 0 then
    return jsonb_build_object('devueltas', 0, 'pendientes', v_quedan);
  end if;

  v_rec := least(v_pedido, m.de_recarga - m.devuelto_recarga);
  v_plan := v_pedido - v_rec;

  -- Lo del plan se devuelve sólo hasta donde el contador lo permite: si el mes
  -- ya se reinició, queries_used está en cero y no hay nada que restar.
  update public.user_profiles
     set queries_used = greatest(0, coalesce(queries_used, 0) - v_plan),
         consultas_recargadas = coalesce(consultas_recargadas, 0) + v_rec,
         updated_at = now()
   where id = m.user_id;

  update public.consultas_movimientos
     set devuelto_plan = devuelto_plan + v_plan,
         devuelto_recarga = devuelto_recarga + v_rec,
         devuelto_en = now()
   where id = m.id;

  return jsonb_build_object('devueltas', v_pedido, 'a_plan', v_plan, 'a_recarga', v_rec,
                            'pendientes', v_quedan - v_pedido);
end;
$function$;

revoke all on function public.consultas_disponibles(uuid) from public, anon, authenticated;
revoke all on function public.consumir_consultas(uuid, integer, text) from public, anon, authenticated;
revoke all on function public.devolver_consultas(uuid, integer) from public, anon, authenticated;
grant execute on function public.consultas_disponibles(uuid) to service_role;
grant execute on function public.consumir_consultas(uuid, integer, text) to service_role;
grant execute on function public.devolver_consultas(uuid, integer) to service_role;
