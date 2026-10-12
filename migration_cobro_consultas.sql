-- ════════════════════════════════════════════════════════════════════
-- Cobro proporcional de consultas: N consultas de una vez, todo o nada
-- (12-oct-2026). La usan la Cronología del expediente y la Bóveda.
--
-- David: «esto consumirá las consultas en proporción, sino el usuario con
-- expedientes completos rebasaría el costo en servicios que lo que paga por
-- su plan».
--
-- POR QUÉ NO SE LLAMA consume_query N VECES. Cada llamada cobra una y
-- decide sola: si a la vigésima no queda saldo, las diecinueve anteriores ya
-- se cobraron y el expediente no se procesa. Aquí se mira el saldo entero
-- bajo el mismo cerrojo y se cobra todo o no se cobra nada.
--
-- EL MISMO CONTADOR QUE EL CHAT. Primero la cuota del plan
-- (`queries_used` contra `queries_limit`, que se reinicia cada mes) y
-- después las recargas (`consultas_recargadas`, que no caducan): el mismo
-- orden de `consume_query`. Y las mismas puertas, en el mismo orden:
-- bloqueo, suspensión por impago, correo sin verificar.
--
-- CADA COBRO DEJA FILA en `consultas_cobros` con cuánto salió del plan y
-- cuánto de la recarga. Es lo que permite devolver EXACTAMENTE lo cobrado
-- si el trabajo falla: `devolver_consulta` sólo sabe bajar `queries_used`, y
-- una recarga devuelta como cuota del plan caducaría a fin de mes.
--
-- SÓLO service_role. Son SECURITY DEFINER: con EXECUTE para PUBLIC
-- cualquiera con la clave pública cobraría o devolvería a quien quisiera.
-- Ver migration_seguridad_funciones_definer.sql.
-- ════════════════════════════════════════════════════════════════════

create table if not exists public.consultas_cobros (
    id           uuid primary key default gen_random_uuid(),
    user_id      uuid not null references auth.users(id) on delete cascade,
    unidades     integer not null check (unidades between 1 and 10000),
    de_plan      integer not null default 0 check (de_plan >= 0),
    de_recarga   integer not null default 0 check (de_recarga >= 0),
    ilimitado    boolean not null default false,
    motivo       text not null check (char_length(motivo) between 1 and 60),
    referencia   text,
    devueltas    integer not null default 0 check (devueltas >= 0),
    created_at   timestamptz not null default now(),
    devuelto_at  timestamptz,
    check (devueltas <= unidades)
);

create index if not exists consultas_cobros_user_idx
    on public.consultas_cobros (user_id, created_at desc);

alter table public.consultas_cobros enable row level security;

drop policy if exists "consultas_cobros: cada quien ve los suyos" on public.consultas_cobros;
create policy "consultas_cobros: cada quien ve los suyos"
    on public.consultas_cobros for select to authenticated
    using (user_id = auth.uid());


create or replace function public.consumir_consultas(
    p_user_id uuid, p_unidades integer, p_motivo text, p_referencia text default null)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'auth'
as $function$
declare
    v_used     integer;
    v_limit    integer;
    v_sub_type text;
    v_susp     timestamptz;
    v_extra    integer;
    v_exigir   boolean;
    v_bloq     timestamptz;
    v_bloq_why text;
    v_plan_disp integer;
    v_de_plan  integer;
    v_de_rec   integer;
    v_id       uuid;
begin
    if auth.uid() is not null and auth.uid() <> p_user_id then
        raise exception 'no autorizado: esa cuota no es tuya' using errcode = '42501';
    end if;
    if p_unidades is null or p_unidades < 1 or p_unidades > 10000 then
        return jsonb_build_object('allowed', false, 'error', 'unidades_invalidas');
    end if;

    -- EL BLOQUEO VA PRIMERO, como en consume_query: es la puerta.
    select blocked_at, reason into v_bloq, v_bloq_why
      from public.blocked_users where user_id = p_user_id;
    if v_bloq is not null then
        return jsonb_build_object('allowed', false, 'error', 'cuenta_bloqueada',
                                  'motivo', v_bloq_why, 'subscription_type', 'bloqueado');
    end if;

    select queries_used, queries_limit, subscription_type, suspendido_at,
           coalesce(consultas_recargadas, 0)
      into v_used, v_limit, v_sub_type, v_susp, v_extra
      from public.user_profiles
     where id = p_user_id
       for update;
    if not found then
        return jsonb_build_object('allowed', false, 'error', 'user_not_found');
    end if;

    if v_susp is not null then
        return jsonb_build_object('allowed', false, 'error', 'suscripcion_suspendida',
                                  'subscription_type', v_sub_type);
    end if;

    select exigir_correo_verificado into v_exigir
      from public.ajustes_seguridad where id = true;
    if coalesce(v_exigir, false) and not public.correo_verificado(p_user_id) then
        return jsonb_build_object('allowed', false, 'error', 'email_no_verificado',
                                  'subscription_type', v_sub_type);
    end if;

    -- Plan sin tope (-1): se deja constancia del cobro, sin descontar nada.
    if v_limit = -1 then
        insert into public.consultas_cobros (user_id, unidades, ilimitado, motivo, referencia)
        values (p_user_id, p_unidades, true, p_motivo, p_referencia)
        returning id into v_id;
        return jsonb_build_object('allowed', true, 'cobro_id', v_id, 'unidades', p_unidades,
                                  'de_plan', 0, 'de_recarga', 0, 'ilimitado', true,
                                  'restantes', null, 'subscription_type', v_sub_type);
    end if;

    v_plan_disp := greatest(coalesce(v_limit, 0) - coalesce(v_used, 0), 0);
    if v_plan_disp + v_extra < p_unidades then
        return jsonb_build_object('allowed', false, 'error', 'saldo_insuficiente',
                                  'necesarias', p_unidades,
                                  'disponibles', v_plan_disp + v_extra,
                                  'del_plan', v_plan_disp, 'recargadas', v_extra,
                                  'subscription_type', v_sub_type);
    end if;

    v_de_plan := least(p_unidades, v_plan_disp);
    v_de_rec  := p_unidades - v_de_plan;

    update public.user_profiles
       set queries_used         = coalesce(queries_used, 0) + v_de_plan,
           consultas_recargadas = coalesce(consultas_recargadas, 0) - v_de_rec,
           updated_at           = now(),
           last_query_at        = now()
     where id = p_user_id;

    insert into public.consultas_cobros (user_id, unidades, de_plan, de_recarga, motivo, referencia)
    values (p_user_id, p_unidades, v_de_plan, v_de_rec, p_motivo, p_referencia)
    returning id into v_id;

    return jsonb_build_object('allowed', true, 'cobro_id', v_id, 'unidades', p_unidades,
                              'de_plan', v_de_plan, 'de_recarga', v_de_rec, 'ilimitado', false,
                              'restantes', v_plan_disp + v_extra - p_unidades,
                              'subscription_type', v_sub_type);
end;
$function$;


-- Devuelve `p_unidades` de un cobro (null = todo lo que quede por devolver).
-- Primero la recarga —se cobró la última y no caduca—, después el plan.
-- Idempotente: lo ya devuelto no se vuelve a devolver.
create or replace function public.devolver_consultas(
    p_cobro_id uuid, p_unidades integer default null)
returns jsonb
language plpgsql
security definer
set search_path to 'public', 'auth'
as $function$
declare
    c           public.consultas_cobros%rowtype;
    v_pendiente integer;
    v_n         integer;
    v_ya_rec    integer;
    v_ya_plan   integer;
    v_rec       integer;
    v_plan      integer;
begin
    select * into c from public.consultas_cobros where id = p_cobro_id for update;
    if not found then
        return jsonb_build_object('devuelta', false, 'error', 'cobro_no_encontrado');
    end if;
    if auth.uid() is not null and auth.uid() <> c.user_id then
        raise exception 'no autorizado' using errcode = '42501';
    end if;

    v_pendiente := c.unidades - c.devueltas;
    v_n := least(coalesce(p_unidades, v_pendiente), v_pendiente);
    if v_n <= 0 then
        return jsonb_build_object('devuelta', false, 'error', 'nada_que_devolver',
                                  'devueltas', c.devueltas);
    end if;

    if not c.ilimitado then
        -- Lo ya devuelto salió primero de la recarga: se descuenta de ahí.
        v_ya_rec  := least(c.devueltas, c.de_recarga);
        v_ya_plan := c.devueltas - v_ya_rec;
        v_rec  := least(v_n, c.de_recarga - v_ya_rec);
        v_plan := least(v_n - v_rec, c.de_plan - v_ya_plan);

        update public.user_profiles
           set consultas_recargadas = coalesce(consultas_recargadas, 0) + v_rec,
               -- Si el mes se reinició entre el cobro y la devolución, el
               -- contador ya está en cero: no se deja en negativo.
               queries_used         = greatest(0, coalesce(queries_used, 0) - v_plan),
               updated_at           = now()
         where id = c.user_id;
    end if;

    update public.consultas_cobros
       set devueltas = devueltas + v_n,
           devuelto_at = now()
     where id = p_cobro_id;

    return jsonb_build_object('devuelta', true, 'unidades', v_n,
                              'devueltas', c.devueltas + v_n, 'de_unidades', c.unidades);
end;
$function$;

revoke all on function public.consumir_consultas(uuid, integer, text, text) from public, anon, authenticated;
revoke all on function public.devolver_consultas(uuid, integer) from public, anon, authenticated;
grant execute on function public.consumir_consultas(uuid, integer, text, text) to service_role;
grant execute on function public.devolver_consultas(uuid, integer) to service_role;
