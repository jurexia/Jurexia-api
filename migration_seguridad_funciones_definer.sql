-- ════════════════════════════════════════════════════════════════════
-- Cierre de las funciones SECURITY DEFINER expuestas al público
-- Aplicado en producción el 4 de septiembre de 2026.
--
-- Se guarda aquí por si hay que reconstruir el proyecto: las migraciones
-- viven en Supabase, pero el porqué tiene que vivir en el repositorio.
--
--
-- QUÉ AVISÓ SUPABASE Y QUÉ ERA EN REALIDAD
--
-- El correo del 23 de agosto decía «Table publicly accessible — Row-Level
-- Security is not enabled». Al mirarlo, eso ya estaba resuelto: las 46
-- tablas del esquema público tienen RLS. Pero debajo había algo peor, que
-- ese aviso no menciona.
--
-- Veintiuna funciones SECURITY DEFINER tenían EXECUTE para `anon`. Una
-- función SECURITY DEFINER corre con los privilegios de quien la creó
-- —`postgres`— y por tanto SE SALTA RLS: da igual lo bien protegidas que
-- estén las tablas. Y `anon` es la clave que va incrustada en el JavaScript
-- de la web: la tiene cualquiera que abra el inspector del navegador.
--
-- Las tres peores:
--
--   admin_get_users()
--       devuelve los 2.331 perfiles con correo, nombre, plan, estado y
--       los identificadores de cliente y suscripción de Stripe.
--
--   change_plan_manual(p_email, p_plan)
--       pone a cualquier correo el plan que se le diga y le reinicia el
--       contador de consultas. No comprueba absolutamente nada.
--
--   update_user_subscription(p_email, tipo, limite, stripe_customer, stripe_sub)
--       lo mismo, y además reescribe el vínculo con Stripe.
--
-- Y detrás: downgrade_to_free, reset_user_queries, reset_monthly_queries,
-- reset_monthly_usage, increment_query_count, marcar_correo_verificado,
-- is_user_blocked, voz_contar, voz_sumar_caracteres,
-- medir_almacenamiento_expedientes, correo_verificado.
--
-- No se probó llamando al endpoint: el permiso está comprobado en el
-- catálogo con has_function_privilege('anon', oid, 'EXECUTE'), que es la
-- misma fuente que usa PostgREST para decidir.
--
--
-- POR QUÉ EL PRIMER REVOKE NO SIRVIÓ
--
-- PostgreSQL concede EXECUTE a PUBLIC en cuanto se crea una función.
-- Revocárselo a `anon` no quita nada, porque `anon` lo tiene por ser parte
-- de PUBLIC. Hay que quitárselo a PUBLIC y devolvérselo, uno a uno, a quien
-- lo necesita. Las dos funciones que ya estaban bien —devolver_consulta y
-- eliminar_cuenta_datos— estaban así exactamente.
--
--
-- QUIÉN LLAMA A QUÉ, comprobado en el código antes de tocar nada
--
--   Navegador, rol `authenticated`, en src/lib/supabase.ts:
--       consume_query, consume_draft, consume_sentencia_query,
--       get_quota_status, get_sentencia_quota_status
--   Servidor con clave de servicio (rutas /api y el backend de Render):
--       todas las demás
--   Nadie, desde ningún cliente:
--       change_plan_manual, update_user_subscription, downgrade_to_free,
--       reset_user_queries, reset_monthly_queries, reset_monthly_usage,
--       increment_query_count, marcar_correo_verificado, correo_verificado
-- ════════════════════════════════════════════════════════════════════


-- ── 1 · Sólo el servidor ─────────────────────────────────────────────
do $$
declare f text;
begin
  foreach f in array array[
    'admin_get_users()',
    'change_plan_manual(text, text)',
    'update_user_subscription(text, text, integer, text, text)',
    'downgrade_to_free(text)',
    'reset_user_queries(text)',
    'reset_monthly_queries()',
    'reset_monthly_usage(uuid)',
    'increment_query_count(uuid)',
    'is_user_blocked(uuid)',
    'marcar_correo_verificado()',
    'correo_verificado(uuid)',
    'medir_almacenamiento_expedientes()',
    'voz_contar(uuid, integer)',
    'voz_sumar_caracteres(uuid, integer)'
  ] loop
    execute format('revoke all on function public.%s from public, anon, authenticated', f);
    execute format('grant execute on function public.%s to service_role', f);
  end loop;
end $$;

-- ── 2 · Las cinco del navegador: sólo con sesión iniciada ────────────
do $$
declare f text;
begin
  foreach f in array array[
    'consume_query(uuid)', 'consume_draft(uuid)', 'consume_sentencia_query(uuid)',
    'get_quota_status(uuid)', 'get_sentencia_quota_status(uuid)'
  ] loop
    execute format('revoke all on function public.%s from public, anon', f);
    execute format('grant execute on function public.%s to authenticated, service_role', f);
  end loop;
end $$;

-- ── 3 · Los disparadores ─────────────────────────────────────────────
-- PostgREST no los deja llamar por RPC, así que no eran superficie de
-- ataque, pero el privilegio sobraba. Se comprobó antes, en una
-- transacción aparte, que un disparador SÍ se ejecuta aunque el rol que
-- hace el INSERT no tenga EXECUTE sobre su función: PostgreSQL comprueba
-- ese permiso al CREAR el disparador, no al dispararlo.
revoke all on function public.handle_new_user()            from public, anon, authenticated;
revoke all on function public.enforce_conversation_limit() from public, anon, authenticated;

-- ── 4 · search_path fijo en todas ────────────────────────────────────
-- Sin él, una función SECURITY DEFINER resuelve los nombres de tabla
-- contra el search_path de quien la llama, y quien la llama puede
-- anteponer un esquema suyo con una `user_profiles` falsa. Es la otra
-- mitad del mismo agujero.
alter function public.admin_get_users()                       set search_path = public, pg_temp;
alter function public.change_plan_manual(text, text)          set search_path = public, pg_temp;
alter function public.consume_draft(uuid)                     set search_path = public, pg_temp;
alter function public.consume_sentencia_query(uuid)           set search_path = public, pg_temp;
alter function public.downgrade_to_free(text)                 set search_path = public, pg_temp;
alter function public.enforce_conversation_limit()            set search_path = public, pg_temp;
alter function public.get_sentencia_quota_status(uuid)        set search_path = public, pg_temp;
alter function public.increment_query_count(uuid)             set search_path = public, pg_temp;
alter function public.is_user_blocked(uuid)                   set search_path = public, pg_temp;
alter function public.reset_monthly_queries()                 set search_path = public, pg_temp;
alter function public.reset_monthly_usage(uuid)               set search_path = public, pg_temp;
alter function public.reset_user_queries(text)                set search_path = public, pg_temp;
alter function public.update_user_subscription(text, text, integer, text, text) set search_path = public, pg_temp;
alter function public.auto_update_queries_limit()             set search_path = public, pg_temp;
alter function public.expedientes_touch()                     set search_path = public, pg_temp;
alter function public.noticias_touch_updated_at()             set search_path = public, pg_temp;
alter function public.update_conversation_updated_at()        set search_path = public, pg_temp;
alter function public.update_updated_at_column()              set search_path = public, pg_temp;

-- ── 5 · La cuota que se gasta es la de uno mismo ─────────────────────
-- Las cinco funciones de cuota reciben el uuid como parámetro y no
-- comprobaban que fuera el de quien llama: con una sesión gratuita y el
-- uuid de otro se le podía gastar el mes, o leerle el plan. La guarda
-- respeta al servidor: auth.uid() es nulo cuando no hay JWT de usuario,
-- que es como llama el backend con la clave de servicio.
--
-- Se reescriben desde su propia definición, insertando la guarda tras el
-- único BEGIN de cada una: no se toca ni una línea de la lógica de cobro.
do $$
declare
  f record;
  def text;
  guarda constant text := '
  -- La cuota es de cada quien. Nulo = llamada del servidor con la clave de
  -- servicio, que no se comprueba porque ya está autorizada por otra vía.
  if auth.uid() is not null and auth.uid() <> p_user_id then
    raise exception ''no autorizado: esa cuota no es tuya'' using errcode = ''42501'';
  end if;
';
begin
  for f in
    select p.oid, p.proname
    from pg_proc p join pg_namespace n on n.oid = p.pronamespace
    where n.nspname = 'public'
      and p.proname in ('consume_query','consume_draft','consume_sentencia_query',
                        'get_quota_status','get_sentencia_quota_status')
  loop
    def := pg_get_functiondef(f.oid);
    if position('auth.uid() <> p_user_id' in def) > 0 then continue; end if;
    def := overlay(def placing 'BEGIN' || guarda from position('BEGIN' in def) for 5);
    execute def;
  end loop;
end $$;


-- ── Estado después de aplicarlo ──────────────────────────────────────
--   tablas públicas ............................. 46
--   tablas sin RLS ............................... 0
--   funciones SECURITY DEFINER .................. 23
--   ejecutables por anon ......................... 0   (eran 21)
--   ejecutables por authenticated ................ 5   (las de cuota, con guarda)
--   funciones sin search_path fijo ............... 0   (eran 18)
--
-- LO QUE NO SE TOCÓ, y por qué:
--
--   19 tablas tienen RLS activo y CERO políticas. El linter lo marca como
--   INFO. Es la configuración correcta para ellas: con RLS activo y sin
--   políticas nadie que no sea la clave de servicio ve nada, y esas 19 se
--   leen únicamente desde rutas de servidor (app/api/*, lib/correo/*) o
--   desde el backend de Render. Se comprobó una por una.
--
--   auth_leaked_password_protection sigue apagado. Es un ajuste del panel
--   de Supabase (Authentication → Policies), no SQL: compara la contraseña
--   contra HaveIBeenPwned al registrarse. Hay que encenderlo a mano.
