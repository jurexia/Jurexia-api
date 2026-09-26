#!/bin/zsh
# LA REGENERACIÓN SEMANAL DEL ÍNDICE DE VIGENCIA — el arranque.   (26-sep-2026)
#
# Lo lanza launchd cada domingo a las 03:30: el plist
# (scripts/launchd/com.iurexia.vigencia-semanal.plist) lleva dentro un lanzador
# que saca ESTE guion de origin/main con `git show` y lo corre, así que no
# depende de que exista ningún checkout con él. Hace sólo lo que no puede hacer
# el Python desde dentro del worktree que va a mover:
#
#   1. el candado (lockf: si ya hay una corrida, sale con 75 y no toca nada);
#   2. fetch, y crea si hace falta el worktree DEDICADO
#      ($MAC/wt-vigencia-semanal, desprendido, sobre origin/main) con
#      `git worktree add`, y le deja su MARCA en la carpeta de administración
#      (.git/worktrees/wt-vigencia-semanal/vigencia-semanal.marca);
#   3. lo pone EXACTAMENTE en origin/main (reset --hard + clean) — sólo ése:
#      comprueba antes que es un worktree enlazado de este repo, que no es el
#      checkout principal y que lleva la marca; el de otra sesión, o uno que se
#      llame igual pero no lo creó este guion, no se toca. Y le enlaza el .env
#      del repo ($WT/.env → $ENVF; está en .gitignore, `clean` no lo borra):
#      el worktree es HERMANO del repo, y main.py, que importan las pruebas, no
#      encontraría ningún .env hacia arriba;
#   4. corre scripts/vigencia_semanal.py DE ESE worktree, o sea, la versión de
#      origin/main y no la de quien instaló el LaunchAgent.
#
# Todo va a ~/Library/Logs/iurexia/vigencia-semanal.log. Sale ≠ 0 ante
# cualquier fallo y entonces lo avisa también con una notificación de macOS
# (lanzado por launchd avisa el lanzador, que lo corre con VIGENCIA_SIN_AVISO).
#
# A mano: `zsh scripts/vigencia_semanal.sh`            (con commit y push)
#         `zsh scripts/vigencia_semanal.sh --seco`     (todo menos commit y push)
#         `zsh scripts/vigencia_semanal.sh --preparar` (sólo los pasos 1-3: deja
#                                        listo el worktree; lo usa la instalación,
#                                        que lo escribe «preparar», sin guiones)
#
# INSTALAR y QUITAR el LaunchAgent: un comando cada uno, en el comentario de
# scripts/launchd/com.iurexia.vigencia-semanal.plist.
# Lo demás que se le pase se lo pasa al Python.

setopt NO_UNSET PIPE_FAIL
# Dentro de una función, $0 es el nombre de la función (FUNCTION_ARGZERO):
# la ruta del guion se toma aquí, fuera.
VIGENCIA_SELF="${0:A}"

# Todo dentro de una función: zsh lee el guion entero antes de ejecutar, y el
# reset de abajo puede reescribir este mismo archivo mientras corre.
vigencia_semanal() {
  local MAC="${VIGENCIA_MAC:-$HOME/Documents/IUREXIA-MAC}"
  local REPO="${VIGENCIA_REPO:-$MAC/jurexia-api-git}"
  local WT="${VIGENCIA_WT:-$MAC/wt-vigencia-semanal}"
  local DIR="${VIGENCIA_DIR:-$MAC/reingesta/vigencia}"
  local PY="${VIGENCIA_PY:-$REPO/.venv/bin/python}"
  local ENVF="${VIGENCIA_ENV:-$REPO/.env}"
  local LOG="${VIGENCIA_LOG:-$HOME/Library/Logs/iurexia/vigencia-semanal.log}"
  local MARCA="vigencia-semanal.marca"
  export PATH="$REPO/.venv/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
  export PYTHONIOENCODING=utf-8 LANG="${LANG:-es_MX.UTF-8}"

  mkdir -p "${LOG:h}" "$DIR" || return 1

  # 1. el candado: se vuelve a lanzar a sí mismo bajo lockf (que lo suelta
  #    solo si el proceso muere). -t 0: si está tomado, no espera.
  if [[ -z "${VIGENCIA_CON_CANDADO:-}" ]]; then
    if [[ -t 1 ]]; then
      exec > >(tee -a "$LOG") 2>&1
    else
      exec >>"$LOG" 2>&1
    fi
    export VIGENCIA_CON_CANDADO=1
    /usr/bin/lockf -t 0 "$DIR/vigencia-semanal.sh.lock" /bin/zsh "$VIGENCIA_SELF" "$@"
    local rc=$?
    if (( rc == 75 )); then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ ya hay otra corrida de la vigencia semanal (candado tomado)"
    fi
    if (( rc != 0 )) && [[ -z "${VIGENCIA_SIN_AVISO:-}" ]]; then
      /usr/bin/osascript -e "display notification \"Salió con código $rc. Revisa ~/Library/Logs/iurexia/vigencia-semanal.log\" with title \"Iurexia · vigencia semanal\"" >/dev/null 2>&1 || true
    fi
    return $rc
  fi

  local t="[$(date '+%Y-%m-%d %H:%M:%S')]"
  echo ""
  echo "$t ══ vigencia semanal · arranque (pid $$) ══"

  local preparar=""
  [[ "${1:-}" == (--|)preparar ]] && { preparar=1; shift }   # sin guiones: así lo escribe el plist

  [[ -x "$PY" ]] || { echo "$t ✗ no está el Python del .venv: $PY"; return 1 }
  [[ -f "$ENVF" && -r "$ENVF" ]] || { echo "$t ✗ no está el .env: $ENVF (sin él no hay Qdrant ni pasan las pruebas)"; return 1 }
  [[ -d "$REPO/.git" || -f "$REPO/.git" ]] || { echo "$t ✗ no está el repo: $REPO"; return 1 }
  [[ "${WT:A}" != "${REPO:A}" ]] || { echo "$t ✗ el worktree dedicado no puede ser el checkout principal"; return 1 }

  # 2. fetch y, la primera vez, el worktree
  # launchd lo corre al despertar si la Mac dormía a las 03:30, y la red tarda
  # en volver: tres intentos, con medio minuto entre uno y otro.
  local intento
  for intento in 1 2 3; do
    git -C "$REPO" fetch -q origin && break
    (( intento == 3 )) && { echo "$t ✗ git fetch falló tres veces (¿sin red?)"; return 5 }
    echo "$t   git fetch falló; otro intento en ${VIGENCIA_ESPERA_RED:-30} s"
    sleep "${VIGENCIA_ESPERA_RED:-30}"
  done
  if [[ ! -e "$WT" ]]; then
    echo "$t   creo el worktree dedicado $WT sobre origin/main"
    git -C "$REPO" worktree add -q --detach "$WT" origin/main || {
      echo "$t ✗ git worktree add falló (si $WT se borró a mano, \`git -C $REPO worktree prune\` lo desregistra)"
      return 5
    }
    local gdn
    gdn="$(git -C "$WT" rev-parse --absolute-git-dir)" || return 5
    print -r -- "creado por scripts/vigencia_semanal.sh el $(date '+%Y-%m-%d %H:%M:%S') desde $REPO" \
      >| "$gdn/$MARCA" || { echo "$t ✗ no pude dejar la marca en $gdn"; return 5 }
  fi

  # 3. ¿es de verdad EL worktree dedicado de ESTE repo, y no otra cosa?
  local top gd gcd rgcd
  top="$(git -C "$WT" rev-parse --show-toplevel 2>/dev/null)" || { echo "$t ✗ $WT no es un checkout de git"; return 5 }
  gd="$(git -C "$WT" rev-parse --absolute-git-dir)"
  gcd="$(cd "$WT" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  rgcd="$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  if [[ "${top:A}" != "${WT:A}" || "${gd:A}" == "${gcd:A}" || "${gcd:A}" != "${rgcd:A}" ]]; then
    echo "$t ✗ $WT no es un worktree enlazado de $REPO (toplevel $top, git-dir $gd, común $gcd); no lo toco"
    return 5
  fi
  if [[ ! -f "$gd/$MARCA" ]]; then
    echo "$t ✗ $WT es un worktree de $REPO pero no lleva la marca del dedicado ($gd/$MARCA): no lo creó"
    echo "$t   este guion, así que no lo reseteo. Si de verdad es el dedicado y no guarda nada, bórralo con"
    echo "$t   \`git -C $REPO worktree remove --force $WT\` y la próxima corrida lo crea con su marca."
    return 5
  fi
  git -C "$WT" rebase --abort >/dev/null 2>&1   # por si una corrida murió a media rebase
  git -C "$WT" reset -q --hard || return 5
  git -C "$WT" checkout -q --detach origin/main || { echo "$t ✗ checkout de origin/main falló"; return 5 }
  git -C "$WT" reset -q --hard origin/main || return 5
  git -C "$WT" clean -qfd || return 5
  # El .env del repo, enlazado (nunca copiado: los secretos no se duplican y
  # .env está en .gitignore). Si alguien dejó ahí un .env de verdad, se respeta.
  if [[ -L "$WT/.env" || ! -e "$WT/.env" ]]; then
    ln -sfn "$ENVF" "$WT/.env" || { echo "$t ✗ no pude enlazar $WT/.env → $ENVF"; return 5 }
  fi
  echo "$t   worktree en origin/main: $(git -C "$WT" log -1 --format='%h %s' | cut -c1-110)"

  if [[ -n "$preparar" ]]; then
    [[ -f "$WT/scripts/vigencia_semanal.py" ]] || { echo "$t ✗ origin/main aún no trae scripts/vigencia_semanal.py"; return 1 }
    echo "$t ✓ listo: worktree dedicado $WT (con su marca), .env enlazado, Python $PY"
    return 0
  fi

  # 4. el trabajo, con el código de origin/main
  "$PY" "$WT/scripts/vigencia_semanal.py" --worktree "$WT" --dir "$DIR" --env "$ENVF" --sin-log "$@"
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ══ fin (código $rc) ══"
  return $rc
}

vigencia_semanal "$@"
exit $?
