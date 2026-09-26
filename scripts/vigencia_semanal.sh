#!/bin/zsh
# LA REGENERACIÓN SEMANAL DEL ÍNDICE DE VIGENCIA — el arranque.   (26-sep-2026)
#
# Lo lanza launchd cada domingo a las 03:30 (scripts/launchd/
# com.iurexia.vigencia-semanal.plist). Hace sólo lo que no puede hacer el
# Python desde dentro del worktree que va a mover:
#
#   1. el candado (lockf: si ya hay una corrida, sale con 75 y no toca nada);
#   2. fetch, y crea si hace falta el worktree DEDICADO
#      ($MAC/wt-vigencia-semanal, desprendido, sobre origin/main) con
#      `git worktree add`;
#   3. lo pone EXACTAMENTE en origin/main (reset --hard + clean) — sólo ése:
#      comprueba antes que es un worktree enlazado de este repo y no el
#      checkout principal ni el de otra sesión, que comparten el .git;
#   4. corre scripts/vigencia_semanal.py DE ESE worktree, o sea, la versión de
#      origin/main y no la de quien instaló el LaunchAgent.
#
# Todo va a ~/Library/Logs/iurexia/vigencia-semanal.log. Sale ≠ 0 ante
# cualquier fallo y entonces lo avisa también con una notificación de macOS.
#
# A mano: `zsh scripts/vigencia_semanal.sh` (con commit y push) o
#         `zsh scripts/vigencia_semanal.sh --seco` (todo menos commit y push).
# Lo que se le pase se lo pasa al Python.

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

  [[ -x "$PY" ]] || { echo "$t ✗ no está el Python del .venv: $PY"; return 1 }
  [[ -d "$REPO/.git" || -f "$REPO/.git" ]] || { echo "$t ✗ no está el repo: $REPO"; return 1 }
  [[ "${WT:A}" != "${REPO:A}" ]] || { echo "$t ✗ el worktree dedicado no puede ser el checkout principal"; return 1 }

  # 2. fetch y, la primera vez, el worktree
  # launchd lo corre al despertar si la Mac dormía a las 03:30, y la red tarda
  # en volver: tres intentos, con medio minuto entre uno y otro.
  local intento
  for intento in 1 2 3; do
    git -C "$REPO" fetch -q origin && break
    (( intento == 3 )) && { echo "$t ✗ git fetch falló tres veces (¿sin red?)"; return 5 }
    echo "$t   git fetch falló; otro intento en 30 s"
    sleep 30
  done
  if [[ ! -e "$WT" ]]; then
    echo "$t   creo el worktree dedicado $WT sobre origin/main"
    git -C "$REPO" worktree add -q --detach "$WT" origin/main || { echo "$t ✗ git worktree add falló"; return 5 }
  fi

  # 3. ¿es de verdad un worktree enlazado de ESTE repo, y no otra cosa?
  local top gd gcd rgcd
  top="$(git -C "$WT" rev-parse --show-toplevel 2>/dev/null)" || { echo "$t ✗ $WT no es un checkout de git"; return 5 }
  gd="$(git -C "$WT" rev-parse --absolute-git-dir)"
  gcd="$(cd "$WT" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  rgcd="$(cd "$REPO" && cd "$(git rev-parse --git-common-dir)" && pwd -P)"
  if [[ "${top:A}" != "${WT:A}" || "${gd:A}" == "${gcd:A}" || "${gcd:A}" != "${rgcd:A}" ]]; then
    echo "$t ✗ $WT no es un worktree enlazado de $REPO (toplevel $top, git-dir $gd, común $gcd); no lo toco"
    return 5
  fi
  git -C "$WT" rebase --abort >/dev/null 2>&1   # por si una corrida murió a media rebase
  git -C "$WT" reset -q --hard || return 5
  git -C "$WT" checkout -q --detach origin/main || { echo "$t ✗ checkout de origin/main falló"; return 5 }
  git -C "$WT" reset -q --hard origin/main || return 5
  git -C "$WT" clean -qfd || return 5
  echo "$t   worktree en origin/main: $(git -C "$WT" log -1 --format='%h %s' | cut -c1-110)"

  # 4. el trabajo, con el código de origin/main
  "$PY" "$WT/scripts/vigencia_semanal.py" --worktree "$WT" --dir "$DIR" --env "$ENVF" --sin-log "$@"
  local rc=$?
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ══ fin (código $rc) ══"
  return $rc
}

vigencia_semanal "$@"
exit $?
