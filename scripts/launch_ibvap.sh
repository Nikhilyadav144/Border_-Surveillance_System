#!/bin/bash
# =====================================================================
# IBVAP launcher
# =====================================================================
#
# Starts the whole platform with one double-click and no typing.
#
# WHAT THIS REPLACES
# ------------------
# The old start.sh ran two servers: uvicorn on :8000 and the Vite dev
# server on :5173. Two processes means two things that can be stale, and
# on this project a stale uvicorn has already produced two separate "the
# app is broken" reports that were really "the old server is still
# running and does not have the new code". uvicorn does not reload on its
# own.
#
# So this script does two things the old one did not:
#
#   1. It kills any IBVAP server that is already running, every time,
#      before starting a new one. A stale server can no longer survive a
#      restart, which removes that whole class of bug.
#
#   2. It starts ONE process. FastAPI now serves the compiled dashboard
#      itself, so there is no Vite, no second port, and no CORS.
#
# It is also loud about failure. A double-clicked app has nowhere to
# print, so anything that goes wrong is shown in a macOS dialog and
# written to logs/launcher.log, rather than the window simply never
# opening.
#
# Usage:
#   ./scripts/launch_ibvap.sh [project-directory]
#
# The IBVAP.app bundle calls this with the project directory it found.
# =====================================================================

set -u

PORT=8000
URL="http://127.0.0.1:${PORT}"
HEALTH="${URL}/api/health"
START_TIMEOUT=120          # seconds to wait for the API to answer

# ---------------------------------------------------------------------
# Locate the project
# ---------------------------------------------------------------------
# Either passed in by the .app stub, or inferred from where this script
# lives (it sits in <project>/scripts/).

if [ "$#" -ge 1 ] && [ -n "${1:-}" ]; then
    PROJECT="$1"
else
    PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

LOG_DIR="$PROJECT/logs"
LOG="$LOG_DIR/launcher.log"
SERVER_LOG="$LOG_DIR/server.log"

mkdir -p "$LOG_DIR" 2>/dev/null

say() {
    printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG" 2>/dev/null
    printf '%s  %s\n' "$(date '+%H:%M:%S')" "$*"
}

# A double-clicked app has no terminal, so failures have to be shown.
# osascript is given the text through an argument rather than spliced
# into the script source, so quotes and newlines in a message cannot
# break the AppleScript or run as code.
alert() {
    local title="$1" message="$2"
    osascript \
        -e 'on run {t, m}' \
        -e 'display alert t message m as critical' \
        -e 'end run' \
        -- "$title" "$message" >/dev/null 2>&1
}

die() {
    say "FAILED: $1"
    alert "IBVAP could not start" "$1

Details are in:
$LOG"
    exit 1
}

say "==================== launching ===================="
say "project: $PROJECT"

# ---------------------------------------------------------------------
# 1. Check the project is really there and really complete
# ---------------------------------------------------------------------

[ -f "$PROJECT/api/server.py" ] || die \
"This does not look like the IBVAP project folder:

$PROJECT

Expected to find api/server.py inside it. If you moved the project, move
IBVAP.app next to it as well, or put the folder back on the Desktop."

# WHY THE INTERPRETER AND NOT `activate`
# -------------------------------------
# A virtualenv writes its own absolute path into `activate` and into the
# shebang of every command in .venv/bin. Move the project folder and all
# of those point at a Python that is no longer there - which is exactly
# what happened to this project once already, and the error it produced
# said nothing about the real cause.
#
# `.venv/bin/python` is a symlink to the system Python, which does not
# move, and Python works out where its own site-packages live from the
# path it was invoked by. So the launcher runs the interpreter directly
# and never sources `activate` at all. Moving the folder cannot break it.

VENV_PY="$PROJECT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$PROJECT/.venv/bin/python3"
[ -x "$VENV_PY" ] || die \
"The Python environment is missing or incomplete.

Expected an interpreter at: $PROJECT/.venv/bin/python

Recreate it from a Terminal in the project folder:

  python3 -m venv .venv
  source .venv/bin/activate
  pip install -r requirements.txt"

# Fail early and clearly if the dependencies are not installed, rather
# than letting the server die later with a traceback nobody sees.
if ! "$VENV_PY" -c "import uvicorn, fastapi" 2>/dev/null; then
    MISSING="$("$VENV_PY" -c "import uvicorn, fastapi" 2>&1 | tail -3)"
    die "The Python environment has no packages installed in it.

$MISSING

Install them from a Terminal in the project folder:

  source .venv/bin/activate
  pip install -r requirements.txt"
fi

[ -f "$PROJECT/frontend/dist/index.html" ] || die \
"The dashboard has not been built.

Expected: $PROJECT/frontend/dist/index.html

Build it from a Terminal:

  cd \"$PROJECT/frontend\"
  npm install
  npm run build"

cd "$PROJECT" || die "Could not enter $PROJECT"

# ---------------------------------------------------------------------
# 2. Stop anything left over from last time
# ---------------------------------------------------------------------
# This is the step the old launcher was missing. Without it, the second
# launch quietly does nothing (the port is taken), the browser opens on
# the OLD server, and every symptom points at the code instead of at the
# process.

# Finding the old server by command line is the only option - it may have
# been started by hand, from the old start.sh, or by a previous launch -
# but a bare `pkill -f "uvicorn api.server"` is too blunt. That pattern
# also matches any SHELL whose command line happens to contain the text:
# a Terminal window where somebody typed the uvicorn command, or the
# script that started it. Killing those instead of the server is a real
# failure, and it was caught in testing rather than theory.
#
# So every match is checked: only a process that is actually running
# python/uvicorn is a server. A shell that merely mentions it is left
# alone.

server_pids() {
    local pid comm
    for pid in $(pgrep -f "uvicorn api\.server" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        comm="$(ps -p "$pid" -o comm= 2>/dev/null)"
        case "${comm##*/}" in
            uvicorn*|python*|Python*) printf '%s ' "$pid" ;;
        esac
    done
}

STALE="$(server_pids)"
if [ -n "${STALE// /}" ]; then
    say "a previous IBVAP server is running (pid ${STALE% }) - stopping it"
    # shellcheck disable=SC2086
    kill $STALE 2>/dev/null
    # Give it a moment to release the port, then insist.
    for _ in 1 2 3 4 5 6 7 8 9 10; do
        STALE="$(server_pids)"
        [ -z "${STALE// /}" ] && break
        sleep 0.5
    done
    STALE="$(server_pids)"
    if [ -n "${STALE// /}" ]; then
        say "it did not stop politely - forcing"
        # shellcheck disable=SC2086
        kill -9 $STALE 2>/dev/null
        sleep 1
    fi
fi

# WHO IS LISTENING ON THE PORT
# ---------------------------
# `lsof -ti tcp:8000` is the obvious command and it is wrong. It matches
# ANY socket involving port 8000, which includes the browser's own
# connection TO the dashboard. So an open browser tab made the launcher
# announce that "Google Chrome Helper" was occupying the port and refuse
# to start. Chrome was a client, not a server.
#
# -sTCP:LISTEN restricts this to processes actually accepting
# connections, which is the only thing that can genuinely block a bind.
# -nP skips host and port name lookups, which makes it quicker too.

listeners() {
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null
}

# A process disappearing and its socket being released are not the same
# instant. Checking the port immediately after the kill can therefore see
# the dying server still holding it and wrongly report a foreign program,
# so the port is given a few seconds to come free before that conclusion
# is drawn.
for _ in $(seq 1 20); do
    [ -z "$(listeners)" ] && break
    sleep 0.5
done

# Still held after that, and it is not one of ours: something else owns
# this port. Name it, rather than letting uvicorn fail with an opaque
# "address already in use" inside a log nobody will open.
HOLDER="$(listeners | head -1)"
if [ -n "$HOLDER" ]; then
    HOLDER_NAME="$(ps -p "$HOLDER" -o comm= 2>/dev/null)"
    die "Port ${PORT} is already being used by another program:

  ${HOLDER_NAME:-unknown} (process $HOLDER)

Quit that program and launch IBVAP again."
fi

# ---------------------------------------------------------------------
# 3. Start the one and only server
# ---------------------------------------------------------------------

# Launched from the Dock there is no window yet and the server takes a
# few seconds to come up, so without this the app looks like it did
# nothing. A notification says it is working before the browser appears.
notify() {
    osascript \
        -e 'on run {t, m}' \
        -e 'display notification m with title t' \
        -e 'end run' \
        -- "$1" "$2" >/dev/null 2>&1
}
notify "IBVAP" "Starting the surveillance platform..."

say "starting the IBVAP server on port ${PORT}"
say "interpreter: $VENV_PY"
: > "$SERVER_LOG"
"$VENV_PY" -m uvicorn api.server:app --host 127.0.0.1 --port "$PORT" \
    >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

# Quitting the app from the Dock sends a signal here. Take the server
# down with us, so closing IBVAP actually closes IBVAP and the next
# launch starts clean.
shutdown() {
    say "shutting down (server pid $SERVER_PID)"
    kill "$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    say "stopped"
    exit 0
}
trap shutdown INT TERM HUP

# ---------------------------------------------------------------------
# 4. Wait until it can actually answer
# ---------------------------------------------------------------------
# Opening the browser after a fixed `sleep 3` is a guess. On a cold start
# the imports alone (torch, ultralytics, opencv) take longer than that,
# and the browser lands on "cannot connect" for a server that was about
# to be fine. Ask the health endpoint instead, and only open once it
# answers.

say "waiting for the API to come up (up to ${START_TIMEOUT}s)"
READY=""
WAITED=0
while [ "$WAITED" -lt "$START_TIMEOUT" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        TAIL="$(tail -20 "$SERVER_LOG" 2>/dev/null)"
        die "The server started and then stopped.

Last lines of its log:

$TAIL"
    fi
    # The health check alone is not proof: a previous server that has not
    # finished dying could answer it, and the browser would then open on
    # a process about to vanish. Requiring OUR pid to be the one holding
    # the port makes the check mean what it is supposed to mean.
    if curl -fsS --max-time 2 "$HEALTH" > /dev/null 2>&1 \
       && listeners | grep -qx "$SERVER_PID"; then
        READY="yes"
        break
    fi
    sleep 1
    WAITED=$((WAITED + 1))
done

if [ -z "$READY" ]; then
    TAIL="$(tail -20 "$SERVER_LOG" 2>/dev/null)"
    kill "$SERVER_PID" 2>/dev/null
    die "The server did not answer within ${START_TIMEOUT} seconds.

Last lines of its log:

$TAIL"
fi

say "API is up after ${WAITED}s"

# ---------------------------------------------------------------------
# 5. Warn if nobody can log in
# ---------------------------------------------------------------------
# Without an active Super Admin the login screen is a locked door with no
# key, and it reports that as "invalid credentials" - which sends you
# looking for a typo instead of a missing account. Better to say it.

ADMINS="$("$VENV_PY" - <<'PY' 2>/dev/null
import sqlite3, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
try:
    from core.config import load_config
    db = Path(load_config().path("database_file"))
    if not db.is_file():
        print(0); raise SystemExit
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    row = con.execute(
        "SELECT COUNT(*) FROM users WHERE role='SUPER_ADMIN' AND is_active=1"
    ).fetchone()
    print(row[0] if row else 0)
except Exception:
    print(-1)          # could not tell; stay quiet rather than cry wolf
PY
)"

if [ "${ADMINS:-0}" = "0" ]; then
    say "no active Super Admin account exists"
    osascript \
        -e 'on run {t, m}' \
        -e 'display alert t message m' \
        -e 'end run' \
        -- "No administrator account yet" \
"IBVAP is running, but there is no Super Admin account to sign in with.

Create one from a Terminal:

  cd \"$PROJECT\"
  source .venv/bin/activate
  python scripts/create_admin.py" >/dev/null 2>&1
fi

# ---------------------------------------------------------------------
# 6. Open the dashboard
# ---------------------------------------------------------------------

say "opening $URL"
notify "IBVAP is ready" "Opening the dashboard."
open "$URL"

say "IBVAP is running. Quit this app from the Dock to stop it."

# Hold the app open. Without this the .app would exit immediately, macOS
# would consider it finished, and the Dock would offer no way to stop the
# server it left behind.
wait "$SERVER_PID"
say "the server exited on its own"
