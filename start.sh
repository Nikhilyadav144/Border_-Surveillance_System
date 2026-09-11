#!/bin/bash
# =====================================================================
# start.sh - the Terminal way in
# =====================================================================
#
# Kept for people who prefer a Terminal, but all of the logic now lives
# in scripts/launch_ibvap.sh, which is also what IBVAP.app runs. One set
# of steps, one place to fix them.
#
# This file used to hard-code the project location:
#
#     PROJECT="$HOME/Desktop/AI-Border-Surveillance"
#
# which broke the moment the folder was moved. It now works out where it
# is from its own location instead, so it follows the project wherever
# the project goes.
#
# It also used to start a second server (`npm run dev` on port 5173).
# That is no longer needed: FastAPI serves the compiled dashboard itself,
# so there is one process on one port.
# =====================================================================

set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec /bin/bash "$HERE/scripts/launch_ibvap.sh" "$HERE"
