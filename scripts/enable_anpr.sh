#!/bin/bash
# =====================================================================
# Turn ANPR on.
# =====================================================================
#
# Number plate reading is the one optional part of IBVAP: without an OCR
# engine the platform runs completely and simply reports that it cannot
# read plates. This script installs that engine.
#
# EVERYTHING IT NEEDS IS ALREADY IN THE PROJECT. It downloads nothing.
# The wheels are in vendor/anpr-wheels and the model weights are in
# vendor/anpr-models, so this works with the wifi off.
#
# Run it from a Terminal:
#
#     cd ~/Desktop/24csu144/Projects/AI-Border-Surveillance
#     ./scripts/enable_anpr.sh
#
# Double-clicking it in Finder will NOT run it - Finder opens .sh files
# in a text editor. It has to be a Terminal.
#
# Everything it prints is also written to logs/enable_anpr.log, so if it
# fails the reason is recorded rather than lost when the window closes.
# =====================================================================

set -u

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WHEELS="$PROJECT/vendor/anpr-wheels"
MODELS="$PROJECT/vendor/anpr-models"
TARGET="$HOME/.EasyOCR/model"
LOG="$PROJECT/logs/enable_anpr.log"

mkdir -p "$PROJECT/logs" 2>/dev/null

# Everything from here on goes to the screen AND to the log. A failure
# that scrolls past or closes with the window is a failure nobody can
# diagnose afterwards.
exec > >(tee "$LOG") 2>&1

VENV_PY="$PROJECT/.venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY="$PROJECT/.venv/bin/python3"

say()  { printf '  %s\n' "$*"; }
fail() {
    printf '\n  FAILED: %s\n' "$*"
    printf '\n  The full log is at:\n    %s\n\n' "$LOG"
    exit 1
}

echo
echo "==============================================="
echo "  IBVAP - enabling number plate recognition"
echo "==============================================="
echo

# ---------------------------------------------------------------------
# 0. Record the environment first
# ---------------------------------------------------------------------
# Written before anything can go wrong, because when a wheel is rejected
# the question is always "rejected for which platform and which Python",
# and that has to be in the log already.

say "project : $PROJECT"
say "log     : $LOG"
[ -x "$VENV_PY" ] || fail "No Python at $PROJECT/.venv/bin/python"
say "python  : $("$VENV_PY" -c 'import sys; print(sys.version.split()[0])' 2>&1)"
say "platform: $("$VENV_PY" -c 'import platform; print(platform.system(), platform.machine(), platform.mac_ver()[0])' 2>&1)"
say "pip     : $("$VENV_PY" -m pip --version 2>&1 | head -1)"
echo

[ -d "$WHEELS" ] || fail "Missing $WHEELS"
[ -d "$MODELS" ] || fail "Missing $MODELS"
say "wheels found : $(ls -1 "$WHEELS"/*.whl 2>/dev/null | wc -l | tr -d ' ')"
say "model chunks : $(ls -1 "$MODELS"/*.gz 2>/dev/null | wc -l | tr -d ' ')"
echo

# ---------------------------------------------------------------------
# 1. The OCR engine
# ---------------------------------------------------------------------
if "$VENV_PY" -c "import easyocr" 2>/dev/null; then
    say "EasyOCR is already installed."
else
    say "Installing the OCR engine from the bundled wheels..."
    echo

    # --no-index: never reach for the network, even when it is available.
    # --no-deps: every other dependency is already installed. Without it
    #   pip insists on opencv-python-headless, which is the same cv2 this
    #   project already has through opencv-python, and having both is a
    #   known source of confusing import behaviour.
    if ! "$VENV_PY" -m pip install --no-index --no-deps \
            --find-links "$WHEELS" \
            easyocr scikit-image imageio tifffile lazy_loader \
            python-bidi pyclipper shapely ninja 2>&1 | sed 's/^/      /'
    then
        echo
        say "pip rejected the install. The tags this Python accepts are:"
        "$VENV_PY" -m pip debug --verbose 2>&1 \
            | sed -n '/Compatible tags/,$p' | head -14 | sed 's/^/      /'
        echo
        say "and the wheels on offer are:"
        ls -1 "$WHEELS" | sed 's/^/      /'
        fail "The bundled wheels do not match this Python. The two lists
      above are what I need to fix it."
    fi

    echo
    if ! "$VENV_PY" -c "import easyocr" 2>&1 | sed 's/^/      /'; then
        say "The install reported success but easyocr still will not import:"
        "$VENV_PY" -c "import easyocr" 2>&1 | tail -5 | sed 's/^/      /'
        fail "easyocr installed but cannot be imported."
    fi
    say "Installed."
fi
echo

# ---------------------------------------------------------------------
# 2. Model weights
# ---------------------------------------------------------------------
# EasyOCR downloads about 95 MB of weights the first time it runs. On a
# demo machine with no internet that is a failure at the worst possible
# moment, so the weights ship with the project.

mkdir -p "$TARGET" || fail "Could not create $TARGET"

reassemble() {
    name="$1"; out="$TARGET/$2"
    if [ -f "$out" ]; then
        say "$2 already in place."
        return 0
    fi
    parts=$(ls "$MODELS/${name}_"*.gz 2>/dev/null | sort)
    [ -n "$parts" ] || fail "No parts for $2 in $MODELS"
    say "Assembling $2..."
    : > "$out.tmp" || fail "Cannot write to $TARGET"
    for p in $parts; do
        gunzip -c "$p" >> "$out.tmp" || fail "Could not expand $p"
    done
    mv "$out.tmp" "$out" || fail "Could not move $out.tmp into place"
}

reassemble craft   craft_mlt_25k.pth
reassemble english english_g2.pth

# Verified, not assumed. A truncated weight file fails much later with an
# error that says nothing about the real cause.
check() {
    file="$TARGET/$1"; want="$2"
    got=$(md5 -q "$file" 2>/dev/null || md5sum "$file" 2>/dev/null | cut -d' ' -f1)
    if [ "$got" = "$want" ]; then
        say "$1 verified."
    else
        rm -f "$file"
        fail "$1 did not match its checksum and was removed.
      expected $want
      got      ${got:-nothing}
      Run this script again."
    fi
}

check craft_mlt_25k.pth 2f8227d2def4037cdb3b34389dcf9ec1
check english_g2.pth    5864788e1821be9e454ec108d61b887d
echo

# ---------------------------------------------------------------------
# 3. Prove it works before claiming it does
# ---------------------------------------------------------------------
say "Starting the engine once to check it really loads..."
cd "$PROJECT" || fail "Could not enter $PROJECT"

"$VENV_PY" - <<'PY' 2>&1 | sed 's/^/      /'
import sys, traceback
sys.path.insert(0, ".")
try:
    from core.anpr import build_backend
    backend = build_backend("easyocr")
    if backend.available:
        print(f"OCR backend ready: {backend.name}")
    else:
        print("The backend loaded but reports itself unavailable.")
        sys.exit(1)
except Exception:
    traceback.print_exc()
    sys.exit(1)
PY

if [ "${PIPESTATUS[0]}" != "0" ]; then
    fail "The engine did not start. The traceback is above and in the log."
fi

echo
echo "  ==============================================="
echo "  ANPR is enabled."
echo
echo "  Restart IBVAP with:  borderguard"
echo "  ==============================================="
echo
