#!/bin/sh
# Install the GPUlayer CLI into its own virtual environment, without sudo.
#
#   curl -fsSL https://raw.githubusercontent.com/antonellof/gpulayer/main/install.sh | sh
#
# Options (environment variables):
#   GPULAYER_REF=main              git branch, tag or commit to install
#   GPULAYER_EXTRAS=examples       optional extras, comma-separated (examples installs torch)
#   GPULAYER_INSTALL_DIR=...       where the virtual environment goes
#                                  (default ~/.local/share/gpulayer)
#   GPULAYER_BIN_DIR=...           where the `gpulayer` link goes (default ~/.local/bin)
#   GPULAYER_SOURCE=...            install from this path or pip URL instead of GitHub
#
# Uses uv when it is installed (it can also provide Python); otherwise python3 >= 3.10.
set -eu

REPO="https://github.com/antonellof/gpulayer"
REF="${GPULAYER_REF:-main}"
EXTRAS="${GPULAYER_EXTRAS:-}"
INSTALL_DIR="${GPULAYER_INSTALL_DIR:-$HOME/.local/share/gpulayer}"
BIN_DIR="${GPULAYER_BIN_DIR:-$HOME/.local/bin}"
VENV="$INSTALL_DIR/venv"

say() { printf '%s\n' "gpulayer-install: $*"; }
fail() { printf '%s\n' "gpulayer-install: error: $*" >&2; exit 1; }

case "$(uname -s)" in
  Linux|Darwin) ;;
  *) fail "unsupported OS $(uname -s); GPUlayer workers run on Linux and macOS" ;;
esac

if [ -n "${GPULAYER_SOURCE:-}" ]; then
  SOURCE="$GPULAYER_SOURCE"
else
  SOURCE="git+$REPO@$REF"
fi
if [ -n "$EXTRAS" ]; then
  SPEC="gpulayer[$EXTRAS] @ $SOURCE"
else
  SPEC="gpulayer @ $SOURCE"
fi
case "$SOURCE" in
  /*|./*|../*)  # a local checkout: plain path syntax
    if [ -n "$EXTRAS" ]; then SPEC="${SOURCE}[${EXTRAS}]"; else SPEC="$SOURCE"; fi ;;
esac

mkdir -p "$INSTALL_DIR" "$BIN_DIR"

if command -v uv >/dev/null 2>&1; then
  say "creating $VENV with uv"
  uv venv --quiet --allow-existing --python ">=3.10" "$VENV"
  say "installing $SPEC"
  VIRTUAL_ENV="$VENV" uv pip install --quiet --upgrade "$SPEC"
else
  PY=""
  for candidate in python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
      'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PY="$candidate"; break
    fi
  done
  [ -n "$PY" ] || fail "Python 3.10 or newer is required (or install uv: https://docs.astral.sh/uv/)"
  command -v git >/dev/null 2>&1 || [ -n "${GPULAYER_SOURCE:-}" ] \
    || fail "git is required to install from GitHub"
  say "creating $VENV with $PY"
  "$PY" -m venv "$VENV"
  say "installing $SPEC"
  "$VENV/bin/python" -m pip install --quiet --upgrade pip
  "$VENV/bin/python" -m pip install --quiet --upgrade "$SPEC"
fi

ln -sf "$VENV/bin/gpulayer" "$BIN_DIR/gpulayer"
say "installed $("$BIN_DIR/gpulayer" --version 2>/dev/null || echo gpulayer) -> $BIN_DIR/gpulayer"

case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) say "add $BIN_DIR to your PATH, for example:"
     say "  echo 'export PATH=\"$BIN_DIR:\$PATH\"' >> ~/.profile" ;;
esac
say "next: gpulayer resources detect   (what this machine can run)"
