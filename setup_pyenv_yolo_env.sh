#!/usr/bin/env bash
set -euo pipefail

# Recreate yolo_env using pyenv + venv (without conda).
# Target Python version is taken from environment.yml: 3.10.20

PYTHON_VERSION="3.10.20"
VENV_DIR="yolo_env"

if ! command -v pyenv >/dev/null 2>&1; then
  echo "pyenv is not installed. Install pyenv first." >&2
  exit 1
fi

# Avoid inheriting host-specific OpenSSL flags (common cause of runtime ABI mismatch on compute nodes).
unset CPPFLAGS || true
unset LDFLAGS || true
unset PKG_CONFIG_PATH || true
unset PYTHON_CONFIGURE_OPTS || true

# Prefer system OpenSSL so runtime and build-time ABI stay compatible across login/compute nodes.
OPENSSL_PREFIX=""
for d in /usr /usr/local; do
  if [ -f "$d/include/openssl/ssl.h" ]; then
    OPENSSL_PREFIX="$d"
    break
  fi
done

if [ -n "$OPENSSL_PREFIX" ]; then
  export PYTHON_CONFIGURE_OPTS="--with-openssl=$OPENSSL_PREFIX"
  echo "Using system OpenSSL prefix: $OPENSSL_PREFIX"
else
  echo "[WARN] OpenSSL headers not found under /usr or /usr/local."
  echo "[WARN] Install OpenSSL development package, then rerun this script."
fi

# Install Python version if needed and create a project-local virtualenv.
pyenv install -s "$PYTHON_VERSION"

if ! PYENV_VERSION="$PYTHON_VERSION" pyenv exec python -c "import ssl" >/dev/null 2>&1; then
  echo "[WARN] Existing pyenv Python $PYTHON_VERSION has no ssl. Rebuilding..."
  pyenv uninstall -f "$PYTHON_VERSION"
  pyenv install "$PYTHON_VERSION"
fi

pyenv local "$PYTHON_VERSION"
python -m venv "$VENV_DIR"

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python - <<'PY'
import ssl
print("ssl module OK:", ssl.OPENSSL_VERSION)
PY

python - <<'PY'
import importlib.util
import subprocess
spec = importlib.util.find_spec("_ssl")
print("_ssl module:", spec.origin if spec else "not found")
if spec and spec.origin:
  out = subprocess.run(["ldd", spec.origin], text=True, capture_output=True, check=False)
  print("ldd(_ssl) first lines:")
  print("\n".join(out.stdout.splitlines()[:8]))
PY

python -m pip install --upgrade pip setuptools wheel

# Keep persistent venv minimal. Heavy packages (torch/ultralytics) are installed in TMPDIR at job runtime.
python -m pip install pyyaml tifffile

python -V
python -m pip freeze | sed -n '1,80p'

echo "Done. Activate with: source $VENV_DIR/bin/activate"
