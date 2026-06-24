#!/usr/bin/env bash
#
# Build a standalone macOS .app for the ANN Classifier.
#
set -e

PY="/opt/homebrew/bin/python3.14"
VENV=".buildenv"
APP_NAME="ANN Classifier"

cd "$(dirname "$0")"

echo "=== ANN Classifier — macOS build ==="
echo "Base interpreter: $PY"
"$PY" --version

# 1) isolated build environment (avoids PEP-668 system-package errors)
echo "--- creating virtual environment ($VENV) ---"
"$PY" -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "--- installing build dependencies ---"
pip install --quiet --upgrade pip
pip install --quiet numpy pyinstaller

# 2) clean previous build artefacts
echo "--- cleaning old build ---"
rm -rf build dist "${APP_NAME}.spec" "ANN.spec"

# 3) build a windowed .app bundle
#    digits.py and algorithms.py are picked up automatically as imports.
echo "--- running PyInstaller ---"
pyinstaller \
  --noconfirm \
  --windowed \
  --name "$APP_NAME" \
  --osx-bundle-identifier "com.ann.classifier" \
  main.py

deactivate

echo ""
echo "=== Build complete ==="
echo "App bundle:  dist/${APP_NAME}.app"
echo "Run it:      open \"dist/${APP_NAME}.app\""
echo ""
echo "Trained models are saved to:  ~/ANN_data/"
