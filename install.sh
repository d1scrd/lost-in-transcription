#!/usr/bin/env bash
set -e

# Verify Python ≥ 3.12
if ! command -v python3 &> /dev/null; then
  echo "Error: python3 not found. Please install Python 3.12 or newer."
  exit 1
fi

PY_VERSION=$(python3 -c 'import sys; print("{}.{}.{}".format(*sys.version_info[:3]))')
REQUIRED="3.12.0"
if [ "$(printf '%s\n%s' "$REQUIRED" "$PY_VERSION" | sort -V | head -n1)" != "$REQUIRED" ]; then
  echo "Error: Python $REQUIRED or newer required. Detected $PY_VERSION."
  exit 1
fi

echo "Python $PY_VERSION detected."

echo "Updating package lists..."
sudo apt update

echo "Installing FFmpeg and venv support..."
sudo apt install -y ffmpeg python3-venv

echo "Creating virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing Python dependencies..."
pip install -r ./requirements.txt

echo "To install the PyTorch CUDA edition, first check your CUDA version:"
echo "  nvidia-smi"
echo "Then visit https://pytorch.org/get-started/locally/"
echo "and copy the appropriate 'pip install' command for your setup."

echo "Setup complete!"
