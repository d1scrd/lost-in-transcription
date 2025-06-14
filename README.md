# Speech-to-Text Pipeline

This project runs a speech-to-text process, cleans the transcribed text, and performs topic modeling on the cleaned text.

## Prerequisites

- Ubuntu 22.04 or newer
- Python 3.12.0 or newer (already installed)
- sudo privileges
- (Optional) CUDA-capable GPU and drivers for PyTorch CUDA edition

## Installation

Open a terminal and navigate to the project root directory:

```bash
cd /path/to/project
```

### Automated Setup

The provided installer script will:

1. Verify you have Python ≥ 3.12
2. Install FFmpeg and the Python venv package
3. Create and activate a virtual environment
4. Upgrade pip and install Python dependencies
5. Provide guidance for installing the appropriate PyTorch CUDA edition

To run it:

```bash
chmod +x install.sh
./install.sh
```

### Manual Setup (optional)

#### On Ubuntu

1. Update package lists:

   ```bash
   sudo apt update
   ```

2. Install FFmpeg and venv support:

   ```bash
   sudo apt install -y ffmpeg python3-venv
   ```

3. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. Upgrade pip and install project dependencies:

   ```bash
   pip install --upgrade pip
   pip install -r ./requirements.txt
   ```

5. Install PyTorch CUDA edition:

   - Check your CUDA version with:

     ```bash
     nvidia-smi
     ```

   - Open the PyTorch local install page and select your specs:
     [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
   - Copy the suggested `pip install` command and run it.

## Verification

```bash
python --version    # Should output Python 3.12.0 or newer
ffmpeg -version     # Should output FFmpeg version
pip list            # Should include project dependencies
```

## Usage

Run each step after activating the venv:
