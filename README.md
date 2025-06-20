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
## Dowload NCHLT

https://repo.sadilar.org/items/4357e52c-f8e8-4109-a373-ed6700dcba77

Download the af_za.tar.gz from the link above. Extract the zip and copy the `nchlt_afr` folder into the `audio_nchlt` folder int he cloned repository. 


## Usage

Run each step after activating the venv:

**Most files have specified paths on files that exist. But the input files can be changed to run on different input files**

## In the scripts folder
```bash
cd scripts
```

### Running NCHLT transcriptions:
**Note this scripts will take several hours to complete**

```bash
python run_nchlt_transcriptions.py
```

#### Then heal the xml for further steps:
**Please specify the inputfiles to heal inside of the heal_xml.py file**
```bash
python heal_xml.py
```

#### Comparing NCHLT results:
```bash
python compare_nchlt_results.py
```

### Running kdd transcriptions
**Run the script on the multiple files and then combine the files using the combine script**

In the run_kdd_transcriptions.py specify the folder containing the auidio files on the topic you want to transcribe

```bash
python run_kdd_transcriptions.py
```

#### Combine the transcribed files
```bash
python combine_kdd_transcripts.py
```


### Running one transcription

Specify the audio file you want to transcribe in the run_one_transcription.py

```bash
python run_one_transcription.py
```

### Running topic modeling
Specify the files you want to run through the topic modeling strip in the `run_topic_modeling.py` for afrikaans files and `run_topic_modeling_english.py` for translated files

```bash
python run_topic_modeling.py
python run_topic_modeling_english.py
```

### Running translation
Specify xml file to translate in translation.py

```bash
python translation.py
```

### Other scripts
For all specify the input files

```bash
python wave_form.py
```

```bash
python graph.py
```

```bash
python intent.py
```