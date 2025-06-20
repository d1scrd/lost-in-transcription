#!/usr/bin/env python3
import whisper
import xml.sax.saxutils as saxutils
import os

#* Configuration
AUDIO_FILE = "../audio_kdd/Besigheid/Besigheid1.m4av"
OUTPUT_XML = "../transcriptions/transcript_besigheid_1.xml"
MODEL_NAME = "medium" #! Can also use "tiny", "small", "medium", and "turbo"
LANGUAGE = "af"

def main():
    #* Check that the audio file exists
    if not os.path.isfile(AUDIO_FILE):
        print(f"Audio file not found: {AUDIO_FILE}")
        return

    #* Load the Whisper model
    print(f"Loading Whisper model '{MODEL_NAME}'...")
    model = whisper.load_model(MODEL_NAME)

    #* Transcribe the audio
    print(f"Transcribing: {AUDIO_FILE}")
    result = model.transcribe(AUDIO_FILE, language=LANGUAGE)
    transcript = result.get("text", "").strip()

    #* Escape XML special characters
    escaped = saxutils.escape(transcript)

    #* Write transcript into XML structure
    with open(OUTPUT_XML, "w", encoding="utf-8") as xml_f:
        xml_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xml_f.write('<transcript>\n')
        xml_f.write(f'  <file name="{os.path.basename(AUDIO_FILE)}">{escaped}</file>\n')
        xml_f.write('</transcript>\n')

    print(f"✅ Done! Transcript saved to: {OUTPUT_XML}")

if __name__ == "__main__":
    main()