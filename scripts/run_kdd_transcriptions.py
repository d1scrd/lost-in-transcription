#* Imports
import os
import re
import whisper
import xml.sax.saxutils as saxutils

#* CONFIGURATION 
AUDIO_DIR      = "../Audio/Skool"    # Specify the folder of the recordings of the topic
FILE_BASENAME  = "Speaker2_Skool_" # The pre-text before the identification of the Recording ID
OUTPUT_XML     = "TransXMLs/Skool_Trans.xml"
MODEL_NAME     = "turbo"  

def main():
    print(f"Loading Whisper model '{MODEL_NAME}' …")
    model = whisper.load_model(MODEL_NAME)

    pattern = re.compile(
        rf"^{re.escape(FILE_BASENAME)}(\d+)\.(wav|mp3|m4a|flac)$",
        re.IGNORECASE
    )

    matches = []
    for fn in os.listdir(AUDIO_DIR):
        m = pattern.match(fn)
        if not m:
            continue
        idx = int(m.group(1))
        matches.append((idx, fn))

    if not matches:
        print(f"No files starting with '{FILE_BASENAME}' found in '{AUDIO_DIR}'.")
        return

    matches.sort(key=lambda x: x[0])

    with open(OUTPUT_XML, "w", encoding="utf-8") as xml_f:
        xml_f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        xml_f.write("<news>\n")

        for idx, filename in matches:
            path = os.path.join(AUDIO_DIR, filename)
            print(f"Transcribing → {path}")
            result = model.transcribe(path, language="af")
            transcript = saxutils.escape(result["text"].strip())

            xml_f.write(f'    <topicblock id="{idx}">\n')
            xml_f.write(f'        <text>{transcript}</text>\n')
            xml_f.write(f'    </topicblock>\n')

        xml_f.write("</news>\n")

    print(f"\nAll done. XML written to: {OUTPUT_XML}")

if __name__ == "__main__":
    main()
