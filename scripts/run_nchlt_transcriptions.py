#* Imports
import os
import whisper
import xml.sax.saxutils as saxutils
import re

AUDIO_FOLDER = r"nchlt.speech.corpus.afr/nchlt_afr/audio"     
TRANSCRIPT_XML = "transcripts.xml"  
WHISPER_MODEL = "medium"        

#* ANSI color codes for user feedback
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_CYAN = "\033[96m"
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"

#* Ensure the XML structure is correct (Used to stop Whisper and resume later)
def fix_xml_structure(xml_file):
    if not os.path.exists(xml_file):
        return 

    with open(xml_file, "r", encoding="utf-8") as file:
        content = file.read()

    #* Fix misnamed block tags to proper XML structure
    content = re.sub(
        r"<(\d{3})>(.*?)</\1>",
        lambda m: f'<block id="{m.group(1)}">{m.group(2)}</block>',
        content,
        flags=re.DOTALL,
    )

    #* Wrap filenames correctly in file tags
    content = re.sub(
        r"<([^<>\s]+\.wav)>(.*?)</\1>",
        lambda m: f'<file name="{m.group(1)}">{m.group(2)}</file>',
        content,
        flags=re.DOTALL,
    )

    content = re.sub(r"\r\n?", "\n", content)
    with open(xml_file, "w", encoding="utf-8") as file:
        file.write(content)

#* Parse existing transcripts from the XML file
def parse_existing_transcripts(xml_file):
    if not os.path.exists(xml_file):
        return [], None, '<?xml version="1.0" encoding="UTF-8"?>\n<transcripts>\n'

    with open(xml_file, "r", encoding="utf-8") as file:
        content = file.read()

    #* Detect completed and incomplete blocks in existing XML
    open_tags = list(re.finditer(r'<block id="(\d{3})">', content))
    close_tags = list(re.finditer(r"</block>", content))

    open_ids = [m.group(1) for m in open_tags]
    close_ids = [m.group(1) for m in close_tags if m.lastindex and m.group(1)]
    finished_ids = open_ids[:len(close_tags)]

    pending_id = None
    if len(open_ids) > len(close_tags):
        pending_id = open_ids[len(close_tags)]

    if pending_id:
        last_open_pos = content.rfind(f'<block id="{pending_id}">')
        partial_content = content[:last_open_pos]
    else:
        partial_content = re.sub(r"</transcripts>\s*$", "", content)

    displayed = set()
    for index, block_id in enumerate(open_ids):
        if block_id in displayed:
            continue
        displayed.add(block_id)
        if index < len(finished_ids):
            print(f"{COLOR_GREEN}Block {block_id} complete{COLOR_RESET}")
        elif block_id == pending_id:
            print(f"{COLOR_RED}Block {block_id} incomplete!{COLOR_RESET}")
        else:
            print(f"{COLOR_YELLOW}Block {block_id} unknown status!{COLOR_RESET}")

    return finished_ids, pending_id, partial_content

def main():
    print(f"Loading Whisper model '{WHISPER_MODEL}' …")
    model = whisper.load_model(WHISPER_MODEL)

    finished_ids, pending_id, current_xml = parse_existing_transcripts(TRANSCRIPT_XML)

    #* Check all existing transcripts and fix XML structure
    if finished_ids:
        print(f"\n{COLOR_GREEN}All completed blocks: {', '.join(finished_ids)}{COLOR_RESET}")
    if pending_id:
        print(f"{COLOR_CYAN}Resuming from block {pending_id}{COLOR_RESET}")
    elif finished_ids:
        print(f"{COLOR_CYAN}All previous blocks complete. Starting next block.{COLOR_RESET}")
    else:
        print(f"{COLOR_CYAN}No previous progress. Starting from scratch.{COLOR_RESET}")

    with open(TRANSCRIPT_XML, "w", encoding="utf-8") as file:
        file.write(current_xml)

    #* Get all available audio folders
    all_audio_folders = sorted(f for f in os.listdir(AUDIO_FOLDER) if os.path.isdir(os.path.join(AUDIO_FOLDER, f)))
    skip_blocks = set(finished_ids)
    folders_to_process = []

    #* If a pending block exists, start from there; otherwise, process all unprocessed folders
    if pending_id:
        start_index = all_audio_folders.index(pending_id)
        folders_to_process = all_audio_folders[start_index:]
    else:
        folders_to_process = [f for f in all_audio_folders if f not in skip_blocks]

    #* If no folders to process, exit early
    if not folders_to_process:
        print("\nNothing to do! All folders transcribed.")
        with open(TRANSCRIPT_XML, "a", encoding="utf-8") as file:
            file.write("</transcripts>\n")
        return

    #* Write the XML header and start processing folders
    with open(TRANSCRIPT_XML, "a", encoding="utf-8") as file:
        for folder in folders_to_process:
            folder_path = os.path.join(AUDIO_FOLDER, folder)
            print(f"\n=== Processing folder: {folder} ===")
            file.write(f'  <block id="{folder}">\n')

            for audio_file in sorted(os.listdir(folder_path)):
                if not audio_file.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
                    continue

                audio_path = os.path.join(folder_path, audio_file)
                print(f"Transcribing →  {audio_path}")

                #! Perform the transcription using Whisper model
                result = model.transcribe(audio_path, language="af")
                transcript = result["text"].strip()
                escaped_transcript = saxutils.escape(transcript)
                file.write(f'    <file name="{audio_file}">{escaped_transcript}</file>\n')
                file.flush()
            file.write(f'  </block>\n')
            file.flush()
        file.write("</transcripts>\n")

    print(f"\nAll done. Transcripts saved incrementally to: {TRANSCRIPT_XML}")

if __name__ == "__main__":
    main()