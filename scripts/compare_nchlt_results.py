#!/usr/bin/env python3

#* Imports
import os
import re
import xml.etree.ElementTree as ET
from jiwer import wer, cer
import difflib

#* ANSI color codes for feedback
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

#* CONFIGURATION
TRANSCRIPT_FILES = [
    "transcript_international1_noise.xml",
    "transcript_international1.xml",
]
REFERENCE_XML = "transcriptions/nchlt_afr.trn.xml"
TEST_REFERENCE_XML = "transcriptions/nchlt_afr.tst.xml"
METRICS_DIR = "metrics"

#* Get completed folders from transcript XML
def get_completed_folders_from_transcripts(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()
    open_blocks = [m.group(1) for m in re.finditer(r'<block id="(\d{3})">', xml)]
    close_blocks = [m.start() for m in re.finditer(r"</block>", xml)]
    return open_blocks[:len(close_blocks)]

#* Parse transcripts by completed folders
def parse_transcripts(xml_path, completed_folders):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    blocks = {}
    for block in root.findall("block"):
        folder = block.attrib.get("id")
        if folder not in completed_folders:
            continue
        files = {}
        for file_elem in block.findall("file"):
            fname = file_elem.attrib["name"]
            text = file_elem.text.strip() if file_elem.text else ""
            files[fname] = text
        blocks[folder] = files
    return blocks

#* Parse reference corpus XML
def parse_reference_corpus(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    block_map = {}
    for speaker in root.findall(".//speaker"):
        block = speaker.get("id")
        for rec in speaker.findall(".//recording"):
            audio_path = rec.get("audio") or ""
            filename = os.path.basename(audio_path)
            orth_elem = rec.find("orth")
            if orth_elem is not None and orth_elem.text:
                block_map.setdefault(block, {})[filename] = orth_elem.text.strip()
    return block_map

#* Levenshtein distance utility
def levenshtein(a, b):
    return sum(1 for _ in difflib.ndiff(a, b) if _[0] != ' ')

#* Clean string for WER calculation
def clean_for_wer(s):
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

#* Main processing function for a transcript file
def process_file(transcripts_xml, reference_xml, test_reference_xml, metrics_dir):
    print(f"{CYAN}Processing {transcripts_xml}...{RESET}")

    completed = get_completed_folders_from_transcripts(transcripts_xml)
    transcripts = parse_transcripts(transcripts_xml, completed)
    trn_map = parse_reference_corpus(reference_xml)
    tst_map = parse_reference_corpus(test_reference_xml)

    combined_ref = {**trn_map}
    for blk, files in tst_map.items():
        combined_ref.setdefault(blk, {}).update(files)

    print(f"\n{CYAN}Health check for {transcripts_xml}:{RESET}")
    all_blocks = sorted(set(completed) | set(combined_ref.keys()))
    for folder in all_blocks:
        in_trans = folder in transcripts
        in_ref = folder in combined_ref
        if in_trans and in_ref:
            t_count, r_count = len(transcripts[folder]), len(combined_ref[folder])
            color = GREEN if t_count == r_count else RED
            status = "OK: {0} entries each".format(t_count) if t_count == r_count else "mismatch: {0} vs {1}".format(
                t_count, r_count)
            print(f"{color}Block {folder} {status}{RESET}")
        elif in_trans:
            print(f"{YELLOW}Block {folder} in transcripts only ({len(transcripts[folder])} entries){RESET}")
        else:
            print(f"{YELLOW}Block {folder} in reference only ({len(combined_ref[folder])} entries){RESET}")

    os.makedirs(metrics_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(transcripts_xml))[0]
    metrics_path = os.path.join(metrics_dir, f"{base_name}_metrics.xml")

    with open(metrics_path, "w", encoding="utf-8") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n<transcripts>\n')
        for folder in completed:
            if folder not in combined_ref:
                continue
            out.write(f'  <block id="{folder}">\n')
            for fname, hypo in transcripts[folder].items():
                ref = combined_ref[folder].get(fname, "")
                hypo_c = clean_for_wer(hypo)
                ref_c = clean_for_wer(ref)
                _wer = wer(ref_c, hypo_c)
                _cer = cer(ref_c, hypo_c)
                lev = levenshtein(ref_c, hypo_c)
                out.write(f'    <file name="{fname}">\n')
                out.write(f'      <wer>{_wer:.3f}</wer>\n')
                out.write(f'      <cer>{_cer:.3f}</cer>\n')
                out.write(f'      <levenshtein>{lev}</levenshtein>\n')
                out.write(f'      <ref>{ref}</ref>\n')
                out.write(f'      <hypo>{hypo}</hypo>\n')
                out.write('    </file>\n')
            out.write('  </block>\n')
        out.write('</transcripts>\n')

    print(f"{GREEN}Metrics for {transcripts_xml} written to {metrics_path}{RESET}\n")

#* Main execution
def main():
    os.makedirs(METRICS_DIR, exist_ok=True)
    for transcripts_xml in TRANSCRIPT_FILES:
        if not os.path.isfile(transcripts_xml):
            print(f"{RED}File not found: {transcripts_xml}{RESET}")
            continue
        process_file(transcripts_xml, REFERENCE_XML, TEST_REFERENCE_XML, METRICS_DIR)

if __name__ == "__main__":
    main()
