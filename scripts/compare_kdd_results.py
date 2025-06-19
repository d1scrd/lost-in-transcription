#!/usr/bin/env python3

#* Imports
import os
import re
import xml.etree.ElementTree as ET
from jiwer import wer, cer
import difflib

#* ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"
RESET = "\033[0m"

#* CONFIGURATION
TRANSCRIPT_FILES = [
    "kdd_transcripts_medium_all_copy_healed.xml",
    "kdd_transcripts_medium_all_copy_healed_normalized.xml",
]
REFERENCE_XML = "combined_all.xml"
METRICS_DIR = "metrics"

#* Get completed folders from transcript XML
def get_completed_folders_from_transcripts(xml_path):
    with open(xml_path, "r", encoding="utf-8") as f:
        xml = f.read()
    # now match any number of digits, not exactly 3
    open_tbs = [m.group(1) for m in re.finditer(r'<topicblock id="(\d+)">', xml)]
    close_cnt = len(re.findall(r"</topicblock>", xml))
    return open_tbs[:close_cnt]

#* Parse transcripts by completed folders
def parse_transcripts(xml_path, completed_folders):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    blocks = {}
    # only nested topicblock/text in the new structure
    for tb in root.findall(".//topicblock"):
        folder = tb.attrib.get("id")
        if folder not in completed_folders:
            continue
        text_elem = tb.find("text")
        text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        # use a single key so downstream logic stays identical
        blocks.setdefault(folder, {})['text'] = text
    return blocks

#* Parse reference corpus XML (same nested-topicblock logic for your combined_all.xml)
def parse_reference_corpus(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    block_map = {}
    for tb in root.findall(".//topicblock"):
        block = tb.attrib.get("id")
        text_elem = tb.find("text")
        text = text_elem.text.strip() if text_elem is not None and text_elem.text else ""
        block_map.setdefault(block, {})['text'] = text
    return block_map

#* Levenshtein distance utility
def levenshtein(a, b):
    return sum(1 for _ in difflib.ndiff(a, b) if _[0] != ' ')

#* Clean string for WER calculation
def clean_for_wer(s):
    return re.sub(r"[^\w\s]", "", s.lower()).strip()

#* Main processing function for a transcript file
def process_file(transcripts_xml, reference_xml, metrics_dir):
    print(f"{CYAN}Processing {transcripts_xml}...{RESET}")

    completed = get_completed_folders_from_transcripts(transcripts_xml)
    transcripts = parse_transcripts(transcripts_xml, completed)
    ref_map = parse_reference_corpus(reference_xml)

    print(f"\n{CYAN}Health check for {transcripts_xml}:{RESET}")
    all_blocks = sorted(set(completed) | set(ref_map.keys()))
    for folder in all_blocks:
        in_trans = folder in transcripts
        in_ref = folder in ref_map
        if in_trans and in_ref:
            t_count = len(transcripts[folder])
            r_count = len(ref_map[folder])
            color = GREEN if t_count == r_count else RED
            status = f"OK: {t_count} entries each" if t_count == r_count else f"mismatch: {t_count} vs {r_count}"
            print(f"{color}Block {folder} {status}{RESET}")
        elif in_trans:
            print(f"{YELLOW}Block {folder} in transcripts only ({len(transcripts[folder])} entries){RESET}")
        else:
            print(f"{YELLOW}Block {folder} in reference only ({len(ref_map[folder])} entries){RESET}")

    os.makedirs(metrics_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(transcripts_xml))[0]
    metrics_path = os.path.join(metrics_dir, f"{base_name}_metrics.xml")

    with open(metrics_path, "w", encoding="utf-8") as out:
        out.write('<?xml version="1.0" encoding="UTF-8"?>\n<transcripts>\n')
        for folder in completed:
            if folder not in ref_map:
                continue
            out.write(f'  <block id="{folder}">\n')
            for fname, hypo in transcripts[folder].items():
                ref = ref_map[folder].get(fname, "")
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
        process_file(transcripts_xml, REFERENCE_XML, METRICS_DIR)

if __name__ == "__main__":
    main()
