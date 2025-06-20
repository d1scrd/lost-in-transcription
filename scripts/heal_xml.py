 #** Imports
import os
import re
import html
import unicodedata

 #* ANSI color codes
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
RED = "\033[91m"

 #** CONFIGURATION
INPUT_FILES = [
    "../transcriptions/kdd/kdd_transcripts_medium_all_copy.xml",
]
MODEL_NAME = "medium"

def remove_last_incomplete_block(xml):
    open_tags = list(re.finditer(r"<(?P<id>\d{3})>", xml))
    close_tags = list(re.finditer(r"</(?P<id>\d{3})>", xml))
    open_blocks = [m.group('id') for m in open_tags]
    close_blocks = [m.group('id') for m in close_tags]
    for block in reversed(open_blocks):
        if close_blocks.count(block) < open_blocks.count(block):
            print(f"{YELLOW}Removing incomplete block: {block}{RESET}")
            last_open = f"<{block}>"
            last_open_pos = xml.rfind(last_open)
            xml = xml[:last_open_pos]
            return xml, block
    return xml, None


def ensure_transcripts_root(xml):
    xml = re.sub(r'(<\?xml[^>]*\?>\s*)', '', xml)
    xml = xml.strip()
    xml = re.sub(r"</?transcripts>\s*", "", xml)
    return f"<transcripts>\n{xml}\n</transcripts>\n"


def write_with_header(path, xml):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(xml)


def clean_special_entities_and_nl(xml):
     #* Remove stray SUB (0x1A) and other control chars
    xml = re.sub(r"[\x00-\x1F\x7F]", "", xml)

    def repl(m):
        text = html.unescape(m.group(1))
         #* remove nl tokens
        text = re.sub(r'<\|?nl\|?>|\|nl\|', '', text)
         #* strip XML-special characters
        text = text.replace('<', '').replace('>', '').replace('&', '')
         #* remove quotes and form feeds
        text = text.replace("'", '').replace('"', '').replace('\f', '')

         #* filter out non-Latin characters, but keep Afrikaans letters
        filtered = []
        for ch in text:
             #* keep basic ascii letters, digits, punctuation, whitespace
            if ch.isascii() and (ch.isalnum() or ch.isspace() or ch in ",.?!;:'-—()[]" ):
                filtered.append(ch)
            else:
                 #* keep if Unicode name indicates a Latin character (including diacritics)
                try:
                    if 'LATIN' in unicodedata.name(ch):
                        filtered.append(ch)
                except ValueError:
                    pass
        return f">{''.join(filtered)}<"

    return re.sub(r'>([^><]*)<', repl, xml)

#* Main heal function
def heal_file(xml_path):
    if not os.path.exists(xml_path):
        print(f"{RED}File not found: {xml_path}{RESET}")
        return

    with open(xml_path, 'r', encoding='utf-8') as f:
        xml = f.read()

    xml, removed = remove_last_incomplete_block(xml)
    xml = re.sub(
        r"<(?P<id>d{3})>(?P<content>.*?)</(?P=id)>",
        lambda m: f'<block id="{m.group("id")}">{m.group("content")}</block>',
        xml, flags=re.DOTALL
    )
    xml = re.sub(
        r"<(?P<fname>[^<>\s]+\.wav)>(?P<text>.*?)</(?P=fname)>",
        lambda m: f'<file name="{m.group("fname")}">{m.group("text")}</file>',
        xml, flags=re.DOTALL
    )
    xml = xml.replace('\r\n', '\n')
    xml = ensure_transcripts_root(xml)
    xml = clean_special_entities_and_nl(xml)

    base, ext = os.path.splitext(xml_path)
    healed_path = f"{base}_healed{ext}"
    write_with_header(healed_path, xml)
    print(f"{CYAN}Healed XML written to: {healed_path}{RESET}")


def main():
    for xml_file in INPUT_FILES:
        print(f"{CYAN}Processing: {xml_file}{RESET}")
        heal_file(xml_file)

if __name__ == '__main__':
    main()
