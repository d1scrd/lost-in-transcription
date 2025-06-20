#*Imports

import os
import copy
import xml.etree.ElementTree as ET

#* CONFIG
INPUT_FOLDER = "../transcriptions/TransXMLs"
OUTPUT_FILE = "../transcriptions/TransXMLs/combined_kdd_transcriptions.xml"

def indent(elem, level=0):
    i = "\n" + level * " "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + " "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i

def main():
    #* create the combined root
    combined_root = ET.Element("transcripts")

    #* counter for new ids
    next_id = 1

    #* collect & sort all XML filenames
    files = sorted(
        f for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith(".xml")
    )

    #* loop once over every file
    for fname in files:
        path = os.path.join(INPUT_FOLDER, fname)
        tree = ET.parse(path)
        root = tree.getroot()

        # derive a tag from the filename (no extension), lower‐cased
        fname_no_ext = os.path.splitext(fname)[0].lower()
        # create a wrapper element for this file
        file_wrapper = ET.Element(fname_no_ext)

        # now move each <topicblock> under that wrapper
        for block in root.findall("topicblock"):
            # deep-copy preserves text, children & attributes
            new_block = copy.deepcopy(block)

            # overwrite only the id attribute, keep incrementing
            new_block.set("id", str(next_id))
            next_id += 1

            file_wrapper.append(new_block)

        # append the entire file-wrapper under <news>
        combined_root.append(file_wrapper)

    #* pretty-print & write out
    indent(combined_root)
    combined_tree = ET.ElementTree(combined_root)
    combined_tree.write(
        OUTPUT_FILE,
        encoding="utf-8",
        xml_declaration=True
    )

    print(f"Merged {next_id-1} topicblocks into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
