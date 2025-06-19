import xml.etree.ElementTree as ET
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

#* Paths to input and output XML files
INPUT_PATH = './kdd_transcripts_medium_all_copy_healed_normalized_final.xml'
OUTPUT_PATH = './kdd_transcripts_medium_all_copy_healed_normalized_final_translated.xml'

def load_translator(model_name='facebook/nllb-200-1.3b', src_lang='afr_Latn', tgt_lang='eng_Latn'):
    #* Initialize tokenizer and model for NLLB-200
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang, tgt_lang=tgt_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    #* Build a translation pipeline
    return pipeline('translation', model=model, tokenizer=tokenizer, src_lang=src_lang, tgt_lang=tgt_lang)

def translate_xml(input_path, output_path):
    #* Load XML and translator
    tree = ET.parse(input_path)
    root = tree.getroot()
    translator = load_translator()

    #* Process each topicblock's <text>
    for block in root.findall('.//topicblock'):
        text_elem = block.find('text')
        if not text_elem is None and text_elem.text and text_elem.text.strip():
            original_text = text_elem.text.strip()
            block_id = block.get('id', '')
            print(f"Translating block id={block_id}...")

            #* Run translation
            translated = translator(original_text, max_length=1000, clean_up_tokenization_spaces=True)[0]['translation_text']

            #* Append <translation> element
            ET.SubElement(block, 'translation').text = translated

    #* Write out new XML
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"Translation complete: {output_path}")

if __name__ == '__main__':
    translate_xml(INPUT_PATH, OUTPUT_PATH)