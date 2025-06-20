#* Imports
import os, re, shutil
import xml.etree.ElementTree as ET
import hunspell
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from difflib import SequenceMatcher

#* ANSI colors
green = "\033[92m"; yellow = "\033[93m"; red = "\033[91m"; cyan = "\033[96m"; reset = "\033[0m"

#* CONFIGURATION
INPUT_XML = "../transcriptions/kdd/kdd_transcripts_medium_all_copy_healed.xml"
AF_DIC, AF_AFF = "../dictionaries/af_ZA.dic", "../dictionaries/af_ZA.aff"
NL_DIC, NL_AFF = "../dictionaries/nl.dic", "../dictionaries/nl.aff"
TOP_K = 5
MAX_EDITS = 3
LENGTH_RATIO = (0.8, 1.2)

 #* Metrics Function  
def print_metrics(total_words, misspelled_words, dutch_words, replaced):
    total = len(total_words)
    miss = len(misspelled_words)
    dutch_count = len(dutch_words)
    replaced_count = len(replaced)
    miss_pct = (miss / total * 100) if total else 0
    dutch_pct = (dutch_count / total * 100) if total else 0
    dutch_of_miss = (dutch_count / miss * 100) if miss else 0
    replaced_pct = (replaced_count / dutch_count * 100) if dutch_count else 0

    print("\n" + "-"*35)
    print(f"{cyan}Corpus Metrics:{reset}")
    print(f"  Total unique words:       {total}")
    print(f"  Misspelled words:         {miss} ({green}{miss_pct:.1f}%{reset} of all)")
    print(f"  Dutch words:              {dutch_count} ({yellow}{dutch_pct:.1f}%{reset} of all)")
    print(f"  Dutch as % of misspelled: {yellow}{dutch_of_miss:.1f}%{reset}")
    print(f"  Replaced translations:    {replaced_count} ({green}{replaced_pct:.1f}%{reset} of Dutch)")
    print("-"*35 + "\n")


 #* Utility functions  
def levenshtein(a, b):
    n, m = len(a), len(b)
    if n > m: a, b, n, m = b, a, m, n
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[j - 1] == b[i - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[n]

def extract_predictions(output):
    if isinstance(output, list) and output:
        if isinstance(output[0], list):
            return output[0]
        if isinstance(output[0], dict):
            return output
    return []

def match_case(orig, repl):
    if orig.isupper(): return repl.upper()
    if orig.istitle(): return repl.capitalize()
    # lowercase for everything else
    return repl.lower()

 #* Stage A: Dutch detection, translation & replacement  
def stage_dutch_detect_and_replace(xml_in, xml_out, af_hobj, nl_hobj, translator):
    tree = ET.parse(xml_in)
    translations = {}
    seen = set(); misspelled = set()
    dutch_set = set(); replaced_set = set()

    for tb in tree.getroot().findall('.//topicblock'):
        text = tb.find('text').text or ''
        for w in re.findall(r"\b\w+\b", text):
            lw = w.lower()
            seen.add(lw)
            if not af_hobj.spell(lw): misspelled.add(lw)
            if lw not in translations and not af_hobj.spell(lw) and nl_hobj.spell(lw):
                out = translator(w)[0]['translation_text']
                print(f"{yellow}Dutch '{w}'{reset} → {cyan}'{out}'{reset}", end=' ')
                if out.lower() == w.lower() or len(out.split()) != 1:
                    translations[lw] = None
                    print(f"{red}SKIPPED (no-op or multi-word){reset}")
                elif af_hobj.spell(out.lower()):
                    translations[lw] = out
                    print(f"{green}KEPT{reset}")
                    dutch_set.add(lw)
                else:
                    translations[lw] = None
                    print(f"{red}SKIPPED (misspelled){reset}")
                    dutch_set.add(lw)

    for tb in tree.getroot().findall('.//topicblock'):
        txt = tb.find('text').text or ''
        new_text = txt
        for lw, afr in translations.items():
            if afr:
                pat = rf"(?<!\w){re.escape(lw)}(?!\w)"
                repl = match_case(lw, afr)
                if re.search(pat, new_text, flags=re.IGNORECASE):
                    replaced_set.add(lw)
                new_text = re.sub(pat, repl, new_text, flags=re.IGNORECASE)
        tb.find('text').text = new_text

    tree.write(xml_out, encoding='utf-8', xml_declaration=True)
    print(f"{green}Stage A done: {len(replaced_set)} replacements. Saved to {xml_out}{reset}")

    # Print metrics for Stage A
    print_metrics(seen, misspelled, dutch_set, replaced_set)

    return xml_out

 #* Phase B: Afrikaans Spell‑clean Pipeline (Stages 1–5)  
def stage_spellcheck(tree, af_hobj):
    miss = set()
    for tb in tree.getroot().findall('.//topicblock'):
        for w in re.findall(r"\b\w+\b", tb.find('text').text or ''):
            if not af_hobj.spell(w.lower()):
                miss.add(w)
    print(f"{cyan}Stage 1: {len(miss)} misspelled tokens{reset}")
    if miss:
        for w in sorted(miss):
            print(f"  {red}{w}{reset}")
    return miss

def stage_prune_suggestions(miss, af_hobj, translations):
    cand_map = {}
    for w in sorted(miss):
        cands = af_hobj.suggest(w.lower())
        if translations.get(w):
            cands.insert(0, translations[w])
        good = []
        for c in cands:
            ed = levenshtein(w.lower(), c.lower())
            lr = len(c) / len(w)
            if ed <= MAX_EDITS and LENGTH_RATIO[0] <= lr <= LENGTH_RATIO[1]:
                good.append(c)
        if good:
            cand_map[w] = good
            print(f"{cyan}Stage 2: '{w}': candidates={good}{reset}")
    return cand_map

def stage_context_ranking(tree, cand_map, fill):
    best_map = {}
    print(f"{cyan}Stage 3: contextual ranking with RoBERTa{reset}")
    for w, cands in cand_map.items():
        ctxt = None
        for tb in tree.getroot().findall('.//topicblock'):
            txt = tb.find('text').text or ''
            if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", txt):
                ctxt = re.sub(rf"(?<!\w){re.escape(w)}(?!\w)", fill.tokenizer.mask_token, txt)
                break
        if not ctxt: continue
        preds = extract_predictions(fill(ctxt))
        scores = {c: next((p['score'] for p in preds if p['token_str'].strip().lower() == c.lower()), 0.0)
                  for c in cands}
        best = max(scores, key=scores.get)
        best_map[w] = best
        print(f"  {yellow}'{w}'{reset}: best → {green}'{best}'{reset} from {cands}")
    return best_map

def stage_replace(tree, best_map, translations):
    total = 0
    print(f"{cyan}Stage 4: applying replacements{reset}")
    for tb in tree.getroot().findall('.//topicblock'):
        bid = tb.get('id')
        txt = tb.find('text').text or ''
        nt = txt
        for w, best in best_map.items():
            pat = rf"(?<!\w){re.escape(w)}(?!\w)"
            cnt = len(re.findall(pat, nt))
            if cnt:
                nt = re.sub(pat, match_case(w, best), nt)
                total += cnt
                print(f"  [block {bid}] {yellow}'{w}'{reset} → {green}'{best}'{reset} ({cnt}x)")
        for w, afr in translations.items():
            pat = rf"(?<!\w){re.escape(w)}(?!\w)"
            cnt = len(re.findall(pat, nt))
            if cnt:
                nt = re.sub(pat, match_case(w, afr), nt)
                total += cnt
                print(f"  [block {bid}] {yellow}'{w}'{reset} → {green}'{afr}'{reset} ({cnt}x)")
        tb.find('text').text = nt
    print(f"{green}Stage 4 done: {total} replacements applied{reset}")
    return total

def stage_final_check(tree, af_hobj):
    rem = set()
    for tb in tree.getroot().findall('.//topicblock'):
        for w in re.findall(r"\b\w+\b", tb.find('text').text or ''):
            if not af_hobj.spell(w.lower()):
                rem.add(w)
    print(f"{cyan}Stage 5: Remaining errors{reset}")
    if rem:
        for w in sorted(rem):
            print(f"  {red}- {w}{reset}")
    else:
        print(f"{green}None! All tokens recognized{reset}")

 #* MAIN  
if __name__ == '__main__':
    base, ext = os.path.splitext(INPUT_XML)
    norm = f"{base}_normalized_final{ext}"
    shutil.copy2(INPUT_XML, norm)
    print(f"Copied {INPUT_XML} → {norm}")

    af_hobj = hunspell.HunSpell(AF_DIC, AF_AFF)
    try:
        nl_hobj = hunspell.HunSpell(NL_DIC, NL_AFF)
    except:
        nl_hobj = None
        print(f"{yellow}Warning: Dutch dictionary not found; skipping Stage A{reset}")

    tok = AutoTokenizer.from_pretrained('xlm-roberta-base')
    mdl = AutoModelForMaskedLM.from_pretrained('xlm-roberta-base')
    fill = pipeline('fill-mask', model=mdl, tokenizer=tok, top_k=TOP_K, device=0)

     #* Stage A: Dutch → Afrikaans + metrics  
    if nl_hobj:
        translator = pipeline('translation', model='Helsinki-NLP/opus-mt-nl-af')
        norm = stage_dutch_detect_and_replace(norm, norm, af_hobj, nl_hobj, translator)

     #* Reload XML from disk  
    tree = ET.parse(norm)

     #* Stage 1: Spellcheck and print all misspelled words  
    miss = stage_spellcheck(tree, af_hobj)

     #* Continue pipeline: suggestions, context ranking, replacements  
    cand_map = stage_prune_suggestions(miss, af_hobj, {})
    best_map = stage_context_ranking(tree, cand_map, fill)
    stage_replace(tree, best_map, {})
    tree.write(norm, encoding='utf-8', xml_declaration=True)

     #* Print final spellcheck report as in Stage 5  
    stage_final_check(tree, af_hobj)
