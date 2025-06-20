#* Imports
import os
import re
import xml.etree.ElementTree as ET
import nltk
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
import stopwordsiso as siso
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
import numpy as np
from tqdm import tqdm
import warnings

#* Ignore common runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

#* Download necessary NLTK resources
nltk.download("punkt", quiet=True)
nltk.download('punkt_tab')

SOURCE_XML = "../transcriptions/kdd/kdd_transcripts_medium_all_copy_healed_normalized_translated.xml"      
OUTPUT_XML = "../transcriptions/kdd/kdd_transcripts_medium_all_copy_healed_normalized_translated_topics.xml"
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
TOP_TERMS_PER_TOPIC = 3
CUSTOM_STOPWORD_FILE = "wordlist.txt"

def load_custom_stopwords(filepath):
    stopwords = set()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    except FileNotFoundError:
        print(f"Warning: {filepath} not found, skipping custom list")
    return stopwords

custom_stopwords = load_custom_stopwords(CUSTOM_STOPWORD_FILE)
combined_stopwords = custom_stopwords
for lang_code in ['en','nl']:
    combined_stopwords |= set(siso.stopwords(lang_code))
FINAL_STOPWORDS = {word.lower() for word in combined_stopwords}

def preprocess_text(raw):
    cleaned = re.sub(r"[^\w\s\.,;:?!'\"-]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = word_tokenize(cleaned.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in FINAL_STOPWORDS and len(t) > 2]
    return " ".join(tokens)

#* Safe metric utilities for aggregated evaluation
def safe_mean(values):
    valid = [v for v in values if not np.isnan(v)]
    return np.mean(valid) if valid else np.nan

def safe_std(values):
    valid = [v for v in values if not np.isnan(v)]
    return np.std(valid) if len(valid) > 1 else np.nan

def safe_min(values):
    valid = [v for v in values if not np.isnan(v)]
    return np.min(valid) if valid else np.nan

def safe_max(values):
    valid = [v for v in values if not np.isnan(v)]
    return np.max(valid) if valid else np.nan

def evaluate_coherence(topic_words, token_lists, corpus, dictionary, metric='c_v'):
    print(f"Calculating {metric} coherence for {len(topic_words)} topics")
    valid_topics = [t for t in topic_words if t]
    if not valid_topics:
        print("No valid topic words found, returning NaN")
        return float('nan')
    try:
        model = CoherenceModel(
            topics=valid_topics,
            texts=token_lists,
            corpus=corpus,
            dictionary=dictionary,
            coherence=metric
        )
        score = model.get_coherence()
        print(f"Computed {metric} coherence: {score:.4f}")
        return score
    except Exception as e:
        print(f"Error calculating {metric}: {e}")
        return float('nan')

#* Determine optimal cluster count based on document count
def determine_cluster_count(doc_count):
    print(f"Choosing optimal cluster count for {doc_count} docs")
    if doc_count < 3: return 2
    if doc_count < 10: return 3
    if doc_count < 20: return 4
    return min(8, doc_count // 4)

#* Topic modeling class using BERT embeddings and clustering
class BertTopicWrapper:
    def __init__(self, num_topics):
        print(f"Initializing BERTopicWrapper with {num_topics} topics")
        self.num_topics = num_topics
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.vectorizer = CountVectorizer(
            ngram_range=(1,2),
            min_df=1,
            max_df=0.8,
            stop_words=list(FINAL_STOPWORDS)
        )

    def fit(self, cleaned_texts):
        print("Fitting BERTopic model…")
        try:
            clusterer = KMeans(n_clusters=self.num_topics, random_state=42, n_init=10)
            topic_model = BERTopic(
                embedding_model=self.embedding_model,
                hdbscan_model=clusterer,
                vectorizer_model=self.vectorizer,
                nr_topics=self.num_topics
            )
            assignments, _ = topic_model.fit_transform(cleaned_texts)

            topic_terms = {}
            for topic_id in range(self.num_topics):
                if topic_id in topic_model.topic_representations_:
                    top_words = topic_model.topic_representations_[topic_id][:TOP_TERMS_PER_TOPIC]
                    topic_terms[topic_id] = [(word, score) for word, score in top_words]
                else:
                    topic_terms[topic_id] = []

            print("BERTopic training complete")
            return assignments, topic_terms

        except Exception as error:
            print(f"BERTopic failed: {error}, using fallback assignments")
            fallback = [i % self.num_topics for i in range(len(cleaned_texts))]
            return fallback, {i: [] for i in range(self.num_topics)}

#* Topic modeling with classical LDA
class LDATopicWrapper:
    def __init__(self, num_topics):
        print(f"Initializing LDATopicWrapper with {num_topics} topics")
        self.num_topics = num_topics

    def fit(self, cleaned_texts):
        print("Fitting LDA model…")
        try:
            tokenized_texts = [text.split() for text in cleaned_texts]
            dictionary = Dictionary(tokenized_texts)
            dictionary.filter_extremes(no_below=1, no_above=0.8)
            corpus = [dictionary.doc2bow(doc) for doc in tokenized_texts]

            if not any(corpus):
                print("Corpus empty after filtering — using fallback")
                fallback = [i % self.num_topics for i in range(len(cleaned_texts))]
                return fallback, {i: [] for i in range(self.num_topics)}

            lda_model = LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=self.num_topics,
                random_state=42,
                passes=10,
                alpha='auto',
                per_word_topics=True
            )

            topic_assignments = []
            for doc in corpus:
                doc_topics = lda_model.get_document_topics(doc)
                best_topic = max(doc_topics, key=lambda pair: pair[1])[0] if doc_topics else -1
                topic_assignments.append(best_topic)

            topic_terms = {
                i: lda_model.show_topic(i, TOP_TERMS_PER_TOPIC)
                for i in range(self.num_topics)
            }

            print("LDA training complete")
            return topic_assignments, topic_terms

        except Exception as error:
            print(f"LDA failed: {error}, using fallback assignments")
            fallback = [i % self.num_topics for i in range(len(cleaned_texts))]
            return fallback, {i: [] for i in range(self.num_topics)}

#* Topic modeling using NMF
class NMFTopicWrapper:
    def __init__(self, num_topics):
        print(f"Initializing NMFTopicWrapper with {num_topics} topics")
        self.num_topics = num_topics
        self.vectorizer = CountVectorizer(
            ngram_range=(1,2),
            min_df=1,
            max_df=0.8,
            stop_words=list(FINAL_STOPWORDS)
        )

    def fit(self, cleaned_texts):
        print("Fitting NMF model…")
        try:
            doc_term_matrix = self.vectorizer.fit_transform(cleaned_texts)
            if doc_term_matrix.shape[1] == 0 or doc_term_matrix.sum() == 0:
                print("Matrix is empty — using fallback")
                fallback = [i % self.num_topics for i in range(len(cleaned_texts))]
                return fallback, {i: [] for i in range(self.num_topics)}

            nmf = NMF(n_components=self.num_topics, random_state=42, max_iter=200)
            W = nmf.fit_transform(doc_term_matrix)
            H = nmf.components_
            vocab = self.vectorizer.get_feature_names_out()

            assignments = [int(np.argmax(row)) for row in W]
            topic_terms = {}
            for i in range(self.num_topics):
                top_indices = H[i].argsort()[::-1][:TOP_TERMS_PER_TOPIC]
                topic_terms[i] = [(vocab[j], float(H[i,j])) for j in top_indices]

            print("NMF training complete")
            return assignments, topic_terms

        except Exception as error:
            print(f"NMF failed: {error}, using fallback assignments")
            fallback = [i % self.num_topics for i in range(len(cleaned_texts))]
            return fallback, {i: [] for i in range(self.num_topics)}

#* Utility function to indent XML elements for pretty printing
def indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip(): 
            elem.text = i + "  "
        for child in elem: 
            indent(child, level+1)
        if not child.tail or not child.tail.strip(): 
            child.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()): 
            elem.tail = i

#* Main execution logic
if __name__ == '__main__':
    #* Load and parse the XML input file
    print(f"Parsing input file: {SOURCE_XML}")
    tree = ET.parse(SOURCE_XML)
    root = tree.getroot() 

    output_root = ET.Element('transcripts')
    aggregated_scores = {model: {'umass': [], 'npmi': [], 'c_v': []} for model in ['BER', 'LDA', 'NMF']}

    print("Loading sentence transformer and processing sections…")   
     
    for file_sec in tqdm(list(root), desc="Processing files"):
        sec_name = file_sec.tag
        print(f"\n=== Processing section: {sec_name} ===")
        
        texts = []
        ids = []
        for tb in file_sec.findall('topicblock'):
            bid = tb.get('id')
            print(f"Topicblock {bid}: extracting text")
            txt = tb.findtext('translation') or ''
            print(f"Topicblock {bid}: cleaning text")
            clean = preprocess_text(txt)
            if clean:
                print(f"Topicblock {bid}: cleaned text length {len(clean)}")
                texts.append(clean)
                ids.append(bid)
            else:
                print(f"Topicblock {bid}: cleaned text is empty, skipping")
        
        if not texts:
            print(f"No valid texts in section {sec_name}, skipping")
            continue

        print(f"Section {sec_name}: tokenizing {len(texts)} texts")
        tokens = [t.split() for t in texts]
        dic = Dictionary(tokens)
        dic.filter_extremes(no_below=1, no_above=0.8)
        corpus = [dic.doc2bow(t) for t in tokens]

        n_clust = determine_cluster_count(len(texts))
        print(f"Section {sec_name}: using {n_clust} clusters")
        
        models = {
            'BER': BertTopicWrapper(n_clust),
            'LDA': LDATopicWrapper(n_clust),
            'NMF': NMFTopicWrapper(n_clust)
        }
        results = {}

        for name, mod in tqdm(models.items(), desc="Fitting models", leave=False):
            print(f"Fitting model {name}")
            topics, tdict = mod.fit(texts)
            
            print(f"Model {name}: computing model-level coherence")
            topic_words_list = []
            for tid in range(n_clust):
                if tid in tdict and tdict[tid]:
                    words = [w for w, _ in tdict[tid][:TOP_TERMS_PER_TOPIC]]
                    topic_words_list.append(words)
                else:
                    topic_words_list.append([])
            
            um = evaluate_coherence(topic_words_list, tokens, corpus, dic, 'u_mass')
            nm = evaluate_coherence(topic_words_list, tokens, corpus, dic, 'c_npmi') 
            cv = evaluate_coherence(topic_words_list, tokens, corpus, dic, 'c_v')
            
            print(f"Model {name}: UMass = {um:.4f}, NPMI = {nm:.4f}, c_v = {cv:.4f}")
            
            results[name] = {
                'topics': topics,
                'topic_dict': tdict,
                'model_coherence': (um, nm, cv)
            }

        print(f"Section {sec_name}: building output XML")
        sec_el = ET.SubElement(output_root, sec_name)
        
        for i, bid in enumerate(ids):
            print(f"Writing assignment for topicblock id {bid}")
            block_in = file_sec.find(f"topicblock[@id='{bid}']")
            tb_el = ET.SubElement(sec_el, 'topicblock', id=bid)
            txt_el = ET.SubElement(tb_el, 'text')
            txt_el.text = block_in.find('translation').text
            
            assign_el = ET.SubElement(tb_el, 'topic_assignments')
            for name in ['BER', 'LDA', 'NMF']:
                print(f"Writing {name} assignment for topicblock id {bid}")
                modm = results[name]
                tid = modm['topics'][i]
                m_el = ET.SubElement(assign_el, 'model', name=name)
                tid = results[name]['topics'][i]
                ET.SubElement(m_el, 'assigned_topic').text = str(tid)

                words_scores = results[name]['topic_dict'][tid]
                label = ", ".join(word for word,_ in words_scores)
                ET.SubElement(m_el, 'topic_label').text = label

        print(f"Section {sec_name}: writing model coherence summary")
        sum_el = ET.SubElement(sec_el, 'model_coherence')
        for name in ['BER', 'LDA', 'NMF']:
            print(f"Section {sec_name}: summarizing model {name}")
            um, nm, cv = results[name]['model_coherence']
            
            msum = ET.SubElement(sum_el, 'model', name=name)
            ET.SubElement(msum, 'umass').text = f"{um:.4f}"
            ET.SubElement(msum, 'npmi').text = f"{nm:.4f}"
            ET.SubElement(msum, 'coherence_value').text = f"{cv:.4f}"
            
            aggregated_scores[name]['umass'].append(um)
            aggregated_scores[name]['npmi'].append(nm)
            aggregated_scores[name]['c_v'].append(cv)

    #* Overall summary block
    summary_elem = ET.SubElement(output_root, 'overall_summary')
    for model_name in ['BER', 'LDA', 'NMF']:
        model_elem = ET.SubElement(summary_elem, 'model', name=model_name)

        for metric in ['umass', 'npmi', 'c_v']:
            values = aggregated_scores[model_name][metric]
            metric_elem = ET.SubElement(model_elem, metric)
            ET.SubElement(metric_elem, 'avg').text = f"{safe_mean(values):.4f}"
            ET.SubElement(metric_elem, 'stdev').text = f"{safe_std(values):.4f}"
            ET.SubElement(metric_elem, 'min').text = f"{safe_min(values):.4f}"
            ET.SubElement(metric_elem, 'max').text = f"{safe_max(values):.4f}"

    indent(output_root)
    ET.ElementTree(output_root).write(OUTPUT_XML, encoding='utf-8', xml_declaration=True)
    print(f"\nAll done. Results written to {OUTPUT_XML}")