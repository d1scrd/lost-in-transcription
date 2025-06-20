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

SOURCE_XML = "../transcriptions/kdd/kdd_transcripts_medium_all_copy_healed.xml"      
OUTPUT_XML = "../transcriptions/kdd/kdd_transcripts_medium_all_copy_healed.xml"
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

#* Utility function to determine optimal cluster count based on document count
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
    for section in tqdm(list(root), desc="Processing sections"):
        section_name = section.tag
        print(f"\n=== Section: {section_name} ===")

        raw_blocks = section.findall('topicblock')
        cleaned_texts, block_ids = [], []

        for block in raw_blocks:
            block_id = block.get('id')
            raw_text = block.findtext('text') or ''
            cleaned = preprocess_text(raw_text)

            if cleaned:
                cleaned_texts.append(cleaned)
                block_ids.append(block_id)
                print(f"Block {block_id}: cleaned successfully")
            else:
                print(f"Block {block_id}: skipped (empty after cleaning)")

        if not cleaned_texts:
            print(f"No usable content in section {section_name}, skipping…")
            continue

        print(f"Tokenizing {len(cleaned_texts)} blocks…")
        tokens_list = [t.split() for t in cleaned_texts]
        dictionary = Dictionary(tokens_list)
        dictionary.filter_extremes(no_below=1, no_above=0.8)
        bow_corpus = [dictionary.doc2bow(t) for t in tokens_list]

        num_clusters = determine_cluster_count(len(cleaned_texts))
        print(f"Using {num_clusters} clusters for this section")

        #* Initialize models
        model_instances = {
            'BER': BertTopicWrapper(num_clusters),
            'LDA': LDATopicWrapper(num_clusters),
            'NMF': NMFTopicWrapper(num_clusters)
        }
        model_results = {}

        for model_name, model in tqdm(model_instances.items(), desc="Training models", leave=False):
            print(f"Training model: {model_name}")
            topics, topic_data = model.fit(cleaned_texts)

            word_lists = [[w for w, _ in topic_data[tid][:TOP_TERMS_PER_TOPIC]] if tid in topic_data else [] for tid in range(num_clusters)]

            um = evaluate_coherence(word_lists, tokens_list, bow_corpus, dictionary, 'u_mass')
            npmi = evaluate_coherence(word_lists, tokens_list, bow_corpus, dictionary, 'c_npmi')
            cv = evaluate_coherence(word_lists, tokens_list, bow_corpus, dictionary, 'c_v')

            print(f"{model_name} → UMass: {um:.4f}, NPMI: {npmi:.4f}, c_v: {cv:.4f}")

            model_results[model_name] = {
                'assignments': topics,
                'terms': topic_data,
                'scores': (um, npmi, cv)
            }

        section_elem = ET.SubElement(output_root, section_name)

        for i, block_id in enumerate(block_ids):
            block_elem = ET.SubElement(section_elem, 'topicblock', id=block_id)
            original_text_elem = ET.SubElement(block_elem, 'text')
            original_text_elem.text = section.find(f"topicblock[@id='{block_id}']").find('text').text

            assignments_elem = ET.SubElement(block_elem, 'topic_assignments')
            for model_name in ['BER', 'LDA', 'NMF']:
                model_data = model_results[model_name]
                topic_id = model_data['assignments'][i]
                label = ", ".join(word for word, _ in model_data['terms'][topic_id])

                model_elem = ET.SubElement(assignments_elem, 'model', name=model_name)
                ET.SubElement(model_elem, 'assigned_topic').text = str(topic_id)
                ET.SubElement(model_elem, 'topic_label').text = label

        coherence_elem = ET.SubElement(section_elem, 'model_coherence')
        for model_name in ['BER', 'LDA', 'NMF']:
            um, npmi, cv = model_results[model_name]['scores']
            score_elem = ET.SubElement(coherence_elem, 'model', name=model_name)
            ET.SubElement(score_elem, 'umass').text = f"{um:.4f}"
            ET.SubElement(score_elem, 'npmi').text = f"{npmi:.4f}"
            ET.SubElement(score_elem, 'coherence_value').text = f"{cv:.4f}"

            aggregated_scores[model_name]['umass'].append(um)
            aggregated_scores[model_name]['npmi'].append(npmi)
            aggregated_scores[model_name]['c_v'].append(cv)

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