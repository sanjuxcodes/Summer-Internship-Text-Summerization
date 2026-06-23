import os
import math
import nltk
import numpy as np
import networkx as nx
from collections import Counter
from flask import Flask, request, jsonify, send_from_directory

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)

# Ensure NLTK resources are downloaded silently
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('maxent_ne_chunker_tab', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def extract_named_entities(words):
    tagged = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tagged)
    entities = []
    for chunk in tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(token for token, pos in chunk)
            entities.append(entity)
    return entities

def summarize_text(text, compression_percentage):
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    processed_sentences = []
    all_words = []
    sentence_entities = []

    # PREPROCESSING
    for sentence in sentences:
        words = word_tokenize(sentence.lower())
        entities = extract_named_entities(word_tokenize(sentence))
        sentence_entities.append(entities)
        tagged_words = nltk.pos_tag(words)
        filtered = []

        for word, tag in tagged_words:
            if word.isalpha() and word not in stop_words:
                if tag.startswith('NN') or tag.startswith('VB') or tag.startswith('JJ'):
                    lemma = lemmatizer.lemmatize(word)
                    filtered.append(lemma)
                    all_words.append(lemma)
        processed_sentences.append(filtered)

    if not all_words:
        return text

    # TF CALCULATION
    total_words = len(all_words)
    word_freq = Counter(all_words)
    tf = {word: count / total_words for word, count in word_freq.items()}

    # IDF CALCULATION
    N = len(processed_sentences)
    idf = {}
    for word in word_freq:
        doc_freq = sum(1 for sentence in processed_sentences if word in sentence)
        idf[word] = math.log((N + 1) / (doc_freq + 1)) + 1

    # TF-IDF CALCULATION
    tfidf = {word: tf[word] * idf[word] for word in word_freq}

    # SENTENCE VECTORIZATION
    processed_texts = [" ".join(words) for words in processed_sentences]
    vectorizer = TfidfVectorizer()
    try:
        sentence_vectors = vectorizer.fit_transform(processed_texts)
    except ValueError:
        return text

    # SIMILARITY MATRIX
    similarity_matrix = cosine_similarity(sentence_vectors)
    np.fill_diagonal(similarity_matrix, 0)

    # GRAPH CREATION & TEXTRANK
    graph = nx.from_numpy_array(similarity_matrix)
    try:
        textrank_scores = nx.pagerank(graph, max_iter=200)
    except:
        textrank_scores = {i: 1.0 / len(sentences) for i in range(len(sentences))}

    # HYBRID SENTENCE SCORING
    sentence_scores = []
    total_sentences = len(sentences)

    for idx, words in enumerate(processed_sentences):
        tfidf_score = sum(tfidf.get(word, 0) for word in words)
        textrank_score = textrank_scores.get(idx, 0)
        ner_score = 0.5 * len(sentence_entities[idx])

        event_score = 0
        tagged_sentence = nltk.pos_tag(word_tokenize(sentences[idx]))
        for word, tag in tagged_sentence:
            if tag.startswith("VB"):
                event_score += 1

        position_weight = 1.0
        if idx == 0:
            position_weight = 1.5
        elif idx < total_sentences * 0.15:
            position_weight = 1.3
        elif idx > total_sentences * 0.80:
            position_weight = 1.5

        position_score = position_weight

        # HYBRID FORMULA
        score = (
            0.55 * textrank_score
            + 0.25 * tfidf_score
            + 0.10 * ner_score
            + 0.10 * position_score
            + 0.03 * event_score
        )

        if len(words) > 0:
            score = score / math.sqrt(len(words))

        sentence_scores.append((idx, score))

    # SUMMARY SIZE
    keep_ratio = (100 - compression_percentage) / 100
    summary_size = max(1, round(len(sentences) * keep_ratio))

    # RANK SENTENCES
    ranked_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)

    # MMR REDUNDANCY REMOVAL
    selected = []
    selected_indices = []

    for idx, score in ranked_sentences:
        redundant = False
        current_vector = sentence_vectors[idx]

        for prev_idx in selected_indices:
            sim = cosine_similarity(current_vector, sentence_vectors[prev_idx])[0][0]
            if sim > 0.75:
                redundant = True
                break

        if not redundant:
            selected.append((idx, score))
            selected_indices.append(idx)

        if len(selected) >= summary_size:
            break

    selected = sorted(selected, key=lambda x: x[0])
    summary = " ".join(sentences[idx] for idx, score in selected)
    return summary

# Serve Frontend Routes
@app.route('/')
def index():
    return send_from_directory('.', 'frontend.html')

@app.route('/summarize', methods=['POST'])
def handle_summarize():
    data = request.get_json() or {}
    text = data.get('text', '')
    compression = float(data.get('compression', 50))
    
    if not text.strip():
        return jsonify({'summary': ''})
        
    summary = summarize_text(text, compression)
    return jsonify({'summary': summary})

if __name__ == '__main__':
    # Runs locally on http://127.0.0.1:5000
    app.run(debug=True, port=5000)