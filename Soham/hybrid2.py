import nltk
import math
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================================
# NLTK DOWNLOADS
# ==========================================================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('wordnet')

# ==========================================================
# INITIALIZATION
# ==========================================================

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# ==========================================================
# NER FUNCTION
# ==========================================================

def extract_named_entities(words):

    tagged = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tagged)

    entities = []

    for chunk in tree:

        if hasattr(chunk, 'label'):

            entity = " ".join(
                token
                for token, pos
                in chunk
            )

            entities.append(entity)

    return entities


# ==========================================================
# MAIN SUMMARIZATION FUNCTION
# ==========================================================

def summarize_text(text, compression_percentage):

    sentences = sent_tokenize(text)

    processed_sentences = []
    all_words = []
    sentence_entities = []

    # ------------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------------

    for sentence in sentences:

        words = word_tokenize(sentence.lower())

        entities = extract_named_entities(
            word_tokenize(sentence)
        )

        sentence_entities.append(
            entities
        )

        tagged_words = nltk.pos_tag(
            words
        )

        filtered = []

        for word, tag in tagged_words:

            if (
                word.isalpha()
                and word not in stop_words
            ):

                if (
                    tag.startswith('NN')
                    or tag.startswith('VB')
                    or tag.startswith('JJ')
                ):

                    lemma = lemmatizer.lemmatize(
                        word
                    )

                    filtered.append(
                        lemma
                    )

                    all_words.append(
                        lemma
                    )

        processed_sentences.append(
            filtered
        )

    # ------------------------------------------------------
    # DISPLAY NER
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("NAMED ENTITIES (NER)")
    print("=" * 70)

    for i, entities in enumerate(
        sentence_entities,
        start=1
    ):

        print(
            f"Sentence {i}: {entities}"
        )

    # ------------------------------------------------------
    # TF CALCULATION
    # ------------------------------------------------------

    total_words = len(all_words)

    word_freq = Counter(
        all_words
    )

    tf = {}

    for word, count in word_freq.items():

        tf[word] = (
            count /
            total_words
        )

    # ------------------------------------------------------
    # IDF CALCULATION
    # ------------------------------------------------------

    N = len(
        processed_sentences
    )

    idf = {}

    for word in word_freq:

        doc_freq = sum(

            1

            for sentence

            in processed_sentences

            if word in sentence
        )

        idf[word] = (

            math.log(
                (N + 1)
                /
                (doc_freq + 1)
            )
            + 1
        )

    # ------------------------------------------------------
    # TF-IDF CALCULATION
    # ------------------------------------------------------

    tfidf = {}

    for word in word_freq:

        tfidf[word] = (
            tf[word]
            *
            idf[word]
        )

    # ------------------------------------------------------
    # AUTOMATIC TOP TF-IDF KEYWORDS
    # ------------------------------------------------------

    top_keywords = set(

        word

        for word, score

        in sorted(
            tfidf.items(),
            key=lambda x: x[1],
            reverse=True
        )[:20]
    )

    print("\nTOP TF-IDF KEYWORDS")
    print(top_keywords)

    # ------------------------------------------------------
    # DISPLAY TF-IDF VALUES
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("TF-IDF VALUES")
    print("=" * 70)

    for word, value in sorted(
        tfidf.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"{word:<25}"
            f"{value:.6f}"
        )

    # ------------------------------------------------------
    # SENTENCE VECTORIZATION
    # ------------------------------------------------------

    processed_texts = [

        " ".join(words)

        for words

        in processed_sentences
    ]

    vectorizer = TfidfVectorizer()

    sentence_vectors = (
        vectorizer.fit_transform(
            processed_texts
        )
    )

    # ------------------------------------------------------
    # SIMILARITY MATRIX
    # ------------------------------------------------------

    similarity_matrix = cosine_similarity(
        sentence_vectors
    )

    np.fill_diagonal(
        similarity_matrix,
        0
    )

    print("\n" + "=" * 70)
    print("SENTENCE SIMILARITY MATRIX")
    print("=" * 70)

    matrix_df = pd.DataFrame(
        similarity_matrix,
        index=[
            f"S{i+1}"
            for i
            in range(
                len(sentences)
            )
        ],
        columns=[
            f"S{i+1}"
            for i
            in range(
                len(sentences)
            )
        ]
    )

    print(
        matrix_df.round(3)
    )

    # ------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        similarity_matrix,
        cmap='Blues'
    )

    plt.colorbar()

    plt.xticks(
        range(len(sentences)),
        [
            f"S{i+1}"
            for i
            in range(
                len(sentences)
            )
        ]
    )

    plt.yticks(
        range(len(sentences)),
        [
            f"S{i+1}"
            for i
            in range(
                len(sentences)
            )
        ]
    )

    plt.title(
        "Sentence Similarity Matrix"
    )

    plt.tight_layout()

    plt.show()

    # ------------------------------------------------------
    # GRAPH CREATION
    # ------------------------------------------------------

    graph = nx.from_numpy_array(
        similarity_matrix
    )

    print("\n" + "=" * 70)
    print("TEXT RANK GRAPH")
    print("=" * 70)

    print(
        f"Nodes : {graph.number_of_nodes()}"
    )

    print(
        f"Edges : {graph.number_of_edges()}"
    )

    # ------------------------------------------------------
    # GRAPH VISUALIZATION
    # ------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    pos = nx.spring_layout(
        graph,
        seed=42
    )

    labels = {

        i: f"S{i+1}"

        for i

        in range(
            len(sentences)
        )
    }

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2500
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        labels,
        font_size=10
    )

    nx.draw_networkx_edges(
        graph,
        pos
    )

    edge_labels = {

        (u, v):
        f"{d['weight']:.2f}"

        for u, v, d

        in graph.edges(data=True)
    }

    nx.draw_networkx_edge_labels(
        graph,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )

    plt.title(
        "TextRank Sentence Graph"
    )

    plt.axis(
        "off"
    )

    plt.show()

    # ------------------------------------------------------
    # TEXTRANK
    # ------------------------------------------------------

    textrank_scores = nx.pagerank(
        graph
    )

    print("\n" + "=" * 70)
    print("TEXT RANK SCORES")
    print("=" * 70)

    for idx, score in sorted(
        textrank_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):

        print(
            f"Sentence {idx+1} : "
            f"{score:.6f}"
        )



    # ------------------------------------------------------
    # HYBRID SENTENCE SCORING
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("HYBRID SENTENCE SCORES")
    print("=" * 70)

    sentence_scores = []

    total_sentences = len(
        sentences
    )

    for idx, words in enumerate(
        processed_sentences
    ):

        # TF-IDF SCORE

        tfidf_score = sum(

            tfidf.get(
                word,
                0
            )

            for word

            in words
        )

        # TEXTRANK SCORE

        textrank_score = (
            textrank_scores[idx]
        )

        # NER BONUS

        ner_score = (

            0.5
            *
            len(
                sentence_entities[idx]
            )
        )
        # KEYWORD BONUS

        keyword_score = 0

        for word in words:

            if word in top_keywords:

                keyword_score += 1

        # POSITION WEIGHTING

        position_weight = 1.0

        # Beginning sentences
        if idx < total_sentences * 0.15:
            position_weight = 1.3

        # Ending sentences
        elif idx > total_sentences * 0.80:
            position_weight = 1.4

        position_score = position_weight

        # HYBRID FORMULA

        score = (

            0.40
            *
            textrank_score

            +

            0.25
            *
            tfidf_score

            +

            0.15
            *
            ner_score

            +

            0.10
            *
            position_score

            +

            0.10
            *
            keyword_score
        )
        # LENGTH NORMALIZATION

        if len(words) > 0:

            score = (

                score

                /

                math.sqrt(
                    len(words)
                )
            )

        sentence_scores.append(
            (
                idx,
                score
            )
        )

        print(
            f"Sentence {idx+1:<3}"
            f" : {score:.6f}"
        )

    # ------------------------------------------------------
    # SUMMARY SIZE
    # ------------------------------------------------------

    compression_rate = (
        compression_percentage
        /
        100
    )

    summary_size = max(
        1,
        round(
            len(sentences)
            *
            compression_rate
        )
    )

    # ------------------------------------------------------
    # RANK SENTENCES
    # ------------------------------------------------------

    ranked_sentences = sorted(

        sentence_scores,

        key=lambda x: x[1],

        reverse=True
    )

    # ------------------------------------------------------
    # MMR REDUNDANCY REMOVAL
    # ------------------------------------------------------

    selected = []

    selected_indices = []

    for idx, score in ranked_sentences:

        redundant = False

        current_vector = sentence_vectors[idx]

        for prev_idx in selected_indices:

            sim = cosine_similarity(
                current_vector,
                sentence_vectors[prev_idx]
            )[0][0]

            if sim > 0.75:

                redundant = True

                break

        if not redundant:

            selected.append(
                (idx, score)
            )

            selected_indices.append(
                idx
            )

        if len(selected) >= summary_size:

            break

    selected = sorted(
        selected,
        key=lambda x: x[0]
    )

    # ------------------------------------------------------
    # SUMMARY GENERATION
    # ------------------------------------------------------

    summary = " ".join(

        sentences[idx]

        for idx, score

        in selected
    )

    return summary


# ==========================================================
# MAIN
# ==========================================================

# ==========================================================
# INPUT FILE NAME
# ==========================================================

while True:

    file_name = input(
        "\nEnter text file name (e.g., doc1.txt): "
    )

    try:

        with open(
            file_name,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()

        break

    except FileNotFoundError:

        print(
            f"\nError: '{file_name}' not found."
        )

        print(
            "Please enter a valid file name."
        )


# ==========================================================
# COMPRESSION INPUT
# ==========================================================

compression_percentage = float(
    input(
        "\nEnter compression percentage (10-100): "
    )
)


# ==========================================================
# GENERATE SUMMARY
# ==========================================================

summary = summarize_text(
    text,
    compression_percentage
)


# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(summary)
print("=" * 70)