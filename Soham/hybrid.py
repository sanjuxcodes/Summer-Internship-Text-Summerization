import nltk
import math
import numpy as np
import networkx as nx

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
# NER EXTRACTION
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
# SUMMARIZATION FUNCTION
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
    # TF-IDF WORD WEIGHTS
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
                (N + 1) /
                (doc_freq + 1)
            )
            + 1
        )

    tfidf = {}

    for word in word_freq:

        tfidf[word] = (
            tf[word]
            * idf[word]
        )

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
            f"{word:<25} "
            f"{value:.6f}"
        )

    # ------------------------------------------------------
    # TEXT RANK
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

    similarity_matrix = (
        cosine_similarity(
            sentence_vectors
        )
    )

    np.fill_diagonal(
        similarity_matrix,
        0
    )

    graph = nx.from_numpy_array(
        similarity_matrix
    )

    textrank_scores = nx.pagerank(
        graph
    )

    # ------------------------------------------------------
    # SENTENCE SCORING
    # ------------------------------------------------------

    print("\n" + "=" * 70)
    print("SENTENCE SCORES")
    print("=" * 70)

    sentence_scores = []

    total_sentences = len(
        sentences
    )

    for idx, words in enumerate(
        processed_sentences
    ):

        # TF-IDF Sentence Score

        tfidf_score = sum(
            tfidf.get(word, 0)
            for word in words
        )

        # TextRank Score

        textrank_score = (
            textrank_scores[idx]
        )

        # NER Bonus

        ner_score = (
            0.5 *
            len(
                sentence_entities[idx]
            )
        )

        # Position Bonus

        position_score = (
            (
                total_sentences
                - idx
            )
            /
            total_sentences
        )

        # Hybrid Score

        score = (

            0.5 *
            textrank_score

            +

            0.3 *
            tfidf_score

            +

            0.1 *
            ner_score

            +

            0.1 *
            position_score
        )

        # Length Normalization

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
            f"Sentence {idx+1:<3} : "
            f"{score:.6f}"
        )

    # ------------------------------------------------------
    # SUMMARY SIZE
    # ------------------------------------------------------

    compression_rate = (
        compression_percentage
        / 100
    )

    summary_size = max(
        1,
        round(
            len(sentences)
            * compression_rate
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

    selected = ranked_sentences[
        :summary_size
    ]

    selected = sorted(
        selected,
        key=lambda x: x[0]
    )

    # ------------------------------------------------------
    # GENERATE SUMMARY
    # ------------------------------------------------------

    summary = " ".join(

        sentences[idx]

        for idx, score

        in selected
    )

    return summary


# ==========================================================
# MAIN PROGRAM
# ==========================================================

with open(
    "doc1.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

compression_percentage = float(

    input(
        "Enter compression percentage (10-100): "
    )
)

summary = summarize_text(
    text,
    compression_percentage
)

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(summary)
print("=" * 70)