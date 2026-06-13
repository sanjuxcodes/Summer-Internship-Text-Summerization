
# ============================================================
# ADVANCED EXTRACTIVE TEXT SUMMARIZER
# Optimized TF-IDF + NLP Pipeline
# Combines Best Features of Code 1, 2 and 3
# ============================================================

import nltk
import numpy as np
import math

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from nltk.chunk import ne_chunk
from nltk.tree import Tree

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# DOWNLOADS (RUN ONLY FIRST TIME)
# ============================================================

nltk.download('punkt')

nltk.download('punkt_tab')



nltk.download('stopwords')



nltk.download('averaged_perceptron_tagger')

nltk.download('averaged_perceptron_tagger_eng')



nltk.download('wordnet')

nltk.download('omw-1.4')



nltk.download('maxent_ne_chunker')

nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
# ============================================================
# INITIALIZATION
# ============================================================

stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

# ============================================================
# HELPER FUNCTION : POS CONVERSION
# ============================================================

def get_wordnet_pos(tag):

    if tag.startswith('J'):
        return 'a'

    elif tag.startswith('V'):
        return 'v'

    elif tag.startswith('N'):
        return 'n'

    elif tag.startswith('R'):
        return 'r'

    return 'n'

# ============================================================
# EXTRACT NAMED ENTITIES
# ============================================================

def extract_named_entities(tagged_words):

    entities = []

    chunks = ne_chunk(tagged_words)

    for chunk in chunks:

        if isinstance(chunk, Tree):

            entity = " ".join(c[0] for c in chunk)

            entities.append(entity)

    return entities

# ============================================================
# PREPROCESS SENTENCES
# ============================================================

def preprocess_sentences(sentences):

    clean_sentences = []

    sentence_entities = []

    for sentence in sentences:

        words = word_tokenize(sentence.lower())

        tagged_words = pos_tag(words)

        filtered_words = []

        # NER Extraction
        entities = extract_named_entities(tagged_words)

        sentence_entities.append(entities)

        for word, tag in tagged_words:

            # Remove punctuation + stopwords
            if not word.isalpha():
                continue

            if word in stop_words:
                continue

            # Keep only important POS
            if (
                tag.startswith('NN')
                or tag.startswith('VB')
                or tag.startswith('JJ')
            ):

                wn_pos = get_wordnet_pos(tag)

                lemma = lemmatizer.lemmatize(word, pos=wn_pos)

                filtered_words.append(lemma)

        clean_sentences.append(
            " ".join(filtered_words)
        )

    return clean_sentences, sentence_entities

# ============================================================
# CALCULATE TF-IDF
# ============================================================

def calculate_tfidf(clean_sentences):

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(clean_sentences)

    feature_names = vectorizer.get_feature_names_out()

    tfidf_scores = np.asarray(
        tfidf_matrix.mean(axis=0)
    ).flatten()

    word_rankings = sorted(
        zip(feature_names, tfidf_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return tfidf_matrix, word_rankings

# ============================================================
# SENTENCE SCORING
# ============================================================

def score_sentences(
    tfidf_matrix,
    sentence_entities,
    sentences
):

    sentence_scores = []

    total_sentences = len(sentences)

    base_scores = np.asarray(
        tfidf_matrix.sum(axis=1)
    ).flatten()

    for idx, score in enumerate(base_scores):

        # ------------------------------------------------
        # POSITION BONUS
        # ------------------------------------------------

        position_bonus = (
            (total_sentences - idx)
            / total_sentences
        )

        score += position_bonus

        # ------------------------------------------------
        # NER BONUS
        # ------------------------------------------------

        ner_bonus = 0.2 * len(
            sentence_entities[idx]
        )

        score += ner_bonus

        # ------------------------------------------------
        # LENGTH NORMALIZATION
        # ------------------------------------------------

        sentence_length = len(
            word_tokenize(sentences[idx])
        )

        if sentence_length > 0:

            score = score / math.sqrt(
                sentence_length
            )

        sentence_scores.append(score)

    return sentence_scores

# ============================================================
# REDUNDANCY REMOVAL
# ============================================================

def remove_redundancy(
    ranked_indices,
    tfidf_matrix,
    threshold=0.75
):

    selected = []

    for idx in ranked_indices:

        is_redundant = False

        for chosen in selected:

            similarity = cosine_similarity(
                tfidf_matrix[idx],
                tfidf_matrix[chosen]
            )[0][0]

            if similarity > threshold:

                is_redundant = True

                break

        if not is_redundant:

            selected.append(idx)

    return selected

# ============================================================
# GENERATE SUMMARY
# ============================================================

def generate_summary(
    text,
    compression_ratio=0.3
):

    # ------------------------------------------------
    # SENTENCE TOKENIZATION
    # ------------------------------------------------

    sentences = sent_tokenize(text)

    if len(sentences) == 0:

        return "Empty document."

    # ------------------------------------------------
    # PREPROCESSING
    # ------------------------------------------------

    clean_sentences, sentence_entities = (
        preprocess_sentences(sentences)
    )

    # ------------------------------------------------
    # TF-IDF
    # ------------------------------------------------

    tfidf_matrix, word_rankings = (
        calculate_tfidf(clean_sentences)
    )

    # ------------------------------------------------
    # SENTENCE SCORING
    # ------------------------------------------------

    sentence_scores = score_sentences(
        tfidf_matrix,
        sentence_entities,
        sentences
    )

    # ------------------------------------------------
    # RANK SENTENCES
    # ------------------------------------------------

    ranked_indices = np.argsort(
        sentence_scores
    )[::-1]

    # ------------------------------------------------
    # REDUNDANCY REMOVAL
    # ------------------------------------------------

    ranked_indices = remove_redundancy(
        ranked_indices,
        tfidf_matrix
    )

    # ------------------------------------------------
    # SUMMARY SIZE
    # ------------------------------------------------

    summary_size = max(
        1,
        int(len(sentences) * compression_ratio)
    )

    selected_indices = ranked_indices[
        :summary_size
    ]

    selected_indices = sorted(
        selected_indices
    )

    # ------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------

    summary = " ".join(
        [sentences[i] for i in selected_indices]
    )

    return (
        summary,
        word_rankings,
        sentence_scores
    )

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    with open(
        "sample.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    compression = float(
        input(
            "Enter compression ratio (0.1 - 1.0): "
        )
    )

    summary, word_rankings, sentence_scores = (
        generate_summary(
            text,
            compression
        )
    )

    # =====================================================
    # TF-IDF WORD RANKINGS
    # =====================================================

    print("\n" + "=" * 70)
    print("TOP TF-IDF WORD RANKINGS")
    print("=" * 70)

    for word, score in word_rankings[:30]:

        print(f"{word:<20} --> {score:.6f}")

    # =====================================================
    # SENTENCE SCORES
    # =====================================================

    print("\n" + "=" * 70)
    print("SENTENCE SCORES")
    print("=" * 70)

    for idx, score in enumerate(sentence_scores):

        print(
            f"Sentence {idx+1:<3} --> "
            f"{score:.6f}"
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(summary)

    print("=" * 70)
