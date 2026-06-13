import nltk
import math

from collections import Counter

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def extract_named_entities(words):

    tagged = nltk.pos_tag(words)
    tree = nltk.ne_chunk(tagged)

    entities = []

    for chunk in tree:
        if hasattr(chunk, 'label'):
            entity = " ".join(c[0] for c in chunk)
            entities.append(entity)

    return entities


def summarize_text(text, compression_percentage):

    sentences = sent_tokenize(text)

    processed_sentences = []
    all_words = []
    sentence_entities = []

    for sentence in sentences:

        words = word_tokenize(sentence.lower())

        entities = extract_named_entities(word_tokenize(sentence))
        sentence_entities.append(entities)

        tagged_words = nltk.pos_tag(words)

        filtered = []

        for word, tag in tagged_words:

            if word.isalpha() and word not in stop_words:

                if (
                    tag.startswith('NN')
                    or tag.startswith('VB')
                    or tag.startswith('JJ')
                ):

                    lemma = lemmatizer.lemmatize(word)

                    filtered.append(lemma)
                    all_words.append(lemma)

        processed_sentences.append(filtered)

   

    print("\n" + "=" * 70)
    print("NAMED ENTITIES (NER)")
    print("=" * 70)

    for i, entities in enumerate(sentence_entities, 1):
        print(f"Sentence {i}: {entities}")

    total_words = len(all_words)

    word_freq = Counter(all_words)

    tf = {}

    for word, count in word_freq.items():
        tf[word] = count / total_words

    N = len(processed_sentences)

    idf = {}

    for word in word_freq:

        doc_freq = sum(
            1
            for sentence in processed_sentences
            if word in sentence
        )

        idf[word] = math.log((N + 1) / (doc_freq + 1)) + 1

    tfidf = {}

    for word in word_freq:
        tfidf[word] = tf[word] * idf[word]



    print("\n" + "=" * 70)
    print("TF-IDF VALUES")
    print("=" * 70)

    for word, value in sorted(
        tfidf.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{word:<25} {value:.6f}")

    sentence_scores = []

    print("\n" + "=" * 70)
    print("SENTENCE SCORES")
    print("=" * 70)

    total_sentences = len(sentences)

    for idx, words in enumerate(processed_sentences):

        score = sum(
            tfidf.get(word, 0)
            for word in words
        )

        # NER Bonus
        if sentence_entities[idx]:
            score += 0.5 * len(sentence_entities[idx])

        # Position Bonus
        position_bonus = (
            (total_sentences - idx)
            / total_sentences
        )

        score += position_bonus

        if len(words) > 0:
            score = score / math.sqrt(len(words))

        sentence_scores.append((idx, score))

        print(
            f"Sentence {idx + 1:<3} : "
            f"{score:.6f}"
        )

    compression_rate = compression_percentage / 100

    summary_size = max(
        1,
        round(len(sentences) * compression_rate)
    )

    ranked_sentences = sorted(
        sentence_scores,
        key=lambda x: x[1],
        reverse=True
    )

    selected = ranked_sentences[:summary_size]

    selected = sorted(
        selected,
        key=lambda x: x[0]
    )

    summary = " ".join(
        sentences[idx]
        for idx, _
        in selected
    )

    return summary




with open(
    "document.txt",
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