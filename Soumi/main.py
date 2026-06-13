
import nltk
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from nltk.chunk import ne_chunk
from nltk.tree import Tree

from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# DOWNLOAD REQUIRED NLTK DATA
# =====================================================

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

# =====================================================
# READ TEXT FILE
# =====================================================

with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# =====================================================
# ORIGINAL TEXT
# =====================================================

print("\n===================================")
print("ORIGINAL TEXT")
print("===================================\n")

print(text)

# =====================================================
# STEP 1 : SENTENCE TOKENIZATION
# =====================================================

sentences = sent_tokenize(text)

print("\n===================================")
print("SENTENCE TOKENIZATION")
print("===================================\n")

for sentence in sentences:
    print(sentence)

# =====================================================
# STEP 2 : WORD TOKENIZATION
# =====================================================

words = word_tokenize(text)

print("\n===================================")
print("WORD TOKENIZATION")
print("===================================\n")

print(words)

# =====================================================
# STEP 3 : STOPWORD REMOVAL
# =====================================================

stop_words = set(stopwords.words('english'))

filtered_words = []

for word in words:

    if word.lower() not in stop_words and word.isalpha():

        filtered_words.append(word)

print("\n===================================")
print("STOPWORD REMOVAL")
print("===================================\n")

print(filtered_words)

# =====================================================
# STEP 4 : POS TAGGING
# =====================================================

pos_tags = pos_tag(filtered_words)

print("\n===================================")
print("POS TAGGING")
print("===================================\n")

for word, tag in pos_tags:

    print(word, "-->", tag)

# =====================================================
# IMPORTANT WORD SELECTION
# KEEP ONLY NOUNS AND VERBS
# =====================================================

important_words = []

for word, tag in pos_tags:

    if tag.startswith('NN') or tag.startswith('VB'):

        important_words.append(word)

print("\n===================================")
print("IMPORTANT WORDS")
print("===================================\n")

print(important_words)

# =====================================================
# STEP 5 : STEMMING
# =====================================================

stemmer = PorterStemmer()

stemmed_words = []

for word in important_words:

    stemmed_word = stemmer.stem(word)

    stemmed_words.append(stemmed_word)

print("\n===================================")
print("STEMMING")
print("===================================\n")

print(stemmed_words)

# =====================================================
# STEP 6 : LEMMATIZATION
# =====================================================

lemmatizer = WordNetLemmatizer()

lemmatized_words = []

for word in important_words:

    lemma_word = lemmatizer.lemmatize(word)

    lemmatized_words.append(lemma_word)

print("\n===================================")
print("LEMMATIZATION")
print("===================================\n")

print(lemmatized_words)

# =====================================================
# STEP 7 : NAMED ENTITY RECOGNITION (NER)
# =====================================================

print("\n===================================")
print("NAMED ENTITY RECOGNITION")
print("===================================\n")

named_entities = []

ner_chunks = ne_chunk(pos_tags)

for chunk in ner_chunks:

    if isinstance(chunk, Tree):

        entity_name = " ".join(c[0] for c in chunk)

        entity_type = chunk.label()

        named_entities.append((entity_name, entity_type))

for entity, label in named_entities:

    print(entity, "-->", label)

# =====================================================
# STEP 8 : TF-IDF CALCULATION
# =====================================================

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(sentences)

# GET ALL WORDS

feature_names = vectorizer.get_feature_names_out()

# CALCULATE TF-IDF SCORES

tfidf_scores = tfidf_matrix.mean(axis=0).A1

# STORE WORD + SCORE

word_scores = {}

for word, score in zip(feature_names, tfidf_scores):

    word_scores[word] = score

# SORT SCORES

sorted_word_scores = sorted(
    word_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\n===================================")
print("TF-IDF SCORES")
print("===================================\n")

for word, score in sorted_word_scores:

    print(f"{word} --> {score:.6f}")

# =====================================================
# STEP 9 : SENTENCE SCORING
# =====================================================

sentence_scores = np.array(
    tfidf_matrix.sum(axis=1)
).flatten()

print("\n===================================")
print("SENTENCE SCORES")
print("===================================\n")

for i in range(len(sentences)):

    print(f"Sentence {i+1} --> {sentence_scores[i]:.6f}")

# =====================================================
# STEP 10 : SENTENCE SELECTION
# =====================================================

top_n = max(1, int(len(sentences) * 0.3))

top_sentence_indices = sentence_scores.argsort()[-top_n:]

# =====================================================
# STEP 11 : SUMMARY GENERATION
# =====================================================

summary = []

for i in sorted(top_sentence_indices):

    summary.append(sentences[i])

final_summary = " ".join(summary)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n===================================")
print("FINAL SUMMARY")
print("===================================\n")

print(final_summary)

