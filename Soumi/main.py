import nltk
import numpy as np

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Read text file
with open("sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

print("\nORIGINAL TEXT:\n")
print(text)

# -----------------------------
# STEP 1 : Sentence Tokenization
# -----------------------------
sentences = sent_tokenize(text)

# -----------------------------
# STEP 2 : Word Tokenization
# -----------------------------
words = word_tokenize(text)

# -----------------------------
# STEP 3 : Stopword Removal
# -----------------------------
stop_words = set(stopwords.words('english'))

filtered_words = []

for word in words:
    if word.lower() not in stop_words:
        filtered_words.append(word)

# -----------------------------
# STEP 4 : POS Tagging
# -----------------------------
pos_tags = pos_tag(filtered_words)

# Keep only nouns and verbs
important_words = []

for word, tag in pos_tags:
    
    if tag.startswith('NN') or tag.startswith('VB'):
        important_words.append(word)

# -----------------------------
# STEP 5 : Stemming
# -----------------------------
stemmer = PorterStemmer()

stemmed_words = []

for word in important_words:
    stemmed_word = stemmer.stem(word)
    stemmed_words.append(stemmed_word)

# -----------------------------
# STEP 6 : TF-IDF Calculation
# -----------------------------
vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(sentences)

# -----------------------------
# STEP 7 : Sentence Scoring
# -----------------------------
sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

# -----------------------------
# STEP 8 : Sentence Selection
# -----------------------------
top_n = max(1, int(len(sentences) * 0.4))

top_sentence_indices = sentence_scores.argsort()[-top_n:]

# -----------------------------
# STEP 9 : Summary Generation
# -----------------------------
summary = []

for i in sorted(top_sentence_indices):
    summary.append(sentences[i])

final_summary = " ".join(summary)

# -----------------------------
# OUTPUTS
# -----------------------------
print("\n==============================")
print("PREPROCESSING OUTPUT")
print("==============================\n")

print("Filtered Words:\n")
print(filtered_words)

print("\nPOS Tagged Words:\n")
print(pos_tags)

print("\nStemmed Words:\n")
print(stemmed_words)

print("\n==============================")
print("FINAL SUMMARY")
print("==============================\n")

print(final_summary)