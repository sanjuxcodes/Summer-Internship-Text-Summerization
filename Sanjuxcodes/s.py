# ============================================================
# TF-IDF BASED EXTRACTIVE TEXT SUMMARIZATION
# ============================================================
#
# PURPOSE OF THIS PROJECT:
# ------------------------
# This project automatically generates a summary from a long
# document using the TF-IDF (Term Frequency - Inverse
# Document Frequency) algorithm.
#
# Instead of creating new sentences, the system selects the
# most important sentences from the original document.
#
# This is called EXTRACTIVE TEXT SUMMARIZATION.
#
# WORKFLOW:
# ---------
# 1. Read input document
# 2. Convert text to lowercase
# 3. Split document into sentences
# 4. Split sentences into words
# 5. Remove stopwords
# 6. Keep only nouns and verbs
# 7. Apply stemming
# 8. Calculate TF
# 9. Calculate IDF
# 10. Calculate TF-IDF
# 11. Calculate sentence scores
# 12. Select top-ranked sentences
# 13. Generate final summary
#
# ============================================================


# ============================================================
# STEP 1 : IMPORT REQUIRED LIBRARIES
# ============================================================
#
# NLTK:
# Natural Language Toolkit
#
# Used for:
# - Sentence Tokenization
# - Word Tokenization
# - POS Tagging
# - Stopword Removal
# - Stemming
#
# math:
# Used for logarithmic calculation in IDF.
#
# Counter:
# Used for counting word frequencies.
#
# ============================================================

import nltk
import math

from collections import Counter

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer



# ============================================================
# STEP 2 : DOWNLOAD NLP RESOURCES
# ============================================================
#
# These resources are downloaded only once.
#
# punkt:
# Used for sentence and word tokenization.
#
# stopwords:
# Collection of common words like:
# the, is, are, was, were etc.
#
# averaged_perceptron_tagger:
# Used for POS Tagging.
#
# ============================================================

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')



# ============================================================
# STEP 3 : MAIN SUMMARIZATION FUNCTION
# ============================================================
#
# INPUT:
# ------
# text = complete document
#
# compression_rate:
#
# Example:
# 0.3 = keep 30% of sentences
# 0.5 = keep 50% of sentences
#
# OUTPUT:
# -------
# Final summarized text
#
# ============================================================

def summarize_text(text, compression_rate=0.3):


    # ========================================================
    # STEP 4 : SENTENCE TOKENIZATION
    # ========================================================
    #
    # PURPOSE:
    # --------
    # Divide the entire document into individual sentences.
    #
    # Example:
    #
    # AI is powerful.
    # AI helps doctors.
    #
    # becomes
    #
    # [
    #   "AI is powerful.",
    #   "AI helps doctors."
    # ]
    #
    # Each sentence will later receive a score.
    #
    # ========================================================

    sentences = sent_tokenize(text)


    # ========================================================
    # STEP 5 : INITIALIZE TOOLS
    # ========================================================
    #
    # stop_words:
    # Collection of common words.
    #
    # stemmer:
    # Used to convert words to root form.
    #
    # Example:
    #
    # playing -> play
    # played  -> play
    #
    # ========================================================

    stop_words = set(stopwords.words('english'))

    stemmer = PorterStemmer()


    # ========================================================
    # DATA STRUCTURES
    # ========================================================
    #
    # processed_sentences:
    # Stores cleaned words for every sentence.
    #
    # all_words:
    # Stores all processed words.
    #
    # ========================================================

    processed_sentences = []

    all_words = []



    # ========================================================
    # STEP 6 : PREPROCESSING EACH SENTENCE
    # ========================================================
    #
    # The paper performs:
    #
    # 1. Lowercasing
    # 2. Word Tokenization
    # 3. POS Tagging
    # 4. Stopword Removal
    # 5. Stemming
    #
    # ========================================================

    for sentence in sentences:


        # ====================================================
        # LOWERCASING
        # ====================================================
        #
        # Makes:
        #
        # AI
        # Ai
        # ai
        #
        # identical.
        #
        # ====================================================

        words = word_tokenize(sentence.lower())


        # ====================================================
        # POS TAGGING
        # ====================================================
        #
        # Assign grammatical labels.
        #
        # Example:
        #
        # doctor -> NN
        # helps  -> VB
        #
        # The paper mainly focuses on
        # NOUNS and VERBS.
        #
        # ====================================================

        tagged_words = nltk.pos_tag(words)


        filtered_words = []


        for word, tag in tagged_words:


            # ================================================
            # KEEP ONLY NOUNS AND VERBS
            # ================================================
            #
            # NN = Noun
            # VB = Verb
            #
            # These carry most information.
            #
            # ================================================

            if tag.startswith('NN') or tag.startswith('VB'):


                # ============================================
                # REMOVE STOPWORDS
                # ============================================
                #
                # Example:
                #
                # the
                # is
                # are
                # was
                #
                # are removed.
                #
                # ============================================

                if word.isalpha() and word not in stop_words:


                    # ========================================
                    # STEMMING
                    # ========================================
                    #
                    # playing -> play
                    # running -> run
                    #
                    # Reduces variations of the same word.
                    #
                    # ========================================

                    stemmed_word = stemmer.stem(word)

                    filtered_words.append(stemmed_word)

                    all_words.append(stemmed_word)


        processed_sentences.append(filtered_words)



    # ========================================================
    # STEP 7 : TERM FREQUENCY (TF)
    # ========================================================
    #
    # TF measures:
    #
    # How frequently a word appears
    # inside the document.
    #
    # Formula:
    #
    # TF =
    # Number of occurrences of word
    # --------------------------------
    # Total words in document
    #
    # ========================================================

    total_words = len(all_words)

    word_freq = Counter(all_words)

    tf = {}

    for word, count in word_freq.items():

        tf[word] = count / total_words



    # ========================================================
    # STEP 8 : INVERSE DOCUMENT FREQUENCY (IDF)
    # ========================================================
    #
    # PURPOSE:
    #
    # Penalize common words.
    #
    # Formula:
    #
    # IDF = log(N / DF)
    #
    # N  = Total Sentences
    # DF = Number of Sentences
    #      containing the word
    #
    # Rare words get larger scores.
    #
    # ========================================================

    N = len(processed_sentences)

    idf = {}

    for word in word_freq:

        doc_freq = 0

        for sentence_words in processed_sentences:

            if word in sentence_words:
                doc_freq += 1

        idf[word] = math.log(N / (1 + doc_freq))



    # ========================================================
    # STEP 9 : TF-IDF CALCULATION
    # ========================================================
    #
    # Formula:
    #
    # TF-IDF = TF × IDF
    #
    # A word is important if:
    #
    # - It appears many times
    # - But not everywhere
    #
    # ========================================================

    tfidf = {}

    for word in word_freq:

        tfidf[word] = tf[word] * idf[word]



    # ========================================================
    # STEP 10 : SENTENCE SCORING
    # ========================================================
    # PURPOSE:
    #
    # Determine which sentence is most important.
    #
    # Method:
    #
    # Add TF-IDF scores of all words
    # inside a sentence.
    #
    # Higher score = More important.
    #
    # ========================================================

    sentence_scores = []

    for idx, sentence_words in enumerate(processed_sentences):

        score = 0

        for word in sentence_words:

            score += tfidf.get(word, 0)

        sentence_scores.append((idx, score))



    # ========================================================
    # STEP 11 : DETERMINE SUMMARY LENGTH
    # ========================================================
    # Example:
    #
    # 20 sentences
    # compression = 0.30
    #
    # Summary length = 6 sentences
    #
    # ========================================================

    summary_size = max(
        1,
        int(len(sentences) * compression_rate)
    )



    # ========================================================
    # STEP 12 : SELECT TOP SENTENCES
    # ========================================================
    #
    # Sort sentences by score.
    #
    # Highest score appears first.
    #
    # ========================================================

    top_sentences = sorted(
        sentence_scores,
        key=lambda x: x[1],
        reverse=True
    )[:summary_size]



    # ========================================================
    # STEP 13 : RESTORE ORIGINAL ORDER
    # ========================================================
    #
    # Without this step:
    #
    # Summary sentences may appear
    # in random order.
    #
    # Restoring original order
    # improves readability.
    #
    # ========================================================

    top_sentences = sorted(
        top_sentences,
        key=lambda x: x[0]
    )



    # ========================================================
    # STEP 14 : GENERATE FINAL SUMMARY
    # ========================================================
    #
    # Combine selected sentences.
    #
    # ========================================================

    summary = " ".join(
        sentences[idx]
        for idx, score in top_sentences
    )


    return summary




# ============================================================
# STEP 15 : READ INPUT DOCUMENT
# ============================================================
#
# document.txt contains the article
# or research paper to summarize.
#
# ============================================================

with open("document.txt", "r", encoding="utf-8") as file:

    text = file.read()

# ============================================================
# STEP 16 : GENERATE SUMMARY
# ============================================================
#
# compression_rate = 0.3
#
# Keeps approximately
# 30% of original sentences.
#
# ============================================================

summary = summarize_text(
    text,
    compression_rate=0.3
)

# ============================================================
# STEP 17 : DISPLAY RESULTS
# ============================================================
#
# Final extractive summary
# generated using TF-IDF.
#
# ============================================================

print("\n")
print("=" * 60)
print("GENERATED SUMMARY")
print("=" * 60)
print(summary)
print("=" * 60)

##NER,ER,LEMMATIZATION