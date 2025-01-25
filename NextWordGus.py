import re
import nltk
from nltk import trigrams
from collections import Counter, defaultdict
from nltk.stem import PorterStemmer
import wikipediaapi
import streamlit as st
from math import log2, pow
# Download the NLTK data (if you haven't already)
# Download 'punkt_tab' instead of just 'punkt'
nltk.download('punkt_tab') 

# Initialize Wikipedia API
wiki_wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="NLP Class Assignment - Mohamed Ibrahim, MUST University",
)

# Utility Functions
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()
    return text

def fetch_wikipedia_content(topic):
    page = wiki_wiki.page(topic)
    if page.exists():
        return page.text
    else:
        return ""


def build_corpus(topics, min_words=200000):
    corpus = ""
    for topic in topics:
        content = fetch_wikipedia_content(topic)
        if content:
            corpus += clean_text(content)
        if len(corpus.split()) >= min_words:
            break
    return corpus


# Perform Stemming using the Porter Stemmer
stemmer = PorterStemmer()
def stem_corpus(corpus):
    words = nltk.word_tokenize(corpus)
    print(f"Total number of words collected: {len(words)}")
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = ' '.join(stemmed_words)
    # Tokenize the corpus into words
    tokens = nltk.word_tokenize(stemmed_text.lower())
    return tokens
    
def trigrams_model(tokens):
    # Create trigrams from tokens
    trigrams_model = list(trigrams(tokens))
    # Count the frequency of trigrams
    trigrams_counts = Counter(trigrams_model)
    # Display the most common trigrams
    print("Most common trigrams:")
    for trigram, count in trigrams_counts.most_common(5):
        print(f"{trigram}: {count}")
    return trigrams_counts, trigrams_model

def bigrams_model(tokens):
    # Count the frequency of bigrams
    bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))
    vocabulary_size = len(set(tokens))  # Vocabulary size (unique tokens)
    return bigram_counts, vocabulary_size

# Laplace smoothing function for trigrams
def laplace_smoothing(trigram, trigrams_counts, bigram_counts, vocab_size):
    bigram = trigram[:2]  # (w1, w2)
    trigram_count = trigrams_counts[trigram]
    bigram_count = bigram_counts[bigram] if bigram in bigram_counts else 0
    return (trigram_count + 1) / (bigram_count + vocab_size)



# Calculate perplexity
def calculate_perplexity(trigrams_model, trigrams_counts, bigram_counts, vocab_size):
    log_prob_sum = 0
    for trigram in trigrams_model:
        prob = laplace_smoothing(trigram, trigrams_counts, bigram_counts, vocab_size)
        log_prob_sum += log2(prob)
    N = len(trigrams_model)
    return pow(2, -log_prob_sum / N)


# Prediction function
def predict_next_word(input_text, trigrams_counts):
    words = nltk.word_tokenize(input_text.lower())
    # Perform stemming on the input text
    stemmed_input = [stemmer.stem(word) for word in words]
    if len(words) < 2:
        return "Type at least two words to get predictions!"
    
    last_bigram = tuple(stemmed_input[-2:])  # Get last two words (bigram)
    print(f"Last bigram: {last_bigram}")  # Debug print
    suggestions = {trigram[2]: count for trigram, count in trigrams_counts.items() if trigram[:2] == last_bigram}
    if not suggestions:
        return "No predictions available for the given input."
    sorted_suggestions = sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
    return sorted_suggestions[:5]  # Return top 5 predictions



# Streamlit Interface with Session State
st.title("Trigram Language Model with Autocomplete")
st.write("Enter a topic to build a corpus and test autocomplete suggestions.")

# Initialize session state variables if not already set
if "trigram_counts" not in st.session_state:
    st.session_state.trigram_counts = None
if "bigram_counts" not in st.session_state:
    st.session_state.bigram_counts = None
if "vocabulary_size" not in st.session_state:
    st.session_state.vocabulary_size = None
if "corpus_built" not in st.session_state:
    st.session_state.corpus_built = False

# Input Section
topics_input = st.text_input("Enter topics (comma-separated):", "")
min_words = st.number_input(
    "Minimum words in corpus:", min_value=10000, value=200000, step=5000
)

if st.button("Build Corpus and Train Model"):
    topics = ['Economics', 'Politics', 'Sports', 'Finance', 'Technology', 'Science', 
              'Art', 'History', 'Philosophy', 'Literature', 'Music', 'Film', 'Theater', 
              'Dance', 'Architecture', 'Design', 'Fashion', 'Food', 'Travel', 'Health', 
              'Education', 'Environment', 'Physics' , 'Mathematics' , 'Chemistry' , 'Biology',
              'Astronomy', 'Geology', 'Psychology', 'Neuroscience', 'Anthropology', 'Sociology'
            ]
    topics += [topic.strip() for topic in topics_input.split(",")]
    corpus = build_corpus(topics, min_words=min_words)
    if corpus:
        tokens = stem_corpus(corpus)
        trigram_counts, trigram_model = trigrams_model(tokens)
        bigram_counts, vocabulary_size = bigrams_model(tokens)

        # Save results in session state
        st.session_state.trigram_counts = trigram_counts
        st.session_state.bigram_counts = bigram_counts
        st.session_state.vocabulary_size = vocabulary_size
        st.session_state.corpus_built = True

        st.success("Corpus built successfully!")
        perplexity = calculate_perplexity(trigram_model, trigram_counts, bigram_counts, vocabulary_size)
        st.write(f"Perplexity of the test corpus: {perplexity}")
        st.sidebar.title("Corpus Statistics")
        st.sidebar.write(f"Total words in corpus: {len(tokens)}")
        st.sidebar.write(f"Total trigrams generated: {len(trigram_model)}")
    else:
        st.error("Failed to fetch content for the provided topics.")

# Autocomplete Section
st.write("Autocomplete")
input_text = st.text_input("Enter a prefix (1 or 2 words):")

if input_text:
    if st.session_state.trigram_counts:
        predictions = predict_next_word(input_text, st.session_state.trigram_counts)
        if isinstance(predictions, str):
            st.write(predictions)  # If it's a message like "Type at least two words..."
        else:
            st.write("Predicted next words:")
            for word, count in predictions:
                st.write(f"{word}: {count}")
    else:
        st.error("Model is not trained yet. Please build the corpus and train the model first.")
