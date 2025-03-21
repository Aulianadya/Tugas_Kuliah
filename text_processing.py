import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv(r'C:\Tugas\Tugas Nana\SEM 6\Tugas_Kuliah\apple-twitter-sentiment-texts.csv')

# Display dataset info
print(df.head())

# Assuming there is a column named 'text' that contains text data
text_column = 'text'
df = df.dropna(subset=[text_column])

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(f"[{string.punctuation}]", "", text)  # Remove punctuation
    tokens = word_tokenize(text)  # Tokenization
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    return ' '.join(tokens)

df['processed_text'] = df[text_column].apply(preprocess_text)

# Convert Text to Numerical Form

# Bag of Words
vectorizer_bow = CountVectorizer()
X_bow = vectorizer_bow.fit_transform(df['processed_text'])

# TF-IDF
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(df['processed_text'])

# Word Embeddings using Word2Vec
sentences = [text.split() for text in df['processed_text']]
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)

# Get word embeddings for each text by averaging word vectors
def get_embedding(text):
    words = text.split()
    vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

df['word_embedding'] = df['processed_text'].apply(get_embedding)

# Print sample results
print("Bag of Words shape:", X_bow.shape)
print("TF-IDF shape:", X_tfidf.shape)
print("Sample Word Embedding:", df['word_embedding'].iloc[0])