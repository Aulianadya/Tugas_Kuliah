import pandas as pd
import gensim
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize
import string

# Pastikan Anda telah mengunduh tokenizer NLTK
nltk.download('punkt')

# Membaca dataset
df = pd.read_csv(r'C:\Tugas\Tugas Nana\SEM 6\NLP\Tugas_Kuliah\apple-twitter-sentiment-texts.csv')

# Pastikan kolom "text" ada dalam dataset
if "text" not in df.columns:
    raise ValueError("Kolom 'text' tidak ditemukan dalam dataset!")

# Membersihkan teks
def preprocess_text(text):
    text = text.lower()  # Konversi ke huruf kecil
    text = text.translate(str.maketrans("", "", string.punctuation))  # Hapus tanda baca
    tokens = word_tokenize(text)  # Tokenisasi
    return tokens

# Terapkan preprocessing ke semua teks
df["tokens"] = df["text"].astype(str).apply(preprocess_text)

# Melatih model Word2Vec
word2vec_model = Word2Vec(sentences=df["tokens"], vector_size=100, window=5, min_count=2, workers=4)

# Menampilkan vektor dari kata tertentu (misalnya "apple")
word = "apple"
if word in word2vec_model.wv:
    print(f"Vektor untuk kata '{word}':\n", word2vec_model.wv[word])
else:
    print(f"Kata '{word}' tidak ditemukan dalam model.")

# Menampilkan kata-kata yang paling mirip dengan "apple"
if word in word2vec_model.wv:
    print(f"Kata-kata yang mirip dengan '{word}':\n", word2vec_model.wv.most_similar(word))
