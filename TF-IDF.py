import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Membaca dataset
df = pd.read_csv(r'C:\Tugas\Tugas Nana\SEM 6\NLP\Tugas_Kuliah\apple-twitter-sentiment-texts.csv')

# Pastikan kolom "text" ada dalam dataset
if "text" not in df.columns:
    raise ValueError("Kolom 'text' tidak ditemukan dalam dataset!")

# Membersihkan teks (pastikan kolom ada)
df["clean_text"] = df["text"].astype(str).str.lower()

# Inisialisasi TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
tfidf_matrix = tfidf_vectorizer.fit_transform(df["clean_text"])

# Konversi ke DataFrame untuk visualisasi
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Menampilkan beberapa baris pertama hasil TF-IDF
print(tfidf_df.head())
