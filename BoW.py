import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Membaca dataset
df = pd.read_csv(r'C:\Tugas\Tugas Nana\SEM 6\NLP\Tugas_Kuliah\apple-twitter-sentiment-texts.csv')

if "text" not in df.columns:
    raise ValueError("Kolom 'text' tidak ditemukan dalam dataset!")

# Membersihkan teks (pastikan kolom ada)
df["clean_text"] = df["text"].astype(str).str.lower()

# Inisialisasi CountVectorizer untuk BoW
bow_vectorizer = CountVectorizer(stop_words="english", max_features=1000)
bow_matrix = bow_vectorizer.fit_transform(df["clean_text"])

# Konversi ke DataFrame untuk visualisasi
bow_df = pd.DataFrame(bow_matrix.toarray(), columns=bow_vectorizer.get_feature_names_out())

# Menampilkan beberapa baris pertama hasil BoW
print(bow_df.head())


