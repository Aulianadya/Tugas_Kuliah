import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Membaca dataset
df = pd.read_csv(r'C:\Tugas\Tugas Nana\SEM 6\NLP\Tugas_Kuliah\apple-twitter-sentiment-texts.csv')

# Pastikan dataset memiliki kolom "text" dan "sentiment"
if "text" not in df.columns or "sentiment" not in df.columns:
    raise ValueError("Pastikan dataset memiliki kolom 'text' dan 'sentiment'!")

# Preprocessing teks
df["clean_text"] = df["text"].astype(str).str.lower()

# Memisahkan data menjadi fitur (X) dan label (y)
X = df["clean_text"]
y = df["sentiment"]

# Membagi data menjadi training dan testing (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pilih metode representasi teks (BoW atau TF-IDF)
vectorizer = CountVectorizer(stop_words="english", max_features=1000)  # Gunakan BoW
# vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)  # Gunakan TF-IDF

# Transformasi teks ke dalam vektor numerik
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Inisialisasi dan latih model Naïve Bayes
nb_model = MultinomialNB()
nb_model.fit(X_train_vectorized, y_train)

# Prediksi pada data uji
y_pred = nb_model.predict(X_test_vectorized)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))
