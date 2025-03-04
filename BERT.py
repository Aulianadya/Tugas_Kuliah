import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Cek apakah GPU tersedia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Membaca dataset
file_path = r'C:\Tugas\Tugas Nana\SEM 6\NLP\Tugas_Kuliah\apple-twitter-sentiment-texts.csv'
df = pd.read_csv(file_path)

# Mapping label sentimen (-1 -> 0, 0 -> 1, 1 -> 2)
label_mapping = {-1: 0, 0: 1, 1: 2}
df = df.dropna(subset=['sentiment', 'text'])  # Hapus NaN
df['label'] = df['sentiment'].map(label_mapping)

df = df.dropna(subset=['label'])  # Pastikan tidak ada label NaN
df['label'] = df['label'].astype(int)

# Pisahkan dataset menjadi training & testing (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42
)

# Gunakan tokenizer BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Custom Dataset untuk BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

# Buat dataset dan dataloader
train_dataset = SentimentDataset(X_train, y_train, tokenizer)
test_dataset = SentimentDataset(X_test, y_test, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load model BERT
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)

# Optimizer dan loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

# Evaluasi Model
model.eval()
predictions, actuals = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = (
            batch["input_ids"].to(device),
            batch["attention_mask"].to(device),
            batch["labels"].to(device),
        )
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        actuals.extend(labels.cpu().numpy())

# Menampilkan Hasil Evaluasi
accuracy = accuracy_score(actuals, predictions)
print(f"\nAccuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(actuals, predictions, target_names=["Negative", "Neutral", "Positive"]))
