import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# === 1. Load dataset ===
df = pd.read_csv("C:\Tugas\Tugas Nana\SEM 6\Tugas_Kuliah\Twitter_Data.csv")
df = df[['clean_text', 'category']].dropna()

# Encode label (positif, netral, negatif)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['category'])

texts = df['clean_text'].tolist()
labels = df['label'].tolist()

# === 2. Tokenisasi ===
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
tokens = tokenizer(
    texts,
    truncation=True,
    padding=True,
    max_length=128,
    return_tensors='pt'
)

# === 3. Dataset dan DataLoader ===
class TweetDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Split data
train_idx, val_idx = train_test_split(list(range(len(labels))), test_size=0.2, random_state=42)
train_dataset = TweetDataset(tokens['input_ids'][train_idx],
                             tokens['attention_mask'][train_idx],
                             [labels[i] for i in train_idx])

val_dataset = TweetDataset(tokens['input_ids'][val_idx],
                           tokens['attention_mask'][val_idx],
                           [labels[i] for i in val_idx])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# === 4. Load model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3)
model.to(device)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# === 5. Training loop ===
epochs = 1

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {train_acc:.4f}")

# === 6. Evaluation ===
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

val_acc = correct / total
print(f"Validation Accuracy: {val_acc:.4f}")
