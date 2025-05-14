import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from tqdm import tqdm
import joblib

# ==========================
# Custom Dataset Class
# ==========================
class StreetFighterDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ==========================
# MLP Model
# ==========================
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.3),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.model(x)

# ==========================
# Training and Evaluation Functions
# ==========================
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    for inputs, labels in tqdm(dataloader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()
        
        preds = (outputs >= 0.5).float()
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    f1 = f1_score(all_labels, all_preds, average='micro')
    return running_loss / len(dataloader), f1

def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            preds = (outputs >= 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    f1 = f1_score(all_labels, all_preds, average='micro')
    return running_loss / len(dataloader), f1

# ==========================
# Load and Preprocess Data
# ==========================
# Define dtypes to avoid mixed type warnings
bool_cols = ['p1_jumping', 'p1_crouching', 'p1_in_move', 'p2_jumping', 'p2_crouching', 'p2_in_move',
             'up', 'down', 'left', 'right', 'Y', 'B', 'X', 'A', 'L', 'R']
dtype_dict = {col: 'str' for col in bool_cols}  # Read as strings to handle TRUE/FALSE consistently
df = pd.read_csv('game_log.csv', dtype=dtype_dict, low_memory=False)

# Handle missing values
if df.isnull().sum().any():
    print("Warning: Missing values detected. Filling with 0 for boolean columns and median for numeric.")
    df[bool_cols] = df[bool_cols].fillna('FALSE')  # Assume no button press if missing
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Drop unnecessary or constant columns
df.drop(columns=['player_id'], errors='ignore', inplace=True)

# Define label columns
label_cols = ['up', 'down', 'left', 'right', 'Y', 'B', 'X', 'A', 'L', 'R']
feature_cols = [col for col in df.columns if col not in label_cols]

# Convert boolean strings to integers (case-insensitive)
df[bool_cols] = df[bool_cols].apply(lambda x: x.str.upper().map({'TRUE': 1, 'FALSE': 0}))

# Verify conversion
for col in bool_cols:
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column {col} contains non-numeric values after conversion: {df[col].unique()}")

# Split features and labels
X = df[feature_cols].values
y = df[label_cols].values

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Datasets & Loaders (single-threaded)
train_dataset = StreetFighterDataset(X_train, y_train)
val_dataset = StreetFighterDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=64, pin_memory=True)

# ==========================
# Training Setup
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP(input_dim=X.shape[1], output_dim=len(label_cols)).to(device)

# Check for class imbalance and use weighted loss
pos_weights = torch.tensor([1.0 / y_train[:, i].mean() if y_train[:, i].mean() > 0 else 1.0 for i in range(y_train.shape[1])], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Early stopping parameters
patience = 10
best_val_loss = float('inf')
counter = 0
best_model_path = 'mlp_street_fighter_best.pth'

# Mixed precision training (if GPU)
scaler_torch = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

# Training loop
epochs = 50
for epoch in range(epochs):
    train_loss, train_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_f1 = evaluate(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
    
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model with Val Loss: {val_loss:.4f}")
        counter = 0
    else:
        counter += 1
    
    if counter >= patience:
        print("Early stopping triggered")
        break

print(f"Training complete. Best model saved as {best_model_path}")