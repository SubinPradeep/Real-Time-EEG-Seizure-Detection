import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt

class EEGWindowDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            d = pickle.load(f)
        self.X    = torch.Tensor(d["eeg"])
        self.y    = torch.LongTensor(d["label"])
        self.subj = np.array(d["subj"], dtype=int)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subj[idx]

class EEG_CRNN(nn.Module):
    def __init__(self, n_ch=19, hidden=32, n_layers=1, dropout=0.7):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1,51), stride=(1,4)),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((1,2)),
            nn.Conv2d(64,128, kernel_size=(1,25), stride=(1,4)),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((n_ch,1))
        )
        self.rnn = nn.LSTM(
            input_size=128,
            hidden_size=hidden,
            num_layers=n_layers,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(hidden, 2)

    def forward(self, x):
        x = x.unsqueeze(1)               # (B,1,C,T)
        x = self.conv(x).squeeze(-1)     # (B,128,C)
        x = x.permute(0,2,1)             # (B,C,128)
        rnn_out,_ = self.rnn(x)          # (B,C,hidden)
        h = rnn_out[:,-1,:]              # (B,hidden)
        h = self.dropout(h)
        return self.fc(h)                # (B,2)

def train_epoch(model, loader, criterion, optimizer, device, augment=False):
    model.train()
    losses, preds, labels = [], [], []
    for X, y, _ in loader:
        if augment:
            noise = torch.randn_like(X) * 0.05
            X = X + noise
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds.extend(logits.argmax(1).cpu().numpy())
        labels.extend(y.cpu().numpy())
    return np.mean(losses), accuracy_score(labels, preds)

def eval_model(model, loader, device):
    model.eval()
    probs, preds, labels = [], [], []
    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            logits = model(X)
            p = torch.softmax(logits,1)[:,1].cpu().numpy()
            preds.extend(logits.argmax(1).cpu().numpy())
            probs.extend(p.tolist())
            labels.extend(y.numpy())
    metrics = {
        "acc": accuracy_score(labels, preds),
        "prec": precision_score(labels, preds, zero_division=0),
        "rec": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "roc_auc": roc_auc_score(labels, probs),
        "pr_auc": auc(*precision_recall_curve(labels, probs)[1::-1])
    }
    fpr, tpr, _ = roc_curve(labels, probs)
    cm = confusion_matrix(labels, preds)
    return metrics, (fpr, tpr), cm

def main(
    data_path       = "preproc.pkl",
    holdout_subjects= 6,           
    batch_size      = 64,
    lr              = 1e-3,
    weight_decay    = 5e-5,       
    epochs          = 40,
    patience        = 7,
    device          = None
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = EEGWindowDataset(data_path)
    subjs = ds.subj
    unique = np.unique(subjs)
    test_subj = unique[-holdout_subjects:]
    train_idx = [i for i,s in enumerate(subjs) if s not in test_subj]
    test_idx  = [i for i,s in enumerate(subjs) if s in test_subj]

    train_ds = Subset(ds, train_idx)
    test_ds  = Subset(ds, test_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = EEG_CRNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_auc = 0
    no_improve=0
    history = {"loss":[], "acc":[], "val_auc":[]}

    for ep in range(1, epochs+1):
        loss, acc = train_epoch(model, train_loader, criterion, optimizer, device, augment=True)
        m, (fpr, tpr), _ = eval_model(model, test_loader, device)
        val_auc = np.trapezoid(tpr, fpr)  

        history["loss"].append(loss)
        history["acc"].append(acc)
        history["val_auc"].append(val_auc)
        print(f"Epoch {ep}/{epochs}  loss={loss:.4f} acc={acc:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    metrics, (fpr, tpr), cm = eval_model(model, test_loader, device)
    print("\nFinal Test Metrics:", metrics)

    plt.figure(); plt.plot(history["loss"], label="Loss"); plt.title("Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.figure(); plt.plot(history["acc"], label="Acc");  plt.title("Train Acc")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.figure(); plt.plot(history["val_auc"], label="Val ROC-AUC"); plt.title("Val ROC-AUC")
    plt.xlabel("Epoch"); plt.ylabel("ROC-AUC")

    plt.figure(); plt.plot(fpr, tpr); plt.title("Test ROC Curve")
    plt.xlabel("FPR"); plt.ylabel("TPR")

    plt.figure(); 
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0,1], ["bckg","seiz"])
    plt.yticks([0,1], ["bckg","seiz"])
    plt.xlabel("Predicted"); plt.ylabel("True")

    plt.show()

if __name__ == "__main__":
    main()