import pickle, numpy as np, torch
import torch.nn as nn, torch.optim as optim
from torch.utils.data import Subset, DataLoader
from model import EEGWindowDataset, EEG_CRNN, eval_model, train_epoch
from focal_loss import FocalLoss

def main():
    # 1) Load data + subject‐wise split
    with open('preproc.pkl','rb') as f:
        d = pickle.load(f)
    X, y, subj = d['eeg'], d['label'], d['subj']
    ds = EEGWindowDataset('preproc.pkl')
    unique = np.unique(subj)
    test_subj = unique[-6:]
    train_idx = [i for i,s in enumerate(subj) if s not in test_subj]
    test_idx  = [i for i,s in enumerate(subj) if s in test_subj]
    train_dl = DataLoader(Subset(ds,train_idx), batch_size=64, shuffle=True)
    test_dl  = DataLoader(Subset(ds,test_idx),  batch_size=64)

    # 2) Model + two losses
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_ce    = EEG_CRNN().to(device)
    model_focal = EEG_CRNN().to(device)

    # Cross‑Entropy
    crit_ce = nn.CrossEntropyLoss()
    opt_ce  = optim.Adam(model_ce.parameters(), lr=1e-3, weight_decay=5e-5)
    alpha = torch.tensor([0.5,0.5], device=device)
    crit_focal = FocalLoss(gamma=2.0, alpha=alpha)
    opt_focal  = optim.Adam(model_focal.parameters(), lr=1e-3, weight_decay=5e-5)

    # 3) Train both for a few epochs
    N_EPOCHS = 10
    history = {'ce':[], 'fl':[]}
    for ep in range(1, N_EPOCHS+1):
        # CE
        loss_ce, acc_ce = train_epoch(model_ce, train_dl, crit_ce, opt_ce, device, augment=True)
        m_ce,_,_        = eval_model(model_ce, test_dl, device)
        history['ce'].append((loss_ce, acc_ce, m_ce['roc_auc']))
        # Focal
        loss_fl, acc_fl = train_epoch(model_focal, train_dl, crit_focal, opt_focal, device, augment=True)
        m_fl, _, _      = eval_model(model_focal, test_dl, device)
        history['fl'].append((loss_fl, acc_fl, m_fl['roc_auc']))

        print(f"Epoch {ep:2d} | CE: loss={loss_ce:.4f}, acc={acc_ce:.3f}, AUC={m_ce['roc_auc']:.3f} "
              f"| FL: loss={loss_fl:.4f}, acc={acc_fl:.3f}, AUC={m_fl['roc_auc']:.3f}")

    # 4) Plot comparison
    import matplotlib.pyplot as plt
    epochs = range(1, N_EPOCHS+1)
    ce_auc = [h[2] for h in history['ce']]
    fl_auc = [h[2] for h in history['fl']]

    plt.figure()
    plt.plot(epochs, ce_auc, label='CrossEntropy')
    plt.plot(epochs, fl_auc, label='FocalLoss')
    plt.xlabel('Epoch'); plt.ylabel('Test ROC-AUC')
    plt.title('CE vs. Focal Loss Comparison')
    plt.legend()
    plt.show()

if __name__=='__main__':
    main()
