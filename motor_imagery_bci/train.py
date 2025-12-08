import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from load_dataset import load_bci_iv_2b
from preprocess import preprocess_dataset
from utils import create_dataloaders
from model_cnn import MotorImageryCNN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


X, y = load_bci_iv_2b("./dataset/", subjects=[1])


X = preprocess_dataset(X)


train_loader, val_loader, test_loader = create_dataloaders(X, y)


model = MotorImageryCNN(
    n_channels=X.shape[1],
    n_samples=X.shape[2],
    n_classes=2
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    train_losses = []

    for Xb, yb in train_loader:
        Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)

        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

  
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for Xv, yv in val_loader:
            Xv = Xv.to(DEVICE)
            out = model(Xv)
            pred = torch.argmax(out, dim=1).cpu().numpy()
            preds.extend(pred)
            labels.extend(yv.numpy())

    acc = accuracy_score(labels, preds)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss = {sum(train_losses)/len(train_losses):.4f} | Val Acc = {acc:.4f}")


model.eval()
test_preds = []
test_labels = []

with torch.no_grad():
    for Xt, yt in test_loader:
        Xt = Xt.to(DEVICE)
        out = model(Xt)
        pred = torch.argmax(out, dim=1).cpu().numpy()
        test_preds.extend(pred)
        test_labels.extend(yt.numpy())

test_acc = accuracy_score(test_labels, test_preds)
print("TEST ACCURACY =", test_acc)
