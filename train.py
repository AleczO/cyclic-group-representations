from src.dataset import DatasetModulo
from src.model import FNN
from src.utils import to_one_hot, save_model

import torch

device = torch.device("cuda")

p = 107
EPOCHS = 10000
HIDDEN = 2 * p
SEED = 1

torch.manual_seed(SEED)

model = FNN(p, hidden=HIDDEN).to(device)
dataset = DatasetModulo(p)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.4)
lossFN = torch.nn.CrossEntropyLoss()

x = to_one_hot(dataset.input, p).to(device)
y = dataset.output.to(device)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(x)
    loss = lossFN(logits, y)

    loss.backward()

    optimizer.step()

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        acc = (logits.argmax(dim=1) == y).float().mean().item()

        save_model(model, "model.pt", p=p, hidden=HIDDEN, optimizer=optimizer, epoch=EPOCHS)
        print(f"epoka {epoch:5d} | loss {loss.item():.4f} | acc {acc:.3f}")