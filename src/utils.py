import torch
import torch.nn as nn
import torch.nn.functional as F


def to_one_hot(batch_input, p):
    a_oh = F.one_hot(batch_input[:, 0], num_classes=p).float()
    b_oh = F.one_hot(batch_input[:, 1], num_classes=p).float()
    return torch.cat([a_oh, b_oh], dim=1)




def save_model(model, path, p, hidden, optimizer=None, epoch=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "p": p,
        "hidden": hidden,
        "epoch": epoch,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
        
    torch.save(checkpoint, path)
    print(f"Model zapisany do {path}")
