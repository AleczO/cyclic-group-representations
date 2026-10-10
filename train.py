"""
Example:
    python train.py --p 71 --wd 0.56 --seed 3
    python train.py --p 71 --wd 0.56 --seed 3 --save-snapshots --snapshot-every 100
"""
import argparse
import csv
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from src.dataset import DatasetModulo
from src.model import FNN
from src.utils import save_model, to_one_hot


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=71)
    ap.add_argument("--hidden", type=int, default=None, help="defaults to 2p")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--results", default="results")
    ap.add_argument("--save-snapshots", action="store_true")
    ap.add_argument("--snapshot-every", type=int, default=100)
    ap.add_argument("--print-every", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite a run that has already finished")
    return ap.parse_args()


def main():
    args = parse_args()
    hidden = args.hidden if args.hidden is not None else 2 * args.p

    run_name = f"p{args.p}_wd{args.wd:g}_seed{args.seed}"
    if args.hidden is not None:
        run_name += f"_h{hidden}"
    run_dir = os.path.join(args.results, run_name)

    if os.path.exists(os.path.join(run_dir, "model_final.pt")) and not args.overwrite:
        print(f"[skip] {run_name} already exists (use --overwrite)")
        return

    os.makedirs(run_dir, exist_ok=True)
    if args.save_snapshots:
        os.makedirs(os.path.join(run_dir, "snapshots"), exist_ok=True)

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    model = FNN(args.p, hidden=hidden).to(device)
    dataset = DatasetModulo(args.p)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = torch.nn.CrossEntropyLoss()

    x = to_one_hot(dataset.input, args.p).to(device)
    y = dataset.output.to(device)

    print(f"[start] {run_name} | hidden={hidden} lr={args.lr} epochs={args.epochs} device={device}")

    with open(os.path.join(run_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "loss", "acc"])

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            acc = (logits.argmax(dim=1) == y).float().mean().item()
            writer.writerow([epoch, loss.item(), acc])

            if args.save_snapshots and epoch % args.snapshot_every == 0:
                save_model(model, os.path.join(run_dir, "snapshots", f"model_{epoch:05d}.pt"),
                           p=args.p, hidden=hidden, optimizer=optimizer, epoch=epoch)

            if epoch % args.print_every == 0 or epoch == args.epochs - 1:
                print(f"epoch {epoch:6d} | loss {loss.item():.4f} | acc {acc:.3f}")

    save_model(model, os.path.join(run_dir, "model_final.pt"), p=args.p, hidden=hidden,
               optimizer=optimizer, epoch=args.epochs - 1)
    print(f"[done] {run_name}")


if __name__ == "__main__":
    main()