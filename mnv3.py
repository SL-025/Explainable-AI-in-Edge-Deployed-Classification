import os, argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

from utils_data import make_dataloaders

@torch.no_grad()
def evaluate(model, loader, device):
    """Return average loss and accuracy on a loader."""
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss_sum += criterion(logits, y).item() * y.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        total    += y.size(0)
    return loss_sum/total, correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--logdir", type=str, default="runs/mnv3_cifar10_full")
    ap.add_argument("--ckpt", type=str, default="checkpoints/mnv3_cifar10_best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, test_loader = make_dataloaders(
        batch_size=args.batch_size, num_workers=args.num_workers)

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, 10)
    model = model.to(device)

    #loss + optimizer + scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    tb = SummaryWriter(args.logdir)

    best_val_acc = 0.0
    step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            tb.add_scalar("loss/train_iter", loss.item(), step)
            step += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        tb.add_scalar("loss/train_epoch", running_loss/len(train_loader.dataset), epoch)
        tb.add_scalar("loss/val", val_loss, epoch)
        tb.add_scalar("acc/val", val_acc, epoch)
        tb.flush()  

        print(f"Epoch {epoch:02d}: val_acc={val_acc:.3f}  val_loss={val_loss:.4f}")
        scheduler.step()

        #saving the best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
            torch.save(model.state_dict(), args.ckpt)
            print(f"saved best → {args.ckpt} (val_acc={best_val_acc:.3f})")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"TEST: acc={test_acc:.3f}  loss={test_loss:.4f}")
    tb.close()

if __name__ == "__main__":
    main()
