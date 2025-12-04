import os, math, json, csv
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

#This is CIFAR-10 root which creates a folder "data" to store the dataset on system.
#The Output Folder is "MVIT_C10" Where all the Trained Data set is stored.
#Where "artifacts" folder will have the "classes,txt", "config.json", and "metrics.csv" files which have information about dataset.
#And "checkpoints" folder with "best.pth" and "last.pth" files which have weights of the trained model.
#the folder "tb" will have the TensorBoard logs.
DATA_DIR   = os.environ.get("CIFAR10_DIR", "data")
OUT_DIR    = os.environ.get("MVIT_OUT_DIR", "MVIT_C10")

MODEL_NAME = "mobilevit_s"
IMG_SIZE   = 224
BATCH_SIZE = 128
EPOCHS     = 10
WORKERS    = 4
BASE_LR    = 1e-3
WEIGHT_DEC = 0.05
WARMUP_E   = 5
LABEL_SMOOTH = 0.1
FREEZE_HEADONLY_E = 3   #training only the head first and then unfreeze


def topk_accuracy(logits, targets, topk=(1,)):
    maxk = max(topk)
    bs = targets.size(0)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    out = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        out.append((correct_k * 100.0 / bs).item())
    return out


class WarmupCosine:
    def __init__(self, optimizer, warmup_steps, total_steps):
        self.opt = optimizer
        self.wu = max(1, warmup_steps)
        self.T = max(self.wu + 1, total_steps)
        self.step_no = 0

    def step(self):
        self.step_no += 1
        if self.step_no <= self.wu:
            scale = self.step_no / float(self.wu)
        else:
            progress = (self.step_no - self.wu) / float(self.T - self.wu)
            scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for pg in self.opt.param_groups:
            pg["lr"] = pg["base_lr"] * scale


def make_cifar10_loaders(root, img_size=224, batch=128, workers=4):
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([ transforms.Resize(img_size), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),])
    test_tf = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
        transforms.Normalize(mean, std),])

    train_ds = datasets.CIFAR10(root=root, train=True,  download=True, transform=train_tf)
    val_ds   = datasets.CIFAR10(root=root, train=False, download=True, transform=test_tf)

    train_loader = DataLoader( train_ds, batch_size=batch, shuffle=True,
        num_workers=workers, pin_memory=True )
    val_loader   = DataLoader( val_ds,   batch_size=batch, shuffle=False,
        num_workers=workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes, len(train_ds.classes)


def train_one_epoch(model, loader, optimizer, device, scaler,
                    label_smoothing, writer, step_offset):
    model.train()
    loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    run_loss = run_acc1 = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", ncols=100)

    for i, (x, y) in enumerate(pbar):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(x)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            acc1 = topk_accuracy(logits, y, (1,))[0]

        bs = x.size(0)
        run_loss += loss.item() * bs
        run_acc1 += acc1 * bs
        n += bs

        avg_loss = run_loss / n
        avg_acc1 = run_acc1 / n
        step = step_offset + i + 1
        writer.add_scalar("train/loss", avg_loss, step)
        writer.add_scalar("train/acc1", avg_acc1, step)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc1=f"{avg_acc1:.2f}%")

    return avg_loss, avg_acc1


@torch.no_grad()
def validate(model, loader, device, writer, epoch):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    run_loss = run_acc1 = run_acc5 = 0.0
    n = 0
    pbar = tqdm(loader, desc="valid", ncols=100)

    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        acc1, acc5 = topk_accuracy(logits, y, (1, 5))
        bs = x.size(0)
        run_loss += loss.item() * bs
        run_acc1 += acc1 * bs
        run_acc5 += acc5 * bs
        n += bs
        pbar.set_postfix( loss=f"{run_loss/n:.4f}", acc1=f"{run_acc1/n:.2f}%", acc5=f"{run_acc5/n:.2f}%")

    writer.add_scalar("val/loss", run_loss / n, epoch)
    writer.add_scalar("val/acc1", run_acc1 / n, epoch)
    writer.add_scalar("val/acc5", run_acc5 / n, epoch)
    return run_loss / n, run_acc1 / n, run_acc5 / n


def main():
    out_dir = Path(OUT_DIR)
    (out_dir / "tb").mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_loader, val_loader, class_names, num_classes = make_cifar10_loaders(
        DATA_DIR, img_size=IMG_SIZE, batch=BATCH_SIZE, workers=WORKERS)

#Saving all the classes & config for later Predictions use.
    with open(out_dir / "artifacts" / "classes.txt", "w", encoding="utf-8") as f:
        for c in class_names:
            f.write(c + "\n")

    with open(out_dir / "artifacts" / "config.json", "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "CIFAR-10",
            "model": MODEL_NAME,
            "img_size": IMG_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "base_lr": BASE_LR,
            "weight_decay": WEIGHT_DEC,
            "warmup_epochs": WARMUP_E,
            "label_smoothing": LABEL_SMOOTH,
            "freeze_headonly_epochs": FREEZE_HEADONLY_E,
            "num_classes": num_classes }, f, indent=2)

    model = timm.create_model(MODEL_NAME, pretrained=True,
                              num_classes=num_classes).to(device)
    print(f"model: {MODEL_NAME} | params: "
        f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M")

    backbone, head = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if any(k in n for k in ["head", "fc", "classifier"]) else backbone).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone, "lr": BASE_LR * 0.1, "base_lr": BASE_LR * 0.1},
            {"params": head,     "lr": BASE_LR,       "base_lr": BASE_LR},],
        weight_decay=WEIGHT_DEC)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = max(1, len(train_loader) * WARMUP_E)
    sched = WarmupCosine(optimizer, warmup_steps, total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    writer = SummaryWriter(str(out_dir / "tb"))

    if FREEZE_HEADONLY_E > 0:
        print(f"freezing backbone for {FREEZE_HEADONLY_E} epoch(s)")
        for n, p in model.named_parameters():
            if p.requires_grad and not any(k in n for k in ["head", "fc", "classifier"]):
                p.requires_grad = False

    best_acc1 = 0.0
    metrics_csv = out_dir / "artifacts" / "metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc1",
                 "val_loss", "val_acc1", "val_acc5"])

    for epoch in range(1, EPOCHS + 1):
        print(f"\n=== epoch {epoch}/{EPOCHS} ===")
        if epoch == FREEZE_HEADONLY_E + 1:
            print("unfreezing the whole model now")
            for p in model.parameters():
                p.requires_grad = True

        tr_loss, tr_acc1 = train_one_epoch( model, train_loader, optimizer, device, scaler,
            LABEL_SMOOTH, writer, step_offset=(epoch - 1) * len(train_loader))

        for _ in range(len(train_loader)):
            sched.step()

        vl_loss, vl_acc1, vl_acc5 = validate(model, val_loader, device, writer, epoch)

        ckpt = {"model": model.state_dict(),
            "epoch": epoch,
            "best_val_acc1": max(best_acc1, vl_acc1),
            "class_names": class_names}

        torch.save(ckpt, out_dir / "checkpoints" / "last.pth")
        if vl_acc1 > best_acc1:
            best_acc1 = vl_acc1
            torch.save(ckpt, out_dir / "checkpoints" / "best.pth")
            print(f"new BEST! val_acc1={best_acc1:.2f}%")

        with open(metrics_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch,
                f"{tr_loss:.6f}",f"{tr_acc1:.4f}",f"{vl_loss:.6f}",
                f"{vl_acc1:.4f}",f"{vl_acc5:.4f}",])

    writer.close()
    print("\nDone! Check MVIT_C10 folder for checkpoints, logs and metrics.\n"
        "Open TensorBoard with:  tensorboard --logdir MVIT_C10/tb")

if __name__ == "__main__":
    main()
