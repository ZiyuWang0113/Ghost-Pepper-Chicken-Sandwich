
import argparse
import os
import json
import torch
import torchvision as tv
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np 

def parse_args():
    p = argparse.ArgumentParser(description="fine tune effnet")
    p.add_argument("--data")
    p.add_argument("--model",  default="efficientnet_b3")
    p.add_argument("--size",   type=int, default=300) # input image size
    p.add_argument("--batch",  type=int, default=64)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--lr",     type=float, default=3e-4)
    p.add_argument("--out",    default="checkpoints") # where to output
    p.add_argument("--workers", type=int, default=8) # applicable to oscar, number of workers for job
    p.add_argument("--freeze_epochs", type=int, default=12) # num of epochs to use weight freezing for
    p.add_argument("--patience", type=int, default=7) # num of epochs to wait for val improvement
    return p.parse_args()




def make_loaders(root, img_size, batch_size, num_workers):
    from timm.data import create_transform 
    train_tf = create_transform(img_size, is_training=True,
                                auto_augment='rand-m9-mstd0.5', # just an automatic augmentation for effnet
                                interpolation='bicubic',
                                re_prob=0.1, 
                                hflip=0.5,
                                vflip=0.0) 

    val_tf   = create_transform(img_size, is_training=False, interpolation='bicubic')

    def get_folder_path(split):
        path = os.path.join(root, split)
        return path

    train_dir = get_folder_path("Train")
    val_dir = get_folder_path("Validation")
    test_dir = get_folder_path("Test")
    train_ds = tv.datasets.ImageFolder(train_dir, train_tf)
    val_ds   = tv.datasets.ImageFolder(val_dir,   val_tf)
    test_ds  = tv.datasets.ImageFolder(test_dir,  val_tf)


    # extract labels from training set
    labels = [label for _, label in train_ds.samples] # .samples avoids actually loading the images
    class_counts = torch.bincount(torch.tensor(labels))
    num_classes = len(class_counts)
    weights = 1. / class_counts.float()
    sample_weights = weights[labels] # map class weight to each sample 

    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    use_shuffle = False 


    # create dict of common args to use on pytorch
    kwargs = dict(batch_size=batch_size,
                  num_workers=num_workers,
                  prefetch_factor=4,
                  pin_memory=True,
                  persistent_workers=(num_workers > 0))

    train_dl = DataLoader(train_ds, sampler=sampler, shuffle=use_shuffle, **kwargs)
    val_dl   = DataLoader(val_ds,   shuffle=False, **kwargs)
    test_dl  = DataLoader(test_ds,  shuffle=False, **kwargs)
    return train_dl, val_dl, test_dl

@torch.no_grad() 
def evaluate(model, loader, device):
    model.eval() # model on evaluation mode 
    correct = 0
    total = 0
    ys, ps = [], [] #  for storing the true lables and predictions

    with torch.cuda.amp.autocast(enabled=(device=="cuda")):
        for x, y in loader:
            x = x.to(device, non_blocking=True, memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            output = model(x)
            # cross entropy loss
            pred = output.argmax(1) 
            correct += (pred == y).sum().item()
            total += y.size(0)
            ys.extend(y.cpu().tolist())
            ps.extend(pred.cpu().tolist())

    accuracy = correct / total if total > 0 else 0.0
    return accuracy, ys, ps

args   = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


if device == "cuda":
    torch.backends.cudnn.benchmark = True
if hasattr(torch, 'set_float32_matmul_precision'):
    torch.set_float32_matmul_precision("high")

os.makedirs(args.out, exist_ok=True)


train_dl, val_dl, test_dl = make_loaders(args.data, args.size, args.batch, args.workers)

model = timm.create_model(
    args.model,
    pretrained=True,
    num_classes=2 
).to(device, memory_format=torch.channels_last)

if args.freeze_epochs > 0:
    print(f"Freezing backbone")
    for param in model.parameters():
        param.requires_grad = False

    unfrozen_layers = []
        for param in model.classifier.parameters():
            param.requires_grad = True
        unfrozen_layers.append('classifier')
   

trainable_params = [p for p in model.parameters() if p.requires_grad]
opt   = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-2)

crit  = nn.CrossEntropyLoss(label_smoothing=0.05) 
scaler = torch.cuda.amp.GradScaler(enabled=(device=="cuda"))

initial_T_max = args.epochs
sched = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=initial_T_max, eta_min=args.lr / 50 
)


best_val_acc = 0.0
best_ep = 0 
patience = args.patience 
unfreeze_epoch = args.freeze_epochs 
save_path = os.path.join(args.out, "best_model.pth")


for ep in range(args.epochs):
    if args.freeze_epochs > 0 and ep == unfreeze_epoch:
        for param in model.parameters():
            param.requires_grad = True


        new_lr = args.lr * 0.01
        opt = torch.optim.AdamW(model.parameters(), lr=new_lr, weight_decay=1e-2)
        remaining_epochs = args.epochs - unfreeze_epoch
        new_eta_min = new_lr / 50 
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,
                                                            T_max=remaining_epochs,
                                                            eta_min=new_eta_min)
        print(f"new scheduler created")


    #  Train
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_dl)

    for i, (xb, yb) in enumerate(train_dl):
        xb = xb.to(device, non_blocking=True, memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        opt.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device=="cuda")):
            output = model(xb)
            loss = crit(output, yb)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

    val_acc, _, _ = evaluate(model, val_dl, device)

    if val_acc > best_val_acc + 1e-4: 
        print(f"val acc improved, save model")
        best_val_acc = val_acc
        best_ep = ep
        torch.save(model.state_dict(), save_path)
    elif ep - best_ep >= patience:
        print(f"\nNo val improvement for {patience} epochs, stopping aerly")
        break # exit training loop
    else:
            print(f" No improvement")


    # Step LR scheduler
    sched.step()


model.load_state_dict(torch.load(save_path, map_location=device))

test_acc, ys_true, ys_pred = evaluate(model, test_dl, device)

target_names = ["fake (0)", "real (1)"]
report = classification_report(ys_true, ys_pred, target_names=target_names, digits=4)
print(report)
cm = confusion_matrix(ys_true, ys_pred)
print("\nConfusion Matrix (Rows: True Cols: Pred):")
print(f"       {target_names[0]}  {target_names[1]}")
print(f"  {target_names[0]}: {cm[0,0]:>5}  {cm[0,1]:>5}")
print(f"  {target_names[1]}: {cm[1,0]:>5}  {cm[1,1]:>5}")
cm_list = cm.tolist() 
metrics = {
    "best_validation_accuracy": best_val_acc,
    "best_epoch": best_ep,
    "final_test_accuracy": test_acc,
    "classification_report": report,
    "confusion_matrix (tn, fp, fn, tp)": cm_list,
    "stopped_epoch": ep 
}
metrics_path = os.path.join(args.out, "test_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f)




