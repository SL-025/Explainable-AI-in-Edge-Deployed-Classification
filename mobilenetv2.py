import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, Subset

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Name of the device we used:", device)

DATA_ROOT = r"D:\%SLU\Sem 3\AI-Capstone\Mobilenetv2\Capstone\data"
WEIGHTS_OUT = r"D:\%SLU\Sem 3\AI-Capstone\Mobilenetv2\mobilenetv2_tiny.pth"

#5 classes we will use, from CIFAR-10 indices:
#0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
#5 dog, 6 frog, 7 horse, 8 ship, 9 truck
KEEP_CLASSES = [2, 3, 5, 7, 8]
CLASS_NAMES = {2: "bird", 3: "cat", 5: "dog", 7: "horse", 8: "ship"}

label_map = {}
idx_counter = 0
for c in KEEP_CLASSES:
    label_map[c] = idx_counter
    idx_counter = idx_counter + 1

#Preprocessing, resize to 224x224, change toTensor, normalize with ImageNet mean/std
resize_step = transforms.Resize((224, 224))
to_tensor_step = transforms.ToTensor()
normalize_step = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225] )
transform = transforms.Compose([resize_step, to_tensor_step, normalize_step])

# 3) Download CIFAR-10 train and test once in DATA_ROOT
full_train = datasets.CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform )
full_val = datasets.CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform )

#building a tiny, balanced subset: N images per kept class
def build_small_subset(dataset, per_class=50):
    by_class = {}
    for c in KEEP_CLASSES:
        by_class[c] = []

    i = 0
    while i < len(dataset):
        item = dataset[i]
        label = int(item[1])
        if label in by_class:
            by_class[label].append(i)
        i = i + 1

    selected_indices = []
    j = 0
    while j < len(KEEP_CLASSES):
        c = KEEP_CLASSES[j]
        pool = by_class[c]
        random.shuffle(pool)
        k = 0
        count_taken = 0
        while k < len(pool) and count_taken < per_class:
            selected_indices.append(pool[k])
            k = k + 1
            count_taken = count_taken + 1
        j = j + 1

    subset = Subset(dataset, selected_indices)
    return subset

#Creating tiny train/val subsets (50 per class)
train_ds = build_small_subset(full_train, per_class=50)
val_ds = build_small_subset(full_val, per_class=50)
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

print("Tiny train size:", len(train_ds), "| Tiny val size:", len(val_ds))
class_list_print = []
for c in KEEP_CLASSES:
    class_list_print.append(CLASS_NAMES[c])
print("Classes (in order):", class_list_print)

#Taking ImageNet-pretrained MobileNetV2, freezing the feature extractor (backbone)
base_model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 )

# Freeze all feature extractor parameters to be speedy
for p in base_model.features.parameters():
    p.requires_grad = False

num_classes = len(KEEP_CLASSES)
in_features = base_model.classifier[1].in_features
new_head = nn.Linear(in_features, num_classes)
base_model.classifier[1] = new_head
base_model = base_model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(base_model.parameters(), lr=1e-3) #SO learning rate is 0.001

def map_labels_to_small_range(labels_tensor, device_str):
    mapped_list = []
    i = 0
    while i < labels_tensor.size(0):
        original_label = int(labels_tensor[i].item())
        mapped_value = label_map[original_label]
        mapped_list.append(mapped_value)
        i = i + 1
    mapped_tensor = torch.tensor(mapped_list, device=device_str, dtype=torch.long)
    return mapped_tensor

#training for 1 epoch now
epochs = 1
epoch_index = 0
while epoch_index < epochs:
    base_model.train()
    running_loss = 0.0

    for batch in train_dl:
        images = batch[0]
        labels_original = batch[1]
        #mapping labels to 0..4 according to KEEP_CLASSES order
        labels_small = map_labels_to_small_range(labels_original, device)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = base_model(images)
        loss = criterion(outputs, labels_small)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + float(loss.item())

    #average of train loss for logging
    if len(train_dl) > 0:
        avg_loss = running_loss / len(train_dl)
    else:
        avg_loss = 0.0

    #Calculating the  Validation
    base_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dl:
            images_val = batch[0]
            labels_val_original = batch[1]

            labels_val_small = map_labels_to_small_range(labels_val_original, device)
            images_val = images_val.to(device)

            logits_val = base_model(images_val)
            preds = torch.argmax(logits_val, dim=1)

            correct = correct + int((preds == labels_val_small).sum().item())
            total = total + int(labels_val_small.size(0))

    if total > 0:
        val_acc = (100.0 * correct) / total
    else:
        val_acc = 0.0

    print("Epoch [{}/{}]  Loss: {:.4f}  |  Val Acc: {:.2f}%".format(
        epoch_index + 1, epochs, avg_loss, val_acc ))
    epoch_index = epoch_index + 1

#Saving the trained tiny model weights
torch.save(base_model.state_dict(), WEIGHTS_OUT)
print("Saved weights to:", WEIGHTS_OUT)