import os
import csv
import matplotlib.pyplot as plt

#Change this Path to the directory where all the metrics CSV files are stored.
BASE_DIR = r"D:\%%SLU\Sem 3\AI-Capstone\MVIT\Metrices of all models"

MVIT_METRICS_PATH = os.path.join(BASE_DIR, "MVITmetrics.csv")
MNV3_METRICS_PATH = os.path.join(BASE_DIR, "MNV3metrics.csv")
MNV2_METRICS_PATH = os.path.join(BASE_DIR, "MNV2metrics.csv")
EFFICIENCY_PATH   = os.path.join(BASE_DIR, "efficiency_metrics.csv")

def load_metrics(csv_path):
    epochs = []
    train_loss = []
    train_acc1 = []
    val_loss = []
    val_acc1 = []
    val_acc5 = []

    with open(csv_path, mode="r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc1.append(float(row["train_acc1"]))
            val_loss.append(float(row["val_loss"]))
            val_acc1.append(float(row["val_acc1"]))
            val_acc5.append(float(row["val_acc5"]))
    return {"epochs": epochs,
        "train_loss": train_loss,
        "train_acc1": train_acc1,
        "val_loss": val_loss,
        "val_acc1": val_acc1,
        "val_acc5": val_acc5, }

#Loads all three models metrics
metrics_mvit = load_metrics(MVIT_METRICS_PATH)
metrics_mnv3 = load_metrics(MNV3_METRICS_PATH)
metrics_mnv2 = load_metrics(MNV2_METRICS_PATH)

# 1) Core Figure: Validation Accuracy vs Epoch
fig1, ax1 = plt.subplots()

ax1.plot(metrics_mnv2["epochs"], metrics_mnv2["val_acc1"],
         label="MobileNetV2", color="black")
ax1.plot(metrics_mnv3["epochs"], metrics_mnv3["val_acc1"],
         label="MobileNetV3-Small", color="blue")
ax1.plot(metrics_mvit["epochs"], metrics_mvit["val_acc1"],
         label="MobileViT-S", color="red")

ax1.set_title("Validation Accuracy vs Epoch (All Models)")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Validation Accuracy@1 (%)")
ax1.grid(True)
ax1.legend()

out_path_1 = os.path.join(BASE_DIR, "val_acc_all_models.png")
fig1.savefig(out_path_1, bbox_inches="tight")
plt.close(fig1)
print("Saved:", out_path_1)

#2)Figure: Validation Loss vs Epoch
fig2, ax2 = plt.subplots()

ax2.plot(metrics_mnv2["epochs"], metrics_mnv2["val_loss"],
         label="MobileNetV2", color="blue")
ax2.plot(metrics_mnv3["epochs"], metrics_mnv3["val_loss"],
         label="MobileNetV3-Small", color="black")
ax2.plot(metrics_mvit["epochs"], metrics_mvit["val_loss"],
         label="MobileViT-S", color="green")

ax2.set_title("Validation Loss vs Epoch (All Models)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Validation Loss")
ax2.grid(True)
ax2.legend()

out_path_2 = os.path.join(BASE_DIR, "val_loss_all_models.png")
fig2.savefig(out_path_2, bbox_inches="tight")
plt.close(fig2)
print("Saved:", out_path_2)

#3)Figure: Final Performance Bar Chart
final_v2  = metrics_mnv2["val_acc1"][-1]
final_v3  = metrics_mnv3["val_acc1"][-1]
final_mvit = metrics_mvit["val_acc1"][-1]

models = ["MobileNetV2", "MobileNetV3-Small", "MobileViT-S"]
final_accs = [final_v2, final_v3, final_mvit]
colors = ["yellow", "green", "red"]

fig3, ax3 = plt.subplots()
bars = ax3.bar(models, final_accs, color=colors)

ax3.set_title("Final Validation Accuracy@1 (Summary)")
ax3.set_ylabel("Validation Accuracy@1 (%)")
ax3.set_ylim(0, 100)
ax3.grid(axis="y", linestyle="--", alpha=0.5)

for bar, acc in zip(bars, final_accs):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2.0,
             height + 0.3,
             f"{acc:.2f}%",
             ha="center", va="bottom", fontsize=8)

out_path_3 = os.path.join(BASE_DIR, "final_val_acc_bar.png")
fig3.savefig(out_path_3, bbox_inches="tight")
plt.close(fig3)
print("Saved:", out_path_3)

#4)Efficiency Figure: Latency Bar Chart
eff_models = []
eff_latency = []

with open(EFFICIENCY_PATH, mode="r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        eff_models.append(row["model"])
        eff_latency.append(float(row["latency_ms"]))


color_map = []
for name in eff_models:
    if "MobileNetV2" in name:
        color_map.append("blue")
    elif "MobileNetV3" in name:
        color_map.append("green")
    elif "MobileViT" in name:
        color_map.append("red")
    else:
        color_map.append("gray")

fig4, ax4 = plt.subplots()
bars2 = ax4.bar(eff_models, eff_latency, color=color_map)

ax4.set_title("Model Efficiency: Inference Latency")
ax4.set_ylabel("Latency (ms per image)")
ax4.grid(axis="y", linestyle="--", alpha=0.5)

for bar, lat in zip(bars2, eff_latency):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2.0,
             height + 0.3,
             f"{lat:.2f} ms",
             ha="center", va="bottom", fontsize=8)

out_path_4 = os.path.join(BASE_DIR, "efficiency_latency_bar.png")
fig4.savefig(out_path_4, bbox_inches="tight")
plt.close(fig4)
print("Saved:", out_path_4)

print("\nAll plots generated successfully.")
