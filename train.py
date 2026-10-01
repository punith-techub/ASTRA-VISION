from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

CLASSES = ["aircraft", "helicopter", "drone", "military-vehicle", "naval"]
DISPLAY_NAMES = {
    "aircraft": "Aircraft",
    "helicopter": "Helicopter",
    "drone": "Drone / UAV",
    "military-vehicle": "Military Vehicle",
    "naval": "Naval Vessel",
}

CLIP_PROMPT_ENSEMBLE = {
    "aircraft": [
        "a photo of a military fighter aircraft",
        "a military jet fighter in flight",
        "a combat aircraft or military bomber",
        "a supersonic military fighter jet",
        "a military airplane on a runway or in the sky",
        "a military interceptor or stealth combat aircraft",
    ],
    "helicopter": [
        "a photo of a military helicopter",
        "an attack helicopter or military gunship",
        "a military transport helicopter in flight",
        "a military combat chopper or rotorcraft",
        "an armed attack helicopter hovering",
    ],
    "drone": [
        "a photo of a military drone",
        "an unmanned aerial vehicle or military UAV",
        "a combat drone or surveillance UAV in the sky",
        "a military unmanned reconnaissance drone",
        "an unmanned combat aerial vehicle UCAV",
    ],
    "military-vehicle": [
        "a photo of a military tank",
        "a main battle tank on the ground",
        "an armored fighting vehicle or APC",
        "an infantry fighting vehicle IFV or military armored truck",
        "a military combat vehicle in the field",
    ],
    "naval": [
        "a photo of a naval warship",
        "a military naval ship, destroyer, or frigate",
        "an aircraft carrier warship at sea",
        "a military submarine or naval combat vessel",
        "a naval vessel or warship on the ocean",
    ],
}


class AstraDataset(Dataset):
    def __init__(self, records: list[tuple[Path, int]], transform: transforms.Compose) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.records[index]
        try:
            with Image.open(path) as img:
                return self.transform(img.convert("RGB")), label
        except Exception:
            # Fallback to neutral black image if corrupt
            blank = Image.new("RGB", (224, 224), (0, 0, 0))
            return self.transform(blank), label


def load_all_records(dataset_dir: Path, external_dir: Path | None = None) -> list[tuple[Path, int]]:
    """Loads starter dataset plus any external scraped data, validating files."""
    records: list[tuple[Path, int]] = []
    labels_csv = dataset_dir / "labels.csv"

    # 1. Load starter dataset from labels.csv
    if labels_csv.exists():
        with labels_csv.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                cat = row["category"]
                if cat in CLASSES:
                    img_path = dataset_dir / row["file_name"]
                    if img_path.exists():
                        try:
                            with Image.open(img_path) as im:
                                im.verify()
                            records.append((img_path, CLASSES.index(cat)))
                        except Exception:
                            continue

    # 2. Load external scraped dataset if present
    if external_dir and external_dir.exists():
        for cat in CLASSES:
            cat_dir = external_dir / cat
            if cat_dir.is_dir():
                for file_path in cat_dir.glob("*.*"):
                    if file_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                        try:
                            with Image.open(file_path) as im:
                                im.verify()
                            records.append((file_path, CLASSES.index(cat)))
                        except Exception:
                            continue

    return records


def stratified_split(records: list[tuple[Path, int]], val_fraction: float, seed: int) -> tuple[list, list]:
    groups: dict[int, list[tuple[Path, int]]] = defaultdict(list)
    for path, label in records:
        groups[label].append((path, label))

    rng = random.Random(seed)
    train_records, val_records = [], []
    for label in range(len(CLASSES)):
        cat_records = groups[label]
        rng.shuffle(cat_records)
        cutoff = int(round(len(cat_records) * (1.0 - val_fraction)))
        train_records.extend(cat_records[:cutoff])
        val_records.extend(cat_records[cutoff:])

    return train_records, val_records


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    epochs: int,
    lr: float = 1e-3,
) -> tuple[dict, float]:
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_state = None
    best_val_acc = 0.0
    stale_epochs = 0

    print(f"\n--- Starting Model Training ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        # Quick validation check
        model.eval()
        val_correct, val_total = 0, 0
        with torch.inference_mode():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features).argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.numel()

        val_acc = val_correct / val_total if val_total else 0.0
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale_epochs = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= 8 and epoch >= 12:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    return best_state, best_val_acc


def evaluate_predictions(y_true: np.ndarray, y_probs: np.ndarray) -> dict:
    """Computes comprehensive classification metrics including F1, ROC, AUC."""
    y_pred = y_probs.argmax(axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )
    conf_mat = confusion_matrix(y_true, y_pred).tolist()

    # ROC and AUC for One-vs-Rest
    roc_data: dict[str, dict] = {}
    auc_scores: dict[str, float] = {}

    for i, cls_name in enumerate(CLASSES):
        y_bin = (y_true == i).astype(int)
        scores_cls = y_probs[:, i]
        if len(np.unique(y_bin)) > 1:
            fpr, tpr, thrs = roc_curve(y_bin, scores_cls)
            score_auc = float(roc_auc_score(y_bin, scores_cls))
            auc_scores[cls_name] = round(score_auc, 4)
            roc_data[cls_name] = {
                "fpr": [round(float(x), 4) for x in fpr],
                "tpr": [round(float(x), 4) for x in tpr],
                "auc": round(score_auc, 4),
            }
        else:
            auc_scores[cls_name] = 1.0
            roc_data[cls_name] = {"fpr": [0.0, 1.0], "tpr": [1.0, 1.0], "auc": 1.0}

    # Macro and Micro ROC-AUC
    try:
        y_onehot = np.eye(len(CLASSES))[y_true]
        macro_auc = float(roc_auc_score(y_onehot, y_probs, multi_class="ovr", average="macro"))
        micro_auc = float(roc_auc_score(y_onehot, y_probs, multi_class="ovr", average="micro"))
    except Exception:
        macro_auc = float(np.mean(list(auc_scores.values())))
        micro_auc = macro_auc

    return {
        "accuracy": round(acc * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "per_class": {
            cls_name: {
                "precision": round(report[cls_name]["precision"], 4),
                "recall": round(report[cls_name]["recall"], 4),
                "f1_score": round(report[cls_name]["f1-score"], 4),
                "support": report[cls_name]["support"],
                "roc_auc": auc_scores.get(cls_name, 1.0),
            }
            for cls_name in CLASSES
        },
        "confusion_matrix": conf_mat,
        "macro_roc_auc": round(macro_auc, 4),
        "micro_roc_auc": round(micro_auc, 4),
        "roc_curve_points": roc_data,
    }


def plot_roc_curve(roc_points: dict[str, dict], macro_auc: float, micro_auc: float, out_path: Path) -> None:
    plt.figure(figsize=(9, 7), dpi=150)
    plt.style.use("dark_background")

    colors = {
        "aircraft": "#00f0ff",
        "helicopter": "#a855f7",
        "drone": "#eab308",
        "military-vehicle": "#22c55e",
        "naval": "#3b82f6",
    }

    for cls_name in CLASSES:
        if cls_name in roc_points:
            fpr = roc_points[cls_name]["fpr"]
            tpr = roc_points[cls_name]["tpr"]
            auc_val = roc_points[cls_name]["auc"]
            plt.plot(
                fpr,
                tpr,
                color=colors.get(cls_name, "#ffffff"),
                lw=2.2,
                label=f"{DISPLAY_NAMES[cls_name]} (AUC = {auc_val:.3f})",
            )

    plt.plot([0, 1], [0, 1], color="#6b7280", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.500)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=11, color="#eff5ed", fontweight="bold")
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=11, color="#eff5ed", fontweight="bold")
    plt.title(
        f"ASTRA VISION — Multiclass ROC Curve (One-vs-Rest)\nMacro AUC: {macro_auc:.3f} | Micro AUC: {micro_auc:.3f}",
        fontsize=13,
        color="#b6f25a",
        pad=15,
        fontweight="bold",
    )
    plt.legend(loc="lower right", framealpha=0.9, facecolor="#11221e", edgecolor="#2a443c", fontsize=9.5)
    plt.grid(color="#1f3b33", linestyle=":", linewidth=0.8, alpha=0.7)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor="#08110f")
    plt.close()
    print(f"ROC Curve chart saved to: {out_path.resolve()}")


def plot_confusion_matrix(cm_data: list[list[int]], out_path: Path) -> None:
    cm = np.array(cm_data)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    plt.figure(figsize=(8, 6.5), dpi=150)
    plt.style.use("dark_background")
    plt.imshow(cm_norm, interpolation="nearest", cmap="viridis")
    plt.title("ASTRA VISION — Normalized Confusion Matrix", fontsize=13, color="#b6f25a", pad=15, fontweight="bold")
    plt.colorbar(fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(CLASSES))
    display_labels = [DISPLAY_NAMES[c] for c in CLASSES]
    plt.xticks(tick_marks, display_labels, rotation=35, ha="right", color="#eff5ed", fontsize=9)
    plt.yticks(tick_marks, display_labels, color="#eff5ed", fontsize=9)

    thresh = cm_norm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            plt.text(
                j,
                i,
                f"{val}\n({pct:.1f}%)",
                horizontalalignment="center",
                verticalalignment="center",
                color="black" if cm_norm[i, j] > thresh else "white",
                fontsize=8.5,
                fontweight="bold",
            )

    plt.ylabel("True Defence Class", fontsize=11, color="#eff5ed", fontweight="bold")
    plt.xlabel("Predicted Defence Class", fontsize=11, color="#eff5ed", fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, facecolor="#08110f")
    plt.close()
    print(f"Confusion Matrix chart saved to: {out_path.resolve()}")


def evaluate_clip(val_records: list[tuple[Path, int]], device: str) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates zero-shot CLIP ViT-B/32 using prompt ensembling."""
    from transformers import CLIPModel, CLIPProcessor

    print("\n--- Evaluating Zero-Shot CLIP ViT-B/32 ---")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()

    # Precompute ensemble text prototypes
    with torch.inference_mode():
        class_text_embeddings = []
        for cls_name in CLASSES:
            prompts = CLIP_PROMPT_ENSEMBLE[cls_name]
            txt_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            feats = clip_model.get_text_features(**txt_inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            proto = feats.mean(dim=0)
            proto = proto / proto.norm(dim=-1, keepdim=True)
            class_text_embeddings.append(proto)
        text_matrix = torch.stack(class_text_embeddings, dim=1)  # (D, 5)

    y_true_list, y_probs_list = [], []
    with torch.inference_mode():
        for path, label in val_records:
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                    img_inputs = processor(images=img_rgb, return_tensors="pt").to(device)
                    img_feats = clip_model.get_image_features(**img_inputs)
                    img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
                    logits = (img_feats @ text_matrix) * 100.0
                    probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                    y_true_list.append(label)
                    y_probs_list.append(probs)
            except Exception:
                continue

    return np.array(y_true_list), np.array(y_probs_list)


def main() -> None:
    parser = argparse.ArgumentParser(description="ASTRA VISION: Full Training, Evaluation, and ROC Pipeline")
    parser.add_argument("--dataset", type=Path, default=Path("../ASTRA-Challenge-Starter/challenge-02-vision"))
    parser.add_argument("--external-dataset", type=Path, default=Path("data/external/wikimedia"))
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution device: {device}")

    # 1. Collect records
    all_records = load_all_records(args.dataset, args.external_dataset)
    print(f"Total curated images loaded: {len(all_records)}")
    counts = defaultdict(int)
    for _, l in all_records:
        counts[CLASSES[l]] += 1
    for k, v in counts.items():
        print(f"  - {DISPLAY_NAMES[k]}: {v} images")

    # 2. Stratified train/validation split
    train_records, val_records = stratified_split(all_records, args.val_fraction, args.seed)
    print(f"Train split: {len(train_records)} images | Validation split: {len(val_records)} images")

    # 3. Data augmentations & DataLoaders
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_loader = DataLoader(
        AstraDataset(train_records, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        AstraDataset(val_records, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # 4. Architecture: MobileNetV3-Small (Transfer Learning)
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    mobilenet = models.mobilenet_v3_small(weights=weights)

    # Freeze feature extractor initially
    for p in mobilenet.features.parameters():
        p.requires_grad = False
    # Unfreeze top feature layers for domain adaptation
    for p in mobilenet.features[-3:].parameters():
        p.requires_grad = True

    in_feat = mobilenet.classifier[3].in_features
    mobilenet.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, len(CLASSES)),
    )
    mobilenet.to(device)

    # 5. Train
    best_weights, _ = train_model(
        mobilenet,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
    )
    mobilenet.load_state_dict(best_weights)

    # 6. Evaluate MobileNetV3 on validation split
    mobilenet.eval()
    val_y_true = []
    val_probs_mobilenet = []
    with torch.inference_mode():
        for feats, labels in val_loader:
            feats = feats.to(device)
            logits = mobilenet(feats)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            val_probs_mobilenet.extend(probs)
            val_y_true.extend(labels.numpy())

    y_true_arr = np.array(val_y_true)
    probs_mobilenet_arr = np.array(val_probs_mobilenet)

    mobilenet_metrics = evaluate_predictions(y_true_arr, probs_mobilenet_arr)

    # 7. Evaluate CLIP Zero-Shot on the same validation split
    clip_y_true, clip_probs = evaluate_clip(val_records, device)
    clip_metrics = evaluate_predictions(clip_y_true, clip_probs)

    # 8. Evaluate Hybrid Ensemble (50% MobileNet + 50% CLIP)
    ensemble_probs = 0.5 * probs_mobilenet_arr + 0.5 * clip_probs
    ensemble_metrics = evaluate_predictions(y_true_arr, ensemble_probs)

    # 9. Output directory and artifacts
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_weights, output_dir / "mobilenet_v3_small.pt")

    # Plot ROC curve for the Ensemble & MobileNet
    plot_roc_curve(
        ensemble_metrics["roc_curve_points"],
        macro_auc=ensemble_metrics["macro_roc_auc"],
        micro_auc=ensemble_metrics["micro_roc_auc"],
        out_path=output_dir / "roc_curve.png",
    )

    # Plot Confusion Matrix
    plot_confusion_matrix(ensemble_metrics["confusion_matrix"], output_dir / "confusion_matrix.png")

    # 10. Save comprehensive metrics payload
    full_report = {
        "dataset": {
            "total_images": len(all_records),
            "train_images": len(train_records),
            "val_images": len(val_records),
            "classes": CLASSES,
            "display_names": DISPLAY_NAMES,
        },
        "ensemble_model": {
            "name": "Ensemble (MobileNetV3-Small Fine-Tuned + CLIP ViT-B/32 Zero-Shot)",
            "accuracy": ensemble_metrics["accuracy"],
            "macro_f1": ensemble_metrics["macro_f1"],
            "weighted_f1": ensemble_metrics["weighted_f1"],
            "macro_precision": ensemble_metrics["macro_precision"],
            "macro_recall": ensemble_metrics["macro_recall"],
            "macro_roc_auc": ensemble_metrics["macro_roc_auc"],
            "micro_roc_auc": ensemble_metrics["micro_roc_auc"],
            "per_class": ensemble_metrics["per_class"],
        },
        "fine_tuned_mobilenet": {
            "name": "MobileNetV3-Small (Fine-Tuned)",
            "accuracy": mobilenet_metrics["accuracy"],
            "macro_f1": mobilenet_metrics["macro_f1"],
            "weighted_f1": mobilenet_metrics["weighted_f1"],
            "macro_roc_auc": mobilenet_metrics["macro_roc_auc"],
            "micro_roc_auc": mobilenet_metrics["micro_roc_auc"],
            "per_class": mobilenet_metrics["per_class"],
        },
        "clip_zero_shot": {
            "name": "OpenAI CLIP ViT-B/32 (Prompt-Ensembled Zero-Shot)",
            "accuracy": clip_metrics["accuracy"],
            "macro_f1": clip_metrics["macro_f1"],
            "weighted_f1": clip_metrics["weighted_f1"],
            "macro_roc_auc": clip_metrics["macro_roc_auc"],
            "micro_roc_auc": clip_metrics["micro_roc_auc"],
            "per_class": clip_metrics["per_class"],
        },
    }

    metrics_file = output_dir / "metrics.json"
    metrics_file.write_text(json.dumps(full_report, indent=2), encoding="utf-8")
    print(f"\nMetrics written to: {metrics_file.resolve()}")

    # Print summary table
    print("\n" + "=" * 65)
    print("                    ASTRA VISION EVALUATION SUMMARY")
    print("=" * 65)
    print(f"{'Model':<35} | {'Acc (%)':<8} | {'Macro F1':<8} | {'ROC-AUC':<8}")
    print("-" * 65)
    print(f"{'MobileNetV3-Small (Fine-Tuned)':<35} | {mobilenet_metrics['accuracy']:<8.2f} | {mobilenet_metrics['macro_f1']:<8.4f} | {mobilenet_metrics['macro_roc_auc']:<8.4f}")
    print(f"{'CLIP ViT-B/32 (Zero-Shot)':<35} | {clip_metrics['accuracy']:<8.2f} | {clip_metrics['macro_f1']:<8.4f} | {clip_metrics['macro_roc_auc']:<8.4f}")
    print(f"{'Hybrid Ensemble (Ours)':<35} | {ensemble_metrics['accuracy']:<8.2f} | {ensemble_metrics['macro_f1']:<8.4f} | {ensemble_metrics['macro_roc_auc']:<8.4f}")
    print("=" * 65)
    print("\nPer-Class Breakdown (Hybrid Ensemble):")
    for cls_name in CLASSES:
        info = ensemble_metrics["per_class"][cls_name]
        print(f"  {DISPLAY_NAMES[cls_name]:<18}: Prec={info['precision']:.3f} | Rec={info['recall']:.3f} | F1={info['f1_score']:.3f} | AUC={info['roc_auc']:.3f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
