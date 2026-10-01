"""ASTRA VISION — Online Continuous Learning & Severe Punishment Engine (Version 5.0).
Provides real-time local model verification, penalty punishment, and continuous fine-tuning:
1. Local Model Verification: Compares local predictions with verified ground truth from AI check.
2. Severe Punishment: When local model is wrong, applies high loss penalty, deducts penalty points, and logs failure.
3. Neural & Exemplar Retraining: Immediately updates neural weights via gradient descent and stores normalized visual vectors into online memory so the model answers correctly when encountering similar images next time.
4. Persistent Memory: Retains learned samples and training logs across restarts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
import defense_knowledge as dk

CLASSES = ["aircraft", "helicopter", "drone", "military-vehicle", "naval"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class OnlineTrainer:
    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device = device
        self.lock = Lock()

        self.memory_file = model_dir / "online_memory.pt"
        self.log_file = model_dir / "training_log.json"
        self.state_file = model_dir / "online_trainer_state.json"

        self.memory_vectors: torch.Tensor | None = None  # Shape (M, 512)
        self.memory_records: list[dict[str, Any]] = []

        self.stats = {
            "health_score": 1000,
            "total_evaluations": 0,
            "total_correct": 0,
            "total_wrong": 0,
            "total_punishments": 0,
            "penalty_points_applied": 0,
            "reward_points_applied": 0,
            "learned_exemplars": 0,
            "recent_events": [],
        }

        self._load_state()
        self._load_memory()

    def _load_state(self) -> None:
        if self.state_file.exists():
            try:
                data = json.loads(self.state_file.read_text(encoding="utf-8"))
                self.stats.update(data)
            except Exception as e:
                print(f"[OnlineTrainer] Notice: unable to read state file: {e}")

    def _save_state(self) -> None:
        try:
            self.state_file.write_text(json.dumps(self.stats, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[OnlineTrainer] Notice: unable to write state file: {e}")

    def _load_memory(self) -> None:
        if self.memory_file.exists():
            try:
                data = torch.load(self.memory_file, map_location=self.device, weights_only=False)
                self.memory_vectors = data.get("vectors")
                self.memory_records = data.get("records", [])
                if self.memory_vectors is not None:
                    self.memory_vectors = self.memory_vectors.to(self.device)
                self.stats["learned_exemplars"] = len(self.memory_records)
                print(f"[OnlineTrainer] Loaded {len(self.memory_records)} learned memory exemplars.")
            except Exception as e:
                print(f"[OnlineTrainer] Notice: unable to load memory file: {e}")

    def _save_memory(self) -> None:
        try:
            if self.memory_vectors is not None:
                torch.save(
                    {"vectors": self.memory_vectors.cpu(), "records": self.memory_records},
                    self.memory_file,
                )
        except Exception as e:
            print(f"[OnlineTrainer] Notice: unable to save memory file: {e}")

    def query_memory(
        self,
        img_feats: torch.Tensor,
        threshold: float = 0.82,
    ) -> dict[str, Any] | None:
        """Checks if current image matches any previously trained/punished exemplar."""
        with self.lock:
            if self.memory_vectors is None or len(self.memory_records) == 0:
                return None

            # Compute cosine similarities (img_feats is 1 x 512, memory is M x 512)
            sims = (img_feats @ self.memory_vectors.T)[0]  # Shape (M,)
            max_sim, best_idx = torch.max(sims, dim=0)
            max_sim_val = float(max_sim.item())

            if max_sim_val >= threshold:
                matched_record = dict(self.memory_records[best_idx.item()])
                matched_record["similarity"] = round(max_sim_val * 100, 1)
                matched_record["from_learned_memory"] = True
                return matched_record

        return None

    def compare_answers(
        self,
        local_pred: dict[str, Any],
        verified_truth: dict[str, Any],
    ) -> tuple[bool, str]:
        """Compares local model's prediction with the verified ground truth from AI check.
        Returns: (is_correct, reason)
        """
        local_name = (local_pred.get("platform", {}).get("name") or local_pred.get("label") or "").strip().lower()
        local_cat = (local_pred.get("prediction") or "").strip().lower()
        local_desig = (local_pred.get("platform", {}).get("designation") or "").strip().lower()

        true_name = (verified_truth.get("exact_name") or verified_truth.get("name") or "").strip().lower()
        true_cat = (verified_truth.get("category") or "").strip().lower()
        true_desig = (verified_truth.get("designation") or "").strip().lower()

        # Check Category Consistency
        cat_matches = (local_cat == true_cat) or (
            local_cat in ["civilian-vehicle", "military-vehicle"] and true_cat in ["civilian-vehicle", "military-vehicle"]
        )

        if not cat_matches:
            return False, f"Category mismatch: Local guessed '{local_cat}' but object is '{true_cat}'"

        # Check Name & Platform Similarity
        if local_name and true_name:
            if local_name == true_name:
                return True, "Exact name match"

            # Check designation match
            if local_desig and true_desig and (local_desig in true_name or true_desig in local_name):
                return True, "Designation match"

            # Token overlap check (ignoring generic stop words)
            stop_words = {"the", "a", "an", "and", "of", "in", "for", "with", "system", "mark", "block", "air", "ground"}
            local_tokens = set(local_name.replace("-", " ").replace("/", " ").split()) - stop_words
            true_tokens = set(true_name.replace("-", " ").replace("/", " ").split()) - stop_words

            overlap = local_tokens.intersection(true_tokens)
            if len(overlap) >= 2 or (len(overlap) == 1 and any(len(t) >= 4 for t in overlap)):
                return True, f"Visual entity verified ({', '.join(overlap)})"

            # Substring check
            if local_name in true_name or true_name in local_name:
                return True, "Name substring match"

        # If categories match and local model had high confidence on general class
        if cat_matches and not local_name:
            return True, "Category verified"

        return False, f"Platform mismatch: Local guessed '{local_name or local_cat}' but verified as '{true_name}'"

    def evaluate_and_train(
        self,
        image: Image.Image,
        img_feats: torch.Tensor,
        local_pred: dict[str, Any],
        verified_truth: dict[str, Any],
        classifier_model: nn.Module | None,
        preprocess_fn: Any,
        filename: str = "upload",
    ) -> dict[str, Any]:
        """Evaluates prediction, punishes if wrong, updates neural weights, and memorizes exemplar."""
        is_correct, reason = self.compare_answers(local_pred, verified_truth)

        with self.lock:
            self.stats["total_evaluations"] += 1
            t_now = time.strftime("%Y-%m-%d %H:%M:%S")

            loss_before_val: float | None = None
            loss_after_val: float | None = None
            penalty_applied = 0
            reward_applied = 0

            target_cat = verified_truth.get("category", "aircraft")
            target_idx = CLASS_TO_IDX.get(target_cat)

            if not is_correct:
                # SEVERE PUNISHMENT
                penalty_applied = -50
                self.stats["total_wrong"] += 1
                self.stats["total_punishments"] += 1
                self.stats["penalty_points_applied"] += 50
                self.stats["health_score"] = max(100, self.stats["health_score"] - 50)

                # Train / Fine-tune neural classifier head via gradient descent
                if classifier_model is not None and target_idx is not None and preprocess_fn is not None:
                    try:
                        classifier_model.train()
                        tensor = preprocess_fn(image.convert("RGB")).unsqueeze(0).to(self.device)
                        target_tensor = torch.tensor([target_idx], dtype=torch.long, device=self.device)

                        # Check loss before
                        logits_before = classifier_model(tensor)
                        l_before = F.cross_entropy(logits_before, target_tensor)
                        loss_before_val = float(l_before.item())

                        # Apply severe punishment multiplier to gradient update
                        punishment_loss = l_before * 4.0

                        # Quick gradient step on classifier head parameters
                        params = [p for p in classifier_model.classifier[3].parameters() if p.requires_grad]
                        if params:
                            optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
                            optimizer.zero_grad()
                            punishment_loss.backward()
                            optimizer.step()

                        # Evaluate loss after
                        with torch.no_grad():
                            logits_after = classifier_model(tensor)
                            l_after = F.cross_entropy(logits_after, target_tensor)
                            loss_after_val = float(l_after.item())

                        classifier_model.eval()

                        # Save updated weights back to disk
                        torch.save(classifier_model.state_dict(), self.model_dir / "mobilenet_v3_small.pt")
                    except Exception as e:
                        print(f"[OnlineTrainer] Neural training step note: {e}")

                # Store into visual memory store (so model recognizes it next time!)
                try:
                    norm_feat = (img_feats / img_feats.norm(dim=-1, keepdim=True)).detach().to(self.device)
                    if self.memory_vectors is None or self.memory_vectors.shape[0] == 0:
                        self.memory_vectors = norm_feat
                    else:
                        self.memory_vectors = torch.cat([self.memory_vectors, norm_feat], dim=0)

                    t_year, t_built, t_op = dk.resolve_historical_metrics(
                        name=verified_truth.get("exact_name") or verified_truth.get("name", "Unknown"),
                        designation=verified_truth.get("designation", "TARGET"),
                        year_origin=verified_truth.get("year_origin"),
                        number_built=verified_truth.get("number_built"),
                        operational_count=verified_truth.get("operational_count"),
                        category=target_cat,
                        context_text=f"{verified_truth.get('model_variant', '')} {verified_truth.get('tactical_analysis', '')}",
                    )

                    record = {
                        "name": verified_truth.get("exact_name") or verified_truth.get("name", "Unknown"),
                        "designation": verified_truth.get("designation", "TARGET"),
                        "category": target_cat,
                        "category_label": verified_truth.get("category_label", target_cat.title()),
                        "country": verified_truth.get("country", "Global"),
                        "manufacturer": verified_truth.get("manufacturer", "Manufacturer"),
                        "year_origin": t_year,
                        "number_built": t_built,
                        "operational_count": t_op,
                        "status": verified_truth.get("status", "Active"),
                        "role": verified_truth.get("role", "Operations"),
                        "specs": verified_truth.get("specs", {}),
                        "visual_signatures": verified_truth.get("visual_signatures", "Visual signature learned through online self-training."),
                        "tactical_analysis": verified_truth.get("tactical_analysis", "Learned exemplar."),
                        "confidence": verified_truth.get("confidence", 99.0),
                        "learned_at": t_now,
                        "penalty_applied": -50,
                    }
                    self.memory_records.append(record)
                    self.stats["learned_exemplars"] = len(self.memory_records)
                    self._save_memory()
                except Exception as e:
                    print(f"[OnlineTrainer] Memory exemplar note: {e}")

            else:
                # REWARD FOR CORRECT PREDICTION
                reward_applied = 10
                self.stats["total_correct"] += 1
                self.stats["reward_points_applied"] += 10
                self.stats["health_score"] = min(2000, self.stats["health_score"] + 10)

                # Reinforce memory with positive sample
                try:
                    norm_feat = (img_feats / img_feats.norm(dim=-1, keepdim=True)).detach().to(self.device)
                    if self.memory_vectors is None or self.memory_vectors.shape[0] == 0:
                        self.memory_vectors = norm_feat
                    else:
                        # Only add if not already extremely close
                        sims = (norm_feat @ self.memory_vectors.T)[0]
                        if torch.max(sims).item() < 0.95:
                            self.memory_vectors = torch.cat([self.memory_vectors, norm_feat], dim=0)
                            t_year, t_built, t_op = dk.resolve_historical_metrics(
                                name=verified_truth.get("exact_name") or verified_truth.get("name", "Unknown"),
                                designation=verified_truth.get("designation", "TARGET"),
                                year_origin=verified_truth.get("year_origin"),
                                number_built=verified_truth.get("number_built"),
                                operational_count=verified_truth.get("operational_count"),
                                category=target_cat,
                                context_text=f"{verified_truth.get('model_variant', '')} {verified_truth.get('tactical_analysis', '')}",
                            )
                            record = {
                                "name": verified_truth.get("exact_name") or verified_truth.get("name", "Unknown"),
                                "designation": verified_truth.get("designation", "TARGET"),
                                "category": target_cat,
                                "category_label": verified_truth.get("category_label", target_cat.title()),
                                "country": verified_truth.get("country", "Global"),
                                "manufacturer": verified_truth.get("manufacturer", "Manufacturer"),
                                "year_origin": t_year,
                                "number_built": t_built,
                                "operational_count": t_op,
                                "specs": verified_truth.get("specs", {}),
                                "learned_at": t_now,
                                "status": "Reinforced",
                            }
                            self.memory_records.append(record)
                            self._save_memory()
                except Exception as e:
                    print(f"[OnlineTrainer] Reinforcement memory note: {e}")

            # Record event
            event = {
                "time": t_now,
                "filename": filename,
                "verdict": "CORRECT" if is_correct else "WRONG_PUNISHED",
                "local_guessed": local_pred.get("platform", {}).get("name") or local_pred.get("label"),
                "verified_answer": verified_truth.get("exact_name") or verified_truth.get("name"),
                "penalty": penalty_applied,
                "reward": reward_applied,
                "reason": reason,
                "loss_before": round(loss_before_val, 3) if loss_before_val is not None else None,
                "loss_after": round(loss_after_val, 3) if loss_after_val is not None else None,
            }
            self.stats["recent_events"].insert(0, event)
            self.stats["recent_events"] = self.stats["recent_events"][:30]
            self._save_state()

        return {
            "status": "correct" if is_correct else "wrong_punished",
            "is_correct": is_correct,
            "reason": reason,
            "penalty_points": penalty_applied,
            "reward_points": reward_applied,
            "loss_before": round(loss_before_val, 3) if loss_before_val is not None else None,
            "loss_after": round(loss_after_val, 3) if loss_after_val is not None else None,
            "learned_samples": len(self.memory_records),
            "health_score": self.stats["health_score"],
            "local_answer": {
                "name": local_pred.get("platform", {}).get("name") or local_pred.get("label"),
                "category": local_pred.get("prediction"),
                "confidence": local_pred.get("confidence"),
            },
            "verified_answer": {
                "name": verified_truth.get("exact_name") or verified_truth.get("name"),
                "category": verified_truth.get("category"),
                "confidence": verified_truth.get("confidence", 99.0),
            },
        }

    def status(self) -> dict[str, Any]:
        with self.lock:
            total = self.stats["total_evaluations"]
            acc = round((self.stats["total_correct"] / total * 100), 1) if total > 0 else 100.0
            return {
                **self.stats,
                "accuracy_rate": acc,
                "active_memory_slots": len(self.memory_records),
            }
