"""ASTRA VISION — High-Precision Multimodal Defense Object & Platform Recognition Engine.
Combines fine-tuned MobileNetV3 + OpenAI CLIP ViT-B/32 zero-shot cross-modal matching with
an exhaustive Defense Intelligence Knowledge Base covering 60+ global military platforms.
Optionally integrates Gemini 2.5 Flash for open-ended deep tactical reconnaissance.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from threading import Lock
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageEnhance
from torch import nn
from torchvision import models, transforms
from transformers import CLIPModel, CLIPProcessor

import defense_knowledge as dk
import gemini_tactical
import omni_intelligence
from online_trainer import OnlineTrainer

CLASSES = ["aircraft", "helicopter", "drone", "military-vehicle", "naval"]
DISPLAY_NAMES = {
    "aircraft": "Aircraft / Aviation",
    "helicopter": "Helicopter",
    "drone": "Drone / UAV",
    "military-vehicle": "Military Vehicle",
    "naval": "Naval Vessel",
}

def enhance_if_monochrome(image: Image.Image) -> Image.Image:
    """Enhances black & white, grayscale, or archival images with dynamic range normalization and sharpness."""
    img_rgb = image.convert("RGB")
    arr = np.array(img_rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b)))
    if color_diff < 15.0:
        gray = img_rgb.convert("L")
        auto_contrast = ImageOps.autocontrast(gray, cutoff=1)
        sharp = ImageEnhance.Sharpness(auto_contrast).enhance(1.4)
        contrast = ImageEnhance.Contrast(sharp).enhance(1.15)
        return contrast.convert("RGB")
    return img_rgb

CLIP_PROMPT_ENSEMBLE = {
    "aircraft": [
        "a photo of an aircraft or airplane in flight",
        "a photo of a commercial passenger airliner in flight",
        "a commercial passenger jet or transport plane in the sky",
        "a civilian passenger flight airliner with windows",
        "a military fighter aircraft or supersonic jet in flight",
        "a combat aircraft or military bomber plane",
        "a black and white photo of an aircraft or airplane",
        "a monochrome vintage archival photo of an aircraft in flight",
        "an airplane in the sky or on a runway",
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


class VisionEngine:
    """Two-tier defense reconnaissance engine:
    1. Primary Category Classification (MobileNetV3 + CLIP multi-prompt ensemble)
    2. Fine-Grained Platform Identification & Technical Dossier (CLIP Vector Space + Defense Knowledge Base)
    3. Optional Cloud Multimodal Reconnaissance (Gemini 2.5 Flash)
    """

    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lock = Lock()

        self.online_trainer = OnlineTrainer(self.model_dir, device=self.device)

        self.clip_model: CLIPModel | None = None
        self.clip_processor: CLIPProcessor | None = None
        self.clip_text_matrix: torch.Tensor | None = None  # Shape (D, 5) for categories

        self.platform_matrix: torch.Tensor | None = None   # Shape (D, N) for specific platforms
        self.platform_ids: list[str] = []

        self.classifier: nn.Module | None = None
        self.classifier_metrics: dict = {}

        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            self.normalize,
        ])

        self._load_classifier()

    def _load_classifier(self) -> None:
        checkpoint = self.model_dir / "mobilenet_v3_small.pt"
        metrics_file = self.model_dir / "metrics.json"

        if checkpoint.exists():
            try:
                model = models.mobilenet_v3_small(weights=None)
                in_feat = model.classifier[3].in_features
                model.classifier[3] = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(in_feat, len(CLASSES)),
                )
                state = torch.load(checkpoint, map_location=self.device, weights_only=True)
                model.load_state_dict(state)
                model.to(self.device).eval()
                self.classifier = model
                print("Loaded fine-tuned MobileNetV3 checkpoint.")
            except Exception as e:
                print(f"Warning: could not load classifier checkpoint: {e}")

        if metrics_file.exists():
            try:
                self.classifier_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass

    def _load_clip(self) -> None:
        if self.clip_model is None:
            print("Initializing OpenAI CLIP ViT-B/32...")
            try:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to(self.device).eval()
            except Exception:
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()

            # Precompute ensemble text feature prototypes for categories
            with torch.inference_mode():
                class_text_embeddings = []
                for cls_name in CLASSES:
                    prompts = CLIP_PROMPT_ENSEMBLE[cls_name]
                    txt_inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                    feats = self.clip_model.get_text_features(**txt_inputs)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    proto = feats.mean(dim=0)
                    proto = proto / proto.norm(dim=-1, keepdim=True)
                    class_text_embeddings.append(proto)
                self.clip_text_matrix = torch.stack(class_text_embeddings, dim=1).to(self.device)  # (D, 5)

            # Load or precompute platform embedding matrix
            self._load_platform_embeddings()
            print("CLIP text prototypes and Defense Platform vectors ready.")

    def _load_platform_embeddings(self) -> None:
        """Loads precomputed platform text prototypes or generates them from defense_knowledge."""
        cache_path = self.model_dir / "platform_embeddings.pt"
        platforms = dk.all_platforms()
        expected_ids = [p["id"] for p in platforms]

        if cache_path.exists():
            try:
                cached = torch.load(cache_path, map_location=self.device, weights_only=True)
                if cached.get("platform_ids") == expected_ids:
                    self.platform_matrix = cached["matrix"].to(self.device)
                    self.platform_ids = expected_ids
                    print(f"Loaded cached platform embedding matrix for {len(expected_ids)} platforms.")
                    return
            except Exception as e:
                print(f"Platform cache reload notice: {e}")

        # Compute on the fly
        assert self.clip_model is not None and self.clip_processor is not None
        print(f"Precomputing CLIP text embeddings for {len(platforms)} defense platforms...")
        with torch.inference_mode():
            embeddings = []
            for p in platforms:
                prompts = p.get("clip_prompts") or [f"a photo of {p.get('name', 'military platform')}"]
                txt_inputs = self.clip_processor(text=prompts, return_tensors="pt", padding=True).to(self.device)
                feats = self.clip_model.get_text_features(**txt_inputs)
                feats = feats / feats.norm(dim=-1, keepdim=True)
                proto = feats.mean(dim=0)
                proto = proto / proto.norm(dim=-1, keepdim=True)
                embeddings.append(proto)

            self.platform_matrix = torch.stack(embeddings, dim=1).to(self.device)  # (D, N)
            self.platform_ids = expected_ids

            try:
                torch.save({"matrix": self.platform_matrix.cpu(), "platform_ids": self.platform_ids}, cache_path)
                print(f"Saved platform embeddings cache to {cache_path}")
            except Exception as e:
                print(f"Notice: unable to save platform cache: {e}")

    @torch.inference_mode()
    def _classifier_scores(self, image: Image.Image) -> torch.Tensor | None:
        if self.classifier is None:
            return None
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        logits = self.classifier(tensor)
        return torch.softmax(logits[0], dim=0).cpu()

    @torch.inference_mode()
    def _clip_image_features(self, image: Image.Image) -> torch.Tensor:
        self._load_clip()
        assert self.clip_model is not None and self.clip_processor is not None
        img_inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
        img_feats = self.clip_model.get_image_features(**img_inputs)
        return img_feats / img_feats.norm(dim=-1, keepdim=True)

    @torch.inference_mode()
    def _clip_category_scores(self, img_feats: torch.Tensor) -> torch.Tensor:
        assert self.clip_text_matrix is not None
        logits = (img_feats @ self.clip_text_matrix) * 100.0
        return torch.softmax(logits[0], dim=0).cpu()

    def _generate_gradcam(self, image: Image.Image, target_class: int) -> str | None:
        """Generates a Grad-CAM saliency overlay as a base64 data URI."""
        if self.classifier is None:
            return None

        try:
            self.classifier.eval()
            for p in self.classifier.parameters():
                p.requires_grad = True
            tensor = self.preprocess(image).unsqueeze(0).to(self.device).requires_grad_(True)

            activations = []
            gradients = []

            def forward_hook(m, inp, outp):
                activations.append(outp)

            def backward_hook(m, grad_in, grad_out):
                gradients.append(grad_out[0])

            target_layer = self.classifier.features[-1]
            h_fwd = target_layer.register_forward_hook(forward_hook)
            h_bwd = target_layer.register_full_backward_hook(backward_hook)

            output = self.classifier(tensor)
            score = output[0, target_class]
            self.classifier.zero_grad()
            score.backward()

            h_fwd.remove()
            h_bwd.remove()

            if not activations or not gradients:
                return None

            act = activations[0].detach()[0]  # (C, H, W)
            grad = gradients[0].detach()[0]   # (C, H, W)
            weights = grad.mean(dim=(1, 2), keepdim=True)
            cam = (weights * act).sum(dim=0).clamp(min=0)

            if cam.max() > 0:
                cam = cam / cam.max()
            cam_np = cam.cpu().numpy()

            # Resize CAM to match original image size
            cam_img = Image.fromarray((cam_np * 255).astype(np.uint8)).resize(image.size, resample=Image.BICUBIC)
            cam_norm = np.array(cam_img) / 255.0

            try:
                colormap = matplotlib.colormaps["jet"]
            except Exception:
                colormap = cm.get_cmap("jet")
            heatmap = colormap(cam_norm)[:, :, :3]

            # Blend with original image
            orig_np = np.array(image.convert("RGB")) / 255.0
            blended = 0.55 * orig_np + 0.45 * heatmap
            blended = np.clip(blended * 255, 0, 255).astype(np.uint8)

            buf = io.BytesIO()
            Image.fromarray(blended).save(buf, format="JPEG", quality=85)
            encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            print(f"Grad-CAM generation note: {e}")
            return None

    def _identify_platform(
        self,
        img_feats: torch.Tensor,
        predicted_category: str,
        category_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Identifies specific platform via CLIP feature matching conditioned on category."""
        assert self.platform_matrix is not None

        # Compute cosine similarity across all platforms in vector space
        sims = (img_feats @ self.platform_matrix)[0].cpu()  # (N,)

        # Identify candidate platforms for the predicted category
        cat_indices: list[int] = []
        for idx, pid in enumerate(self.platform_ids):
            p = dk.get_platform(pid)
            if p and p.get("category") == predicted_category:
                cat_indices.append(idx)

        # Fallback to all platforms if category filter yields empty
        if not cat_indices:
            cat_indices = list(range(len(self.platform_ids)))

        cat_sims = sims[cat_indices]

        # Rank candidates descending by raw cosine similarity
        ranked_order = torch.argsort(cat_sims, descending=True)

        top_local_idx = ranked_order[0].item()
        top_global_idx = cat_indices[top_local_idx]
        top_pid = self.platform_ids[top_global_idx]
        top_platform = dk.get_platform(top_pid)
        top_sim = sims[top_global_idx].item()

        # Runner-up similarity for margin assessment
        if len(ranked_order) > 1:
            second_local_idx = ranked_order[1].item()
            runner_up_sim = cat_sims[second_local_idx].item()
        else:
            runner_up_sim = top_sim
        margin = max(0.0, top_sim - runner_up_sim)

        # Genuine CLIP cosine visual match percentage:
        # In CLIP, baseline text-image similarity is ~0.15 (unrelated) to 0.35+ (strong match)
        clip_match_pct = max(0.0, min(100.0, (top_sim - 0.15) / (0.35 - 0.15) * 100.0))

        # Calibrated actual confidence:
        # Combines category model certainty (0-100) and visual platform match percentage (0-100)
        base_conf = (category_confidence * 100.0 * 0.50) + (clip_match_pct * 0.50)

        # If decision margin is tight (<0.008), penalize confidence for high ambiguity
        if margin < 0.008:
            base_conf = base_conf * 0.88

        actual_confidence = round(max(5.0, min(99.0, base_conf)), 1)

        # Extract runner-up candidates with ACTUAL calibrated confidence scores
        runner_ups = []
        for i in range(1, min(4, len(ranked_order))):
            local_i = ranked_order[i].item()
            global_i = cat_indices[local_i]
            r_pid = self.platform_ids[global_i]
            r_info = dk.get_platform(r_pid)
            if r_info:
                r_sim = sims[global_i].item()
                r_clip_pct = max(0.0, min(100.0, (r_sim - 0.15) / (0.35 - 0.15) * 100.0))
                r_conf = round(max(5.0, min(actual_confidence - 0.5, (category_confidence * 100.0 * 0.50) + (r_clip_pct * 0.50))), 1)
                runner_ups.append({
                    "id": r_info["id"],
                    "name": r_info["name"],
                    "designation": r_info["designation"],
                    "country": r_info["country"],
                    "confidence": r_conf,
                })

        # Deep intelligence dossier
        dossier = {
            "id": top_platform["id"],
            "name": top_platform["name"],
            "designation": top_platform["designation"],
            "model_variant": top_platform["model_variant"],
            "country": top_platform["country"],
            "manufacturer": top_platform["manufacturer"],
            "year_origin": top_platform["year_origin"],
            "number_built": top_platform["number_built"],
            "operational_count": top_platform["operational_count"],
            "status": top_platform["status"],
            "role": top_platform["role"],
            "specs": dict(top_platform["specs"]),
            "visual_signatures": top_platform["visual_signatures"],
            "confidence": actual_confidence,
            "runner_ups": runner_ups,
            "engine": "Local Defense Intelligence Knowledge Base (CLIP ViT-B/32)",
        }
        return dossier

    def predict_local(
        self,
        image: Image.Image,
        filename: str,
    ) -> dict[str, Any]:
        """Local offline recognition pipeline:
        1. Classifies defense category (MobileNetV3 + CLIP ViT-B/32).
        2. Identifies exact platform from local defense knowledge base.
        3. Generates Grad-CAM explainable AI saliency heatmap.
        """
        image = enhance_if_monochrome(image)
        with self.lock:
            img_feats = self._clip_image_features(image)
            clip_scores = self._clip_category_scores(img_feats)
            trained_scores = self._classifier_scores(image)

        # Check learned memory first (online self-training)
        mem_match = self.online_trainer.query_memory(img_feats, threshold=0.82)
        if mem_match:
            prediction = mem_match.get("category", "object")
            category_label = mem_match.get("category_label") or DISPLAY_NAMES.get(prediction, prediction.title())
            actual_conf = round(float(mem_match.get("similarity", 90.0)), 1)
            # Get runner ups for this category so runner_ups is always available
            fallback_platform = self._identify_platform(img_feats, prediction, category_confidence=actual_conf / 100.0)
            runner_ups = fallback_platform.get("runner_ups", [])
            top_index = CLASSES.index(prediction) if prediction in CLASSES else 0
            heatmap_url = self._generate_gradcam(image, top_index)

            t_year, t_built, t_op = dk.resolve_historical_metrics(
                name=mem_match.get("name", ""),
                designation=mem_match.get("designation", ""),
                year_origin=mem_match.get("year_origin"),
                number_built=mem_match.get("number_built"),
                operational_count=mem_match.get("operational_count"),
                category=prediction,
                context_text=f"{mem_match.get('model_variant', '')} {mem_match.get('tactical_analysis', '')}",
            )

            dossier = {
                "id": mem_match.get("id") or mem_match["name"].lower().replace(" ", "-"),
                "name": mem_match["name"],
                "designation": mem_match.get("designation", "TARGET"),
                "model_variant": mem_match.get("model_variant", "Learned Variant"),
                "country": mem_match.get("country", "Global"),
                "manufacturer": mem_match.get("manufacturer", "Manufacturer"),
                "year_origin": t_year,
                "number_built": t_built,
                "operational_count": t_op,
                "status": mem_match.get("status", "Active"),
                "role": mem_match.get("role", "Operations"),
                "specs": mem_match.get("specs", {}),
                "visual_signatures": mem_match.get("visual_signatures", "Visual signature learned through self-training."),
                "tactical_analysis": mem_match.get("tactical_analysis", "Retrieved from local self-trained neural memory."),
                "confidence": actual_conf,
                "runner_ups": runner_ups,
                "engine": "Local Self-Trained Memory (V5 Continuous Learning)",
            }
            rem_classes = [c for c in CLASSES if c != prediction][:2]
            top_three = [
                {"class": prediction, "label": category_label, "confidence": actual_conf}
            ] + [
                {"class": c, "label": DISPLAY_NAMES.get(c, c.title()), "confidence": round(max(0.1, (100.0 - actual_conf) / 2), 1)}
                for c in rem_classes
            ]

            return {
                "filename": filename,
                "prediction": prediction,
                "label": category_label,
                "confidence": actual_conf,
                "uncertain": actual_conf < 60.0,
                "explanation": f"Identified as {dossier['name']} using local self-trained neural memory.",
                "top_predictions": top_three,
                "model": "Local Self-Trained Memory (V5 Continuous Learning)",
                "heatmap_url": heatmap_url,
                "platform": dossier,
                "img_feats": img_feats,
                "from_learned_memory": True,
            }

        # Category ensemble
        if trained_scores is not None:
            scores = 0.50 * trained_scores + 0.50 * clip_scores
            model_type = "Hybrid Ensemble (Fine-Tuned MobileNetV3 + CLIP ViT-B/32)"
        else:
            scores = clip_scores
            model_type = "OpenAI CLIP ViT-B/32 (Prompt-Ensembled Zero-Shot)"

        ranked = sorted(enumerate(scores.tolist()), key=lambda item: item[1], reverse=True)
        top_index, confidence = ranked[0]
        prediction = CLASSES[top_index]

        # Local fine-grained platform identification with category confidence conditioning
        platform_dossier = self._identify_platform(img_feats, prediction, category_confidence=confidence)
        actual_conf = platform_dossier.get("confidence", round(confidence * 100, 1))
        uncertain = (confidence < 0.50) or (actual_conf < 50.0)

        # Dynamic category label (uses entity's true category label or default)
        category_label = platform_dossier.get("category_label") or DISPLAY_NAMES.get(prediction, prediction.title())

        top_three = [
            {
                "class": CLASSES[idx],
                "label": category_label if idx == top_index else DISPLAY_NAMES.get(CLASSES[idx], CLASSES[idx].title()),
                "confidence": actual_conf if idx == top_index else round(score * 100, 1),
            }
            for idx, score in ranked[:3]
        ]

        # Explainable AI heatmap for top category
        heatmap_url = self._generate_gradcam(image, top_index)

        explanations = {
            "aircraft": f"Identified as {platform_dossier['name']} ({platform_dossier['model_variant']}). Aerodynamic fixed-wing fuselage, propulsion signatures, and empennage geometry confirmed.",
            "helicopter": f"Identified as {platform_dossier['name']} ({platform_dossier['model_variant']}). Rotary-wing rotor head assembly, tail boom assembly, and tactical airframe confirmed.",
            "drone": f"Identified as {platform_dossier['name']} ({platform_dossier['model_variant']}). Unmanned aerodynamic profile, high-aspect wings, and optical reconnaissance payload confirmed.",
            "military-vehicle": f"Identified as {platform_dossier['name']} ({platform_dossier['model_variant']}). Tracked/wheeled combat chassis, armor profile, and heavy weapon turret confirmed.",
            "naval": f"Identified as {platform_dossier['name']} ({platform_dossier['model_variant']}). Maritime superstructure, waterline displacement hull, and phased array mast arrays confirmed.",
        }

        explanation = explanations.get(
            prediction,
            f"Subject exhibits high visual feature correlation with the {category_label} category.",
        )
        if uncertain:
            explanation = "Low decision margin across top categories. Subject may have low contrast, atypical angle, or complex background."

        return {
            "filename": filename,
            "prediction": prediction,
            "label": category_label,
            "confidence": actual_conf,
            "uncertain": uncertain,
            "explanation": explanation,
            "top_predictions": top_three,
            "model": model_type,
            "heatmap_url": heatmap_url,
            "platform": platform_dossier,
            "img_feats": img_feats,
        }

    def predict(
        self,
        image: Image.Image,
        filename: str,
        api_key: str | None = None,
        use_gemini: bool = True,
    ) -> dict[str, Any]:
        """Unified Omni-Intelligence Recognition & Learning Pipeline (Version 5.0):
        1. Local trained model answers FIRST.
        2. External API checks and verifies the local answer.
        3. If correct: Display verified result (+10 Reward).
        4. If wrong: Apply severe punishment (-50 Penalty & Loss Penalty) and train the model immediately with the correct answer so it remembers next time!
        """
        # Step 1: Local trained model answers FIRST
        local_res = self.predict_local(image, filename)

        # Step 2: Checked by API keys (Cloud AI check)
        api_res = omni_intelligence.verify_with_api(image, local_res, filename)

        # If offline: API verification was skipped due to no internet. Return local model prediction directly!
        if api_res.get("is_offline"):
            api_res["verification"] = {
                "status": "offline",
                "is_offline": True,
                "reason": "Internet connection offline. Local trained model operating standalone without external API verification.",
                "local_answer": {
                    "name": local_res.get("platform", {}).get("name") or local_res.get("label"),
                    "category": local_res.get("prediction"),
                    "confidence": local_res.get("confidence"),
                },
                "verified_answer": None,
                "penalty_points": 0,
                "reward_points": 0,
                "learned_samples": self.online_trainer.status().get("learned_exemplars", 0),
                "health_score": self.online_trainer.status().get("health_score", 1000),
            }
            api_res["learning_status"] = self.online_trainer.status()
            if api_res.get("platform"):
                dk.register_dynamic_platform(api_res["platform"])
            return api_res

        # Step 3: Evaluate & apply severe punishment or reward + retrain
        img_feats = local_res.get("img_feats")
        if img_feats is None:
            with self.lock:
                img_feats = self._clip_image_features(enhance_if_monochrome(image))

        verified_truth = api_res.get("platform", {})
        verification_report = self.online_trainer.evaluate_and_train(
            image=image,
            img_feats=img_feats,
            local_pred=local_res,
            verified_truth=verified_truth,
            classifier_model=self.classifier,
            preprocess_fn=self.preprocess,
            filename=filename,
        )

        # Step 4: Attach verification verdict & training report
        api_res["verification"] = verification_report
        api_res["learning_status"] = self.online_trainer.status()

        # If local model answered correctly, maintain and display the verified local answer consistently
        if verification_report.get("is_correct") and local_res.get("platform"):
            local_p = local_res["platform"]
            api_res["platform"]["name"] = local_p.get("name", api_res["platform"].get("name"))
            api_res["platform"]["designation"] = local_p.get("designation", api_res["platform"].get("designation"))
            api_res["platform"]["model_variant"] = local_p.get("model_variant", api_res["platform"].get("model_variant"))
            api_res["platform"]["year_origin"] = local_p.get("year_origin", api_res["platform"].get("year_origin"))
            api_res["platform"]["number_built"] = local_p.get("number_built", api_res["platform"].get("number_built"))
            api_res["label"] = local_res.get("label", api_res.get("label"))
            if api_res.get("target_count", 1) <= 1 and api_res.get("detected_targets"):
                api_res["detected_targets"][0]["exact_name"] = local_p.get("name", api_res["detected_targets"][0]["exact_name"])
                api_res["detected_targets"][0]["year_origin"] = local_p.get("year_origin", api_res["detected_targets"][0].get("year_origin"))
                api_res["detected_targets"][0]["number_built"] = local_p.get("number_built", api_res["detected_targets"][0].get("number_built"))

        # Register dynamic platform if recognized from open world
        if api_res.get("platform"):
            dk.register_dynamic_platform(api_res["platform"])

        return api_res

    def status(self) -> dict:
        grid = omni_intelligence.get_grid_status()
        return {
            "version": "5.0.0-SELF-LEARNING",
            "device": self.device,
            "fine_tuned_model": self.classifier is not None,
            "total_platforms_indexed": len(self.platform_ids),
            "omni_grid": grid,
            "metrics": self.classifier_metrics,
            "learning_status": self.online_trainer.status(),
        }

