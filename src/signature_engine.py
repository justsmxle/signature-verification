from __future__ import annotations
from huggingface_hub import hf_hub_download

import argparse
import io
from pathlib import Path
from typing import Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class SignatureEmbedder(nn.Module):
    def __init__(self) -> None:
        super(SignatureEmbedder, self).__init__()
        base_model = models.resnet18(weights=None)
        self.features = nn.Sequential(*(list(base_model.children())[:-1]))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.fc(x)
        return x


ImageInput = Union[str, Path, bytes, bytearray, Image.Image]


class SignatureVerificationAPI:
    def __init__(self, weights_path: str = None, threshold: float = 1.0) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.model = SignatureEmbedder().to(self.device)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        if weights_path is None:
            weights_path = hf_hub_download(
                repo_id="SMXLEEE/signature-verifier", 
                filename="siamese_signature_best.pth"
            )

        state_dict = self._extract_state_dict(Path(weights_path))
        self._load_model_weights(state_dict)
        self.model.eval()

    @staticmethod
    def _is_state_dict(candidate: object) -> bool:
        if not isinstance(candidate, dict) or not candidate:
            return False
        for key, value in candidate.items():
            if not isinstance(key, str) or not torch.is_tensor(value):
                return False
        return True

    @staticmethod
    def _normalize_state_key(key: str) -> str:
        if key.startswith("module."):
            key = key[len("module.") :]
        for marker in ("features.", "fc."):
            marker_index = key.find(marker)
            if marker_index != -1:
                return key[marker_index:]
        return key

    def _extract_state_dict(self, weights_path: Path) -> dict[str, torch.Tensor]:
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(weights_path, map_location=self.device)
        if self._is_state_dict(checkpoint):
            return checkpoint  # type: ignore[return-value]

        if isinstance(checkpoint, dict):
            for key in (
                "state_dict",
                "model_state_dict",
                "model",
                "net",
                "network",
                "siamese_state_dict",
            ):
                value = checkpoint.get(key)
                if self._is_state_dict(value):
                    return value  # type: ignore[return-value]

        raise RuntimeError(
            "Unsupported checkpoint format. Expected a state_dict or checkpoint with model state_dict."
        )

    def _load_model_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        model_state_dict = self.model.state_dict()
        model_keys = set(model_state_dict.keys())

        # Берем только нужные тензоры и приводим ключи к формату SignatureEmbedder.
        normalized_state_dict: dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            normalized_key = self._normalize_state_key(key)
            if normalized_key in model_keys:
                normalized_state_dict[normalized_key] = value

        if len(normalized_state_dict) == len(model_keys):
            self.model.load_state_dict(normalized_state_dict, strict=True)
            return

        # Fallback: иногда state_dict уже полностью совместим без нормализации ключей.
        try:
            self.model.load_state_dict(state_dict, strict=True)
            return
        except RuntimeError as err:
            missing = [key for key in model_keys if key not in normalized_state_dict]
            missing_preview = ", ".join(sorted(missing)[:6])
            raise RuntimeError(
                "Could not load model weights into SignatureEmbedder. "
                f"Missing keys sample: {missing_preview}. Original error: {err}"
            ) from err

    @staticmethod
    def _to_pil_image(image_path_or_bytes: ImageInput) -> Image.Image:
        if isinstance(image_path_or_bytes, Image.Image):
            return image_path_or_bytes.convert("RGB")

        if isinstance(image_path_or_bytes, (str, Path)):
            with Image.open(image_path_or_bytes) as image:
                return image.convert("RGB")

        if isinstance(image_path_or_bytes, (bytes, bytearray)):
            with Image.open(io.BytesIO(image_path_or_bytes)) as image:
                return image.convert("RGB")

        raise TypeError(
            "image_path_or_bytes must be str | Path | bytes | bytearray | PIL.Image.Image"
        )

    def get_signature_vector(self, image_path_or_bytes: ImageInput) -> list[float]:
        image = self._to_pil_image(image_path_or_bytes)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            vector = self.model(image_tensor).squeeze(0).cpu().numpy().astype(np.float32)

        return vector.tolist()

    def compare_vectors(self, vector_1: list[float], vector_2: list[float]) -> dict[str, float | bool]:
        v1 = np.asarray(vector_1, dtype=np.float32).reshape(-1)
        v2 = np.asarray(vector_2, dtype=np.float32).reshape(-1)

        if v1.shape[0] != 128 or v2.shape[0] != 128:
            raise ValueError("Both vectors must have 128 elements.")

        distance = float(np.linalg.norm(v1 - v2))
        is_match = distance < self.threshold
        confidence_score = max(0.0, (1.0 - (distance / 2.0)) * 100.0)

        return {
            "distance": distance,
            "is_match": bool(is_match),
            "confidence_score": confidence_score,
        }

    def enroll_user(self, image_paths: list[str]) -> list[float]:
        if not image_paths:
            raise ValueError("image_paths must contain at least one image path.")

        vectors = [
            np.asarray(self.get_signature_vector(image_path), dtype=np.float32)
            for image_path in image_paths
        ]
        mean_vector = np.mean(np.stack(vectors, axis=0), axis=0).astype(np.float32)
        return mean_vector.tolist()


class SignatureEngine(SignatureVerificationAPI):
    pass


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    default_weights = project_dir / "siamese_signature_best.pth"
    default_image_1 = project_dir / "sign_1.png"
    default_image_2 = project_dir / "sign_2.png"

    parser = argparse.ArgumentParser(description="Offline Signature Verification Inference Demo")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to model weights (.pth).",
    )
    parser.add_argument(
        "--image-1",
        type=str,
        default=str(default_image_1),
        help="Path to enrollment image.",
    )
    parser.add_argument(
        "--image-2",
        type=str,
        default=str(default_image_2),
        help="Path to verification image.",
    )
    args = parser.parse_args()

    required_files = [Path(args.weights), Path(args.image_1), Path(args.image_2)]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        print("Missing required file(s):")
        for missing_file in missing_files:
            print(f"- {missing_file}")
        raise SystemExit(1)

    engine = SignatureVerificationAPI(weights_path=args.weights)
    enrolled_vector = engine.enroll_user([args.image_1])
    probe_vector = engine.get_signature_vector(args.image_2)
    result = engine.compare_vectors(enrolled_vector, probe_vector)

    print(result)
