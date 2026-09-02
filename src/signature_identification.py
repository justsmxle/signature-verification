from __future__ import annotations

import argparse
import json
import sqlite3
import threading
from pathlib import Path
from typing import Any

import numpy as np

from signature_engine import ImageInput, SignatureVerificationAPI


class SignatureIdentificationService:
    def __init__(
        self,
        weights_path: str,
        db_path: str = "signatures.sqlite3",
        confidence_threshold: float = 65.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.engine = SignatureVerificationAPI(weights_path=weights_path)
        self.db_path = Path(db_path)
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self._db_lock = threading.Lock()
        self._engine_lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._db_lock:
            with self.connection:
                self.connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS signature_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        full_name TEXT NOT NULL,
                        vector_json TEXT NOT NULL,
                        source_image_path TEXT,
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def close(self) -> None:
        with self._db_lock:
            self.connection.close()

    def __enter__(self) -> "SignatureIdentificationService":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _load_profiles(self) -> list[sqlite3.Row]:
        with self._db_lock:
            cursor = self.connection.execute(
                "SELECT id, full_name, vector_json, source_image_path, created_at FROM signature_profiles"
            )
            return cursor.fetchall()

    def _save_profile(self, full_name: str, vector: list[float], source_image_path: str) -> int:
        vector_json = json.dumps(vector, ensure_ascii=False)
        with self._db_lock:
            with self.connection:
                cursor = self.connection.execute(
                    """
                    INSERT INTO signature_profiles (full_name, vector_json, source_image_path)
                    VALUES (?, ?, ?)
                    """,
                    (full_name, vector_json, source_image_path),
                )
            return int(cursor.lastrowid)

    def _find_best_match(self, query_vector: list[float]) -> dict[str, Any] | None:
        best_match: dict[str, Any] | None = None
        for profile in self._load_profiles():
            stored_vector = np.asarray(json.loads(profile["vector_json"]), dtype=np.float32).tolist()
            comparison = self.engine.compare_vectors(query_vector, stored_vector)
            candidate = {
                "profile_id": int(profile["id"]),
                "full_name": profile["full_name"],
                "distance": float(comparison["distance"]),
                "confidence_score": float(comparison["confidence_score"]),
            }
            if best_match is None or candidate["confidence_score"] > best_match["confidence_score"]:
                best_match = candidate
        return best_match

    def get_profiles_count(self) -> int:
        with self._db_lock:
            cursor = self.connection.execute("SELECT COUNT(*) AS cnt FROM signature_profiles")
            row = cursor.fetchone()
        return int(row["cnt"]) if row else 0

    def get_signature_vector(self, image_path_or_bytes: ImageInput) -> list[float]:
        with self._engine_lock:
            return self.engine.get_signature_vector(image_path_or_bytes)

    def find_best_match(self, query_vector: list[float]) -> dict[str, Any] | None:
        return self._find_best_match(query_vector)

    def save_profile(self, full_name: str, vector: list[float], source_image_path: str = "") -> int:
        return self._save_profile(
            full_name=full_name.strip(),
            vector=vector,
            source_image_path=source_image_path,
        )

    def identify_or_enroll(
        self,
        image_path: str,
        full_name: str | None = None,
        interactive: bool = True,
    ) -> dict[str, Any]:
        query_vector = self.get_signature_vector(image_path)
        best_match = self.find_best_match(query_vector)

        if best_match and best_match["confidence_score"] > self.confidence_threshold:
            return {
                "status": "matched_existing",
                "profile_id": best_match["profile_id"],
                "full_name": best_match["full_name"],
                "distance": best_match["distance"],
                "confidence_score": best_match["confidence_score"],
            }

        if not full_name and interactive:
            full_name = input("Новая подпись. Введите ФИО: ").strip()

        if not full_name:
            raise ValueError(
                "Подпись не найдена в базе и ФИО не передано. "
                "Передайте --name или используйте интерактивный режим."
            )

        new_profile_id = self.save_profile(
            full_name=full_name,
            vector=query_vector,
            source_image_path=image_path,
        )
        response: dict[str, Any] = {
            "status": "new_enrolled",
            "profile_id": new_profile_id,
            "full_name": full_name,
            "confidence_threshold": self.confidence_threshold,
        }
        if best_match:
            response["best_existing_candidate"] = best_match
        return response


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent
    default_weights = project_dir / "siamese_signature_best.pth"
    default_db = project_dir / "signatures.sqlite3"

    parser = argparse.ArgumentParser(description="Signature identification with SQLite storage")
    parser.add_argument("--image", type=str, required=True, help="Path to signature image")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(default_weights),
        help="Path to .pth weights file",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(default_db),
        help="Path to SQLite database file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=65.0,
        help="Treat as same person if confidence_score is greater than this value",
    )
    parser.add_argument("--name", type=str, default=None, help="Full name for enrolling a new signature")
    parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Disable interactive prompt for full name when signature is new",
    )
    args = parser.parse_args()

    required_paths = [Path(args.image), Path(args.weights)]
    missing_paths = [str(path) for path in required_paths if not path.exists()]
    if missing_paths:
        print("Missing required file(s):")
        for missing_path in missing_paths:
            print(f"- {missing_path}")
        raise SystemExit(1)

    with SignatureIdentificationService(
        weights_path=args.weights,
        db_path=args.db_path,
        confidence_threshold=args.confidence_threshold,
    ) as service:
        result = service.identify_or_enroll(
            image_path=args.image,
            full_name=args.name,
            interactive=not args.non_interactive,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
