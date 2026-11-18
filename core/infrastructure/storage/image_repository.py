from __future__ import annotations
import hashlib
from datetime import datetime, timezone
from pathlib import Path
import cv2
from sqlalchemy import select
from core.infrastructure.database.models import CaptureImage, SystemSetting


class ImageRepository:
    def __init__(self, database, root: str | Path):
        self.database = database
        self.root = Path(root).resolve()

    def register(self, *, capture_id: str, camera_id: str, image_type: str, path: str | Path,
                 captured_at_ms: int, width: int, height: int, confidence: float | None = None,
                 quality_score: float | None = None, parent_image_id: int | None = None) -> int:
        target = Path(path).resolve()
        if self.root not in target.parents or not target.is_file():
            raise ValueError("image must exist under configured image storage root")
        row = CaptureImage(capture_id=capture_id, camera_id=camera_id, image_type=image_type,
            file_path=target.relative_to(self.root).as_posix(), captured_at_ms=captured_at_ms,
            width=width, height=height, file_size=target.stat().st_size, confidence=confidence,
            quality_score=quality_score, parent_image_id=parent_image_id, created_at_ms=captured_at_ms)
        with self.database.sessions.begin() as session:
            session.add(row); session.flush(); return row.id

    def import_legacy_files(self) -> int:
        """One-time index of existing v0.1.x JPEG files; image bytes are never copied into SQLite."""
        with self.database.sessions() as session:
            if session.get(SystemSetting, "legacy_images_imported") is not None:
                return 0
            existing = set(session.scalars(select(CaptureImage.file_path)).all())
        groups: dict[tuple[str, str], list[tuple[str, Path, int | None]]] = {}
        if self.root.exists():
            for camera_dir in self.root.iterdir():
                if not camera_dir.is_dir() or camera_dir.name in {"database", "secrets", "temp"}:
                    continue
                for path in camera_dir.glob("*.jpg"):
                    name = path.name
                    if name.endswith("_frame.jpg"):
                        kind, stamp, score = "frame", name[:-10], None
                    elif name.endswith("_detections.jpg"):
                        kind, stamp, score = "annotated", name[:-15], None
                    elif "_face_" in name:
                        kind, stamp = "face", name.split("_face_", 1)[0]
                        try: score = int(float(name.split("_score_", 1)[1].removesuffix(".jpg")) * 10000)
                        except (IndexError, ValueError): score = None
                    else:
                        continue
                    try:
                        datetime.strptime(stamp, "%Y%m%dT%H%M%S_%fZ")
                    except ValueError:
                        continue
                    groups.setdefault((camera_dir.name, stamp), []).append((kind, path, score))
        imported = 0
        now = int(datetime.now(timezone.utc).timestamp() * 1000)
        with self.database.sessions.begin() as session:
            for (camera_id, stamp), files in sorted(groups.items()):
                captured_at_ms = int(datetime.strptime(stamp, "%Y%m%dT%H%M%S_%fZ").replace(tzinfo=timezone.utc).timestamp() * 1000)
                capture_id = hashlib.sha256(f"{camera_id}:{stamp}".encode()).hexdigest()[:32]
                frame_id = None
                order = {"frame": 0, "annotated": 1, "face": 2}
                for kind, path, score_scaled in sorted(files, key=lambda x: order[x[0]]):
                    relative = path.resolve().relative_to(self.root).as_posix()
                    if relative in existing:
                        continue
                    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                    if image is None:
                        continue
                    height, width = image.shape[:2]
                    confidence = score_scaled / 10000.0 if kind == "face" and score_scaled is not None else None
                    row = CaptureImage(capture_id=capture_id, camera_id=camera_id,
                        parent_image_id=frame_id if kind == "face" else None, image_type=kind,
                        file_path=relative, captured_at_ms=captured_at_ms, confidence=confidence,
                        quality_score=None, width=width, height=height, file_size=path.stat().st_size,
                        created_at_ms=now)
                    session.add(row); session.flush()
                    if kind == "frame": frame_id = row.id
                    existing.add(relative); imported += 1
            session.add(SystemSetting(key="legacy_images_imported", value_json=str(imported), updated_at_ms=now))
        return imported
