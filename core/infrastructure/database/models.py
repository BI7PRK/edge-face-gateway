from __future__ import annotations

from sqlalchemy import Boolean, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    main_url_encrypted: Mapped[str] = mapped_column(Text, nullable=False)
    sub_url_encrypted: Mapped[str | None] = mapped_column(Text)
    stream_mode: Mapped[str] = mapped_column(String(16), nullable=False, default="single")
    detection_fps: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    config_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    updated_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class CameraStatus(Base):
    __tablename__ = "camera_status"
    camera_id: Mapped[str] = mapped_column(ForeignKey("cameras.id", ondelete="CASCADE"), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="STOPPED")
    last_connected_at_ms: Mapped[int | None] = mapped_column(Integer)
    last_disconnected_at_ms: Mapped[int | None] = mapped_column(Integer)
    last_error: Mapped[str | None] = mapped_column(Text)
    reconnect_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    current_fps: Mapped[float] = mapped_column(default=0.0)
    worker_state: Mapped[str] = mapped_column(String(24), nullable=False, default="STOPPED")
    updated_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class CaptureImage(Base):
    __tablename__ = "capture_images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    capture_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    camera_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    parent_image_id: Mapped[int | None] = mapped_column(ForeignKey("capture_images.id", ondelete="SET NULL"))
    image_type: Mapped[str] = mapped_column(String(24), nullable=False)
    file_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)
    captured_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column()
    quality_score: Mapped[float | None] = mapped_column()
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    cloud_key: Mapped[str | None] = mapped_column(Text)
    cloud_uploaded_at_ms: Mapped[int | None] = mapped_column(Integer)
    cloud_upload_status: Mapped[str | None] = mapped_column(String(24))
    __table_args__ = (Index("ix_capture_camera_time", "camera_id", "captured_at_ms", "id"),
                      Index("ix_capture_type_time", "image_type", "captured_at_ms", "id"))


class ApplicationLog(Base):
    __tablename__ = "application_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False)
    logger: Mapped[str] = mapped_column(String(200), nullable=False)
    category: Mapped[str] = mapped_column(String(32), nullable=False, default="SYSTEM")
    camera_id: Mapped[str | None] = mapped_column(String(128))
    message: Mapped[str] = mapped_column(Text, nullable=False)
    details_json: Mapped[str | None] = mapped_column(Text)
    exception: Mapped[str | None] = mapped_column(Text)
    __table_args__ = (Index("ix_logs_time", "timestamp_ms", "id"),
                      Index("ix_logs_camera_time", "camera_id", "timestamp_ms", "id"),
                      Index("ix_logs_level_time", "level", "timestamp_ms", "id"))


class BusinessEvent(Base):
    __tablename__ = "business_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_type: Mapped[str] = mapped_column(String(40), nullable=False)
    timestamp_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(128))
    capture_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str | None] = mapped_column(String(24))
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    __table_args__ = (Index("ix_events_time", "timestamp_ms", "id"),
                      Index("ix_events_type_time", "event_type", "timestamp_ms", "id"),
                      Index("ix_events_camera_time", "camera_id", "timestamp_ms", "id"))


class SystemSetting(Base):
    __tablename__ = "system_settings"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value_json: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)


class MqttOutboxEvent(Base):
    __tablename__ = "mqtt_outbox"
    event_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)
    qos: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    retain: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_attempt_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    published_at_ms: Mapped[int | None] = mapped_column(Integer)
    __table_args__ = (Index("ix_mqtt_outbox_due", "status", "next_attempt_at_ms", "created_at_ms"),)
