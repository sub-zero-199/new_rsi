"""Typed configuration for the unified ANPR service.

Structure comes from a YAML file; secrets come from the environment (loaded from .env).
A single Config object is built once and passed around. Validates the ≤2-cameras-per-point
rule and that every camera is tagged with a valid point.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # python-dotenv optional at runtime
    pass

# The three physical points == the three storage buckets.
PointName = Literal["gate_in", "weighbridge", "gate_out"]
POINTS: tuple[str, ...] = ("gate_in", "weighbridge", "gate_out")


class CameraConfig(BaseModel):
    id: str
    point: PointName
    mac: str  # cameraId sent to the portal (camera MAC address)
    enabled: bool = True
    description: str = ""

    # RTSP source (used to build the URL). For dev/testing, set `source` to a file/video path.
    address: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None  # falls back to env RTSP_PASSWORD
    port: int = 554
    channel: int = 1
    subtype: int = 0
    rtsp_url: Optional[str] = None  # explicit override of the built URL
    source: Optional[str] = None  # file/video path for offline testing (overrides RTSP)

    def build_source(self, default_user: str, default_pass: str) -> str:
        """Return the OpenCV source: a file/video path, an explicit rtsp_url, or a built RTSP URL."""
        if self.source:
            return self.source
        if self.rtsp_url:
            return self.rtsp_url
        user = self.username or default_user
        pwd = self.password or default_pass
        return (
            f"rtsp://{user}:{pwd}@{self.address}:{self.port}"
            f"/rtsp/streaming?channel={self.channel}&subtype={self.subtype}"
        )


class DetectionConfig(BaseModel):
    window_seconds: float = 10.0
    min_detections: int = 3
    min_confidence: float = 0.7  # OCR char confidence gate
    detector_confidence: float = 0.75  # m1 plate-detect gate
    max_detections_per_window: int = 50
    top_n_candidates: int = 4
    # Smallest plate worth reading, as a fraction of FRAME WIDTH (so it survives a camera
    # swap). Below this the crop is too coarse for OCR: measured on 62 real frames, 0 of 13
    # reads under 4.2% of frame width produced a valid plate, while 85% did above it.
    # The pipeline's own `area < 100` guard only rejects ~17px-wide boxes — 6x too small.
    min_plate_width_frac: float = 0.042
    # Whether a point keeps detecting while one of its cameras is down.
    #   True  — run on whatever cameras are live; a damaged camera costs an angle, not the lane.
    #   False — STOP detecting at that point until every enabled camera is back, then resume
    #           automatically. Use it where one angle is not trusted to read the lane alone.
    # Only ENABLED cameras are judged, so setting a damaged camera to `enabled: false`
    # deliberately leaves a working one-camera point rather than a permanently halted one.
    allow_degraded_cameras: bool = True
    # How long a camera must stay down before it halts the point (only when the above is
    # False). RTSP streams drop a frame and reconnect routinely; without this the point
    # would stop and start on every blip.
    camera_down_grace_seconds: float = 30.0


class MatchingConfig(BaseModel):
    fuzzy_threshold: float = 0.80  # local known_plates match
    dedup_threshold: float = 0.80  # same-place "20% variation" dedup


class CooldownConfig(BaseModel):
    same_place_seconds: int = 60  # discard same plate at same point within this window
    reentry_gap_seconds: int = 300  # min gap after gate-out before re-entry (gate-in)
    weighbridge_after_gateout_gap_seconds: int = 0  # min gap after gate-out before weighbridge (0 = off)
    weighbridge_second_gap_seconds: int = 600
    weighbridge_max_per_24h: int = 10
    auto_gateout_hours: int = 24


class ApiConfig(BaseModel):
    """The RSI cloud API. One endpoint serves two purposes:

    step 1 posts the detection, step 3 re-posts the SAME payload enriched with what the
    local ANPR Data API said, so the cloud sees the local server's result even when it was
    the only thing reachable at the time.
    """
    mode: Literal["online", "offline"] = "online"
    remote_base_url: str = "https://cwc.lynkid.io"
    rsi_endpoint: str = "/api/vms/vehicle/create-triplogs-for-rsi"
    # Sent as the `secret_key` header on every RSI call. Leave it out of the file and set
    # RSI_SECRET_KEY in the environment instead — the old system hard-coded this in source.
    secret_key: str = ""
    remote_timeout: float = 30.0
    # Retries apply ONLY to a timeout or network error. Any HTTP reply — including a
    # business rejection — is final and the flow proceeds to steps 2 and 3 regardless.
    remote_retries: int = 3
    remote_retry_delay: float = 2.0
    # The exact reply that means step 1 accepted the vehicle.
    success_message: str = "Vehicle validated successfully."


class AnprDataApiConfig(BaseModel):
    """The local ANPR Data API (step 2) — on the site LAN, so it works with no internet.

    It is the only call that carries the plate image itself, base64-encoded in the JSON
    body rather than as an S3 reference.
    """
    enabled: bool = True
    url: str = "http://192.168.1.153:5000/api/ANPRData"
    timeout: float = 10.0


class S3Config(BaseModel):
    enabled: bool = True
    bucket: str = "lynkid-s3-bucket"
    prefix: str = "ANPR"  # key = {prefix}/{point}/{vehicle}/{detectionTime}/plate.jpg
    region: Optional[str] = None
    access_key_id: Optional[str] = None  # env AWS_ACCESS_KEY_ID
    secret_access_key: Optional[str] = None  # env AWS_SECRET_ACCESS_KEY
    upload_full_frame: bool = True  # also upload vehicleno.jpg alongside plate.jpg


class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379
    queue_db: int = 0
    tracker_db: int = 1


class SqliteConfig(BaseModel):
    path: str = "anpr.db"


class InferenceConfig(BaseModel):
    engine: Literal["mock", "ultralytics"] = "mock"
    device: str = "cuda"
    half: bool = True  # FP16
    use_tensorrt: bool = False
    detector_weights: Optional[str] = None
    detector_imgsz: int = 640
    rectifier_weights: Optional[str] = None
    rectifier_imgsz: int = 256
    rectified_size: tuple[int, int] = (192, 96)  # (w, h) warp output fed to OCR
    ocr_weights: Optional[str] = None
    ocr_imgsz: int = 320
    target_fps: float = 5.0  # per-camera frame-skip target
    frame_queue_size: int = 2
    # Testing only: dump the rectifier (keypoint-warp) output + the pre-warp crop that get
    # fed to OCR, so the warp can be eyeballed. OFF by default (writes many files).
    save_rectified: bool = False
    rectified_dir: str = "debug/rectified"
    rectified_limit: int = 500  # stop after N images (protects the disk)


class RetentionConfig(BaseModel):
    """Disk retention for generated data (logs, debug images, temp upload staging)."""
    enabled: bool = True
    max_age_days: int = 14          # delete logs/data older than this
    max_total_gb: float = 200.0     # hard cap: over this, delete oldest first
    check_interval_seconds: int = 3600  # how often the janitor sweeps


class Config(BaseModel):
    cameras: list[CameraConfig]
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    # per-point detection overrides, e.g. {"weighbridge": {"window_seconds": 5}}
    detection_per_point: dict[str, dict] = Field(default_factory=dict)
    matching: MatchingConfig = Field(default_factory=MatchingConfig)
    cooldown: CooldownConfig = Field(default_factory=CooldownConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    anpr_data_api: AnprDataApiConfig = Field(default_factory=AnprDataApiConfig)
    s3: S3Config = Field(default_factory=S3Config)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    sqlite: SqliteConfig = Field(default_factory=SqliteConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    retention: RetentionConfig = Field(default_factory=RetentionConfig)

    whitelist_path: str = "shared_config/whitelist.yaml"
    doppelganger_path: str = "doppelganger.csv"
    rtsp_default_username: str = "admin"
    rtsp_default_password: str = ""  # env RTSP_PASSWORD
    dry_run: bool = False

    @field_validator("cameras")
    @classmethod
    def _non_empty(cls, v):
        if not v:
            raise ValueError("config.cameras must not be empty")
        return v

    @model_validator(mode="after")
    def _max_two_per_point(self):
        counts = Counter(c.point for c in self.cameras if c.enabled)
        over = {p: n for p, n in counts.items() if n > 2}
        if over:
            raise ValueError(f"at most 2 enabled cameras per point; got {dict(over)}")
        ids = [c.id for c in self.cameras]
        if len(ids) != len(set(ids)):
            raise ValueError("camera ids must be unique")
        return self

    def detection_for(self, point: str) -> DetectionConfig:
        """DetectionConfig for a point, applying per-point overrides over the defaults."""
        override = self.detection_per_point.get(point)
        if not override:
            return self.detection
        return self.detection.model_copy(update=override)

    def enabled_cameras(self) -> list[CameraConfig]:
        return [c for c in self.cameras if c.enabled]


def _apply_env_secrets(data: dict) -> dict:
    """Overlay secrets from the environment onto the parsed YAML dict."""
    s3 = data.setdefault("s3", {})
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        s3["access_key_id"] = os.environ["AWS_ACCESS_KEY_ID"]
    if os.environ.get("AWS_SECRET_ACCESS_KEY"):
        s3["secret_access_key"] = os.environ["AWS_SECRET_ACCESS_KEY"]
    if os.environ.get("AWS_DEFAULT_REGION") and not s3.get("region"):
        s3["region"] = os.environ["AWS_DEFAULT_REGION"]

    if os.environ.get("RSI_SECRET_KEY"):
        data.setdefault("api", {})["secret_key"] = os.environ["RSI_SECRET_KEY"]

    if os.environ.get("RTSP_USERNAME"):
        data["rtsp_default_username"] = os.environ["RTSP_USERNAME"]
    if os.environ.get("RTSP_PASSWORD"):
        data["rtsp_default_password"] = os.environ["RTSP_PASSWORD"]
    return data


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    data = _apply_env_secrets(data)
    return Config.model_validate(data)
