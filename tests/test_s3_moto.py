"""Image upload — the four-key layout, and the step-3 call that follows a deferred upload."""
import json
import os

import boto3
import pytest
from moto import mock_aws

from anpr.s3uploader import (ImageBackgroundProcessor, S3Uploader, compress_and_save_image,
                             latest_s3_key, plate_s3_key)
from anpr.storage_redis import RedisQueue


class FakeNet:
    def is_online(self):
        return True

    def check(self, force=False):
        return True


class _Portal:
    """Records step-3 calls so a test can assert on what was sent, and when."""

    def __init__(self):
        self.calls = []

    def send_enriched(self, payload, details):
        self.calls.append((payload, details))
        return True


def _bucket():
    boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="test-bucket")
    return boto3.client("s3", region_name="us-east-1")


# ---- key layout ----

def test_keys_match_the_layout_the_rsi_backend_reads(cfg):
    """The point is spelled without an underscore, as rsi_current wrote it."""
    assert plate_s3_key("ANPR", "gate_out", "MH12AB1234", 7) == "ANPR/gateout/MH12AB1234/7/plate.jpg"
    assert plate_s3_key("ANPR", "gate_in", "MH12AB1234", 7) == "ANPR/gatein/MH12AB1234/7/plate.jpg"
    assert plate_s3_key("ANPR", "weighbridge", "X", 7) == "ANPR/weighbridge/X/7/plate.jpg"
    assert latest_s3_key("ANPR", "gate_out", "MH12AB1234") == "ANPR/gateout/MH12AB1234/plate.jpg"


@mock_aws
def test_s3_upload_verifies(cfg, img, tmp_path):
    cfg.s3.bucket, cfg.s3.region, cfg.dry_run = "test-bucket", "us-east-1", False
    s3 = _bucket()
    plate_path = str(tmp_path / "plate.jpg")
    assert compress_and_save_image(img, plate_path, is_plate=True)
    key = plate_s3_key(cfg.s3.prefix, "gate_out", "MH12AB1234", 1, "plate.jpg")

    assert S3Uploader(cfg.s3).upload(plate_path, key)

    s3.head_object(Bucket="test-bucket", Key=key)


@mock_aws
def test_each_image_lands_twice_timestamped_and_latest(cfg, img, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg.s3.bucket, cfg.s3.region, cfg.dry_run = "test-bucket", "us-east-1", False
    cfg.s3.upload_full_frame = True
    s3 = _bucket()

    q = RedisQueue(namespace="test_s3")            # local-file fallback
    plate_path, full_path = str(tmp_path / "plate.jpg"), str(tmp_path / "vehicleno.jpg")
    compress_and_save_image(img, plate_path, is_plate=True)
    compress_and_save_image(img, full_path, is_plate=False)
    q.store_detection({"vehicle_number": "MH12AB1234", "plate_img_path": plate_path,
                       "full_img_path": full_path, "detection_time": 1, "point": "gate_out"})

    proc = ImageBackgroundProcessor(cfg, q, FakeNet())
    proc._process_one(*q.get_pending_detections(5)[0])

    for key in ("ANPR/gateout/MH12AB1234/1/plate.jpg",
                "ANPR/gateout/MH12AB1234/plate.jpg",
                "ANPR/gateout/MH12AB1234/1/vehicleno.jpg",
                "ANPR/gateout/MH12AB1234/vehicleno.jpg"):
        s3.head_object(Bucket="test-bucket", Key=key)
    assert not os.path.exists(plate_path)          # locals go only after every upload verified


# ---- the deferred step-3 call ----

@mock_aws
def test_a_deferred_detection_sends_step3_once_its_images_land(cfg, img, tmp_path, monkeypatch):
    """Queued while the cloud was down. Only after the images are in S3 can the cloud
    resolve anprImagePath, so that is when the enriched call is worth making."""
    monkeypatch.chdir(tmp_path)
    cfg.s3.bucket, cfg.s3.region, cfg.dry_run = "test-bucket", "us-east-1", False
    cfg.s3.upload_full_frame = False
    _bucket()

    q = RedisQueue(namespace="test_s3_deferred")
    plate_path = str(tmp_path / "plate.jpg")
    compress_and_save_image(img, plate_path, is_plate=True)
    payload = {"cameraId": "90f7", "vehicleNumber": "MH12AB1234", "detectionTime": 1_700_000_000}
    details = {"url": "http://local/api/ANPRData", "request_body": {},
               "response": {"error": True, "message": "refused"},
               "request_time": "t0", "response_time": "t1"}
    q.store_detection({"vehicle_number": "MH12AB1234", "plate_img_path": plate_path,
                       "full_img_path": "", "detection_time": 1_700_000_000,
                       "point": "gate_out", "offline_mode": True,
                       "rsi_payload": json.dumps(payload),
                       "anpr_details": json.dumps(details)})

    portal = _Portal()
    proc = ImageBackgroundProcessor(cfg, q, FakeNet(), portal=portal)
    proc._process_one(*q.get_pending_detections(5)[0])

    assert len(portal.calls) == 1
    sent_payload, sent_details = portal.calls[0]
    assert sent_payload["detectionTime"] == 1_700_000_000   # when the vehicle passed, not now
    assert sent_details["response"]["message"] == "refused"  # the local failure is carried up


@mock_aws
def test_an_online_detection_does_not_send_step3_again(cfg, img, tmp_path, monkeypatch):
    """Online, step 3 already went out inline — the uploader must not repeat it."""
    monkeypatch.chdir(tmp_path)
    cfg.s3.bucket, cfg.s3.region, cfg.dry_run = "test-bucket", "us-east-1", False
    cfg.s3.upload_full_frame = False
    _bucket()

    q = RedisQueue(namespace="test_s3_online")
    plate_path = str(tmp_path / "plate.jpg")
    compress_and_save_image(img, plate_path, is_plate=True)
    q.store_detection({"vehicle_number": "MH12AB1234", "plate_img_path": plate_path,
                       "full_img_path": "", "detection_time": 1, "point": "gate_out"})

    portal = _Portal()
    proc = ImageBackgroundProcessor(cfg, q, FakeNet(), portal=portal)
    proc._process_one(*q.get_pending_detections(5)[0])

    assert portal.calls == []


@mock_aws
def test_a_failed_upload_defers_step3_too(cfg, tmp_path, monkeypatch):
    """No images in S3 means anprImagePath would not resolve — so do not announce it yet."""
    monkeypatch.chdir(tmp_path)
    cfg.s3.bucket, cfg.s3.region, cfg.dry_run = "missing-bucket", "us-east-1", False
    cfg.s3.upload_full_frame = False

    q = RedisQueue(namespace="test_s3_failed")
    plate_path = str(tmp_path / "plate.jpg")
    with open(plate_path, "wb") as f:
        f.write(b"\xff\xd8not-a-real-upload")
    q.store_detection({"vehicle_number": "MH12AB1234", "plate_img_path": plate_path,
                       "full_img_path": "", "detection_time": 1, "point": "gate_out",
                       "rsi_payload": json.dumps({"vehicleNumber": "MH12AB1234"}),
                       "anpr_details": json.dumps({"url": "u", "response": {}})})

    portal = _Portal()
    proc = ImageBackgroundProcessor(cfg, q, FakeNet(), portal=portal)
    proc._process_one(*q.get_pending_detections(5)[0])

    assert portal.calls == []
    assert os.path.exists(plate_path)              # preserved for the retry
