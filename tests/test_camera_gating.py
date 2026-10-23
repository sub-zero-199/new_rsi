"""A point that loses one of its cameras: keep reading, or stop until it is back.

`detection.allow_degraded_cameras` decides. True (the default) keeps the lane running on
whatever cameras are live. False halts that point once a camera has been down longer than
`camera_down_grace_seconds`, and resumes on its own when it returns.

The grace period is the point of the design: RTSP streams drop a frame and reconnect as a
matter of course, and halting on every blip would stop and start the lane all day.
"""
from anpr.models.pipeline import Bbox, MockEngine, PlateCandidate
from anpr.worker import InferenceWorker

T0 = 1_700_000_000.0


class _FakeCam:
    """Stands in for a CameraCapture — the worker only reads `.status` and `.cam`."""

    def __init__(self, cam_id, point, status="running"):
        self.cam = type("Cam", (), {"id": cam_id, "point": point})()
        self.status = status


class _FakeCaptures:
    def __init__(self, *caps):
        self.captures = {c.cam.id: c for c in caps}


def _worker(cfg, *caps):
    """Build a worker over these cameras. Config is read at construction, as in the service."""
    return InferenceWorker(cfg, MockEngine(), _FakeCaptures(*caps), lambda det: None)


def _pair(point="gate_in", second_status="running"):
    return _FakeCam(f"{point}_cam1", point), _FakeCam(f"{point}_cam2", point, second_status)


def _observe(w):
    """One poll at T0, as the running worker does every 20ms — this is what starts the
    grace timer. Without it the outage looks like it began at the moment of the next call."""
    assert w._apply_halts(T0) == set()
    return w


# ---- default: a damaged camera costs an angle, not the lane ----

def test_degraded_operation_is_allowed_by_default(cfg):
    assert cfg.detection.allow_degraded_cameras is True
    w = _worker(cfg, *_pair(second_status="disconnected"))

    assert w._apply_halts(T0 + 86_400) == set()   # still running a day later


# ---- strict mode ----

def test_strict_mode_waits_out_the_grace_period(cfg):
    """A brief reconnect must not stop the lane."""
    cfg.detection.allow_degraded_cameras = False
    cfg.detection.camera_down_grace_seconds = 30.0
    w = _worker(cfg, *_pair(second_status="reconnecting"))

    assert w._apply_halts(T0) == set()            # first seen down now
    assert w._apply_halts(T0 + 29) == set()       # still inside the grace window
    assert w._apply_halts(T0 + 31) == {"gate_in"}


def test_strict_mode_resumes_by_itself(cfg):
    cfg.detection.allow_degraded_cameras = False
    cam1, cam2 = _pair(second_status="disconnected")
    w = _observe(_worker(cfg, cam1, cam2))
    assert w._apply_halts(T0 + 60) == {"gate_in"}

    cam2.status = "running"
    assert w._apply_halts(T0 + 61) == set()


def test_only_the_affected_point_halts(cfg):
    """Gate-in stopping must not touch the weighbridge or gate-out."""
    cfg.detection.allow_degraded_cameras = False
    w = _observe(_worker(cfg, *_pair("gate_in", second_status="disconnected"),
                         *_pair("weighbridge")))

    assert w._apply_halts(T0 + 60) == {"gate_in"}


def test_a_point_may_be_strict_while_others_are_not(cfg):
    """The per-point override is what makes one lane stricter than the rest."""
    cfg.detection.allow_degraded_cameras = True
    cfg.detection_per_point = {"gate_out": {"allow_degraded_cameras": False}}
    w = _observe(_worker(cfg,
                         *_pair("gate_in", second_status="disconnected"),
                         *_pair("gate_out", second_status="disconnected")))

    assert w._apply_halts(T0 + 60) == {"gate_out"}


# ---- a deliberately disabled camera is not a fault ----

def test_a_disabled_camera_never_halts_the_point(cfg):
    """CaptureManager is built from enabled_cameras(), so a camera set `enabled: false`
    never reaches the worker. Disabling a damaged one is how you keep a lane running."""
    cfg.detection.allow_degraded_cameras = False
    w = _worker(cfg, _FakeCam("gatein_cam2", "gate_in"))   # cam1 disabled in config

    assert w._apply_halts(T0 + 86_400) == set()


# ---- a halt must not leave half a read behind ----

def test_in_flight_windows_are_discarded_on_halt(cfg, img, frame):
    """Half a window collected before the camera died must not finalize later as though it
    were a full read of the vehicle."""
    cfg.detection.allow_degraded_cameras = False
    cam1, cam2 = _pair(second_status="disconnected")
    w = _observe(_worker(cfg, cam1, cam2))
    w.aggregator.add("gatein_cam1", "gate_in",
                     PlateCandidate("MH12AB1234", 0.9, Bbox(0, 0, 10, 10), img), frame, T0)
    assert len(w.aggregator._windows) == 1

    w._apply_halts(T0 + 60)

    assert w.aggregator._windows == {}
    assert w.aggregator.sweep(T0 + 600) == []     # nothing resurfaces after the window elapses


def test_other_points_keep_their_windows(cfg, img, frame):
    cfg.detection.allow_degraded_cameras = False
    w = _observe(_worker(cfg, *_pair("gate_in", second_status="disconnected"),
                         *_pair("weighbridge")))
    w.aggregator.add("weighbridge_cam1", "weighbridge",
                     PlateCandidate("MH12AB1234", 0.9, Bbox(0, 0, 10, 10), img), frame, T0)

    w._apply_halts(T0 + 60)

    assert len(w.aggregator._windows) == 1
