import time
from anpr.aggregation import WindowAggregator
from anpr.models.pipeline import PlateCandidate, Bbox


def _cand(text, img):
    return PlateCandidate(text, 0.9, Bbox(0, 0, 10, 10), img, [0.9] * len(text))


def test_window_finalizes_after_expiry(cfg, img, frame):
    agg = WindowAggregator(cfg)
    t = time.time()
    for i in range(5):
        agg.add("gatein_cam1", "gate_in", _cand("MH12AB1234", img), frame, t + i * 0.1)
    assert agg.sweep(t + 1) == []          # 10s window not elapsed
    out = agg.sweep(t + 11)
    assert len(out) == 1
    assert out[0].best_text == "MH12AB1234"
    assert out[0].detection_count == 5
    assert out[0].point == "gate_in"


def test_too_few_detections_dropped(cfg, img, frame):
    agg = WindowAggregator(cfg)
    t = time.time()
    agg.add("c", "gate_in", _cand("KA05CD9999", img), frame, t)  # only 1 < min 3
    assert agg.sweep(t + 11) == []


def test_ocr_variants_group_into_one_window(cfg, img, frame):
    agg = WindowAggregator(cfg)
    t = time.time()
    # OCR jitter of the same plate should collapse into one window (>=0.8 similar)
    for i, txt in enumerate(["MH12AB1234", "MH12A81234", "MH12AB1234"]):
        agg.add("gatein_cam1", "gate_in", _cand(txt, img), frame, t + i * 0.1)
    out = agg.sweep(t + 11)
    assert len(out) == 1
    assert out[0].detection_count == 3
