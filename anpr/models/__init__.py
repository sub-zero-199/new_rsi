"""Model layer: the 3-stage plate pipeline (detect -> rectify -> read).

`PlatePipeline` is the contract; `MockEngine` runs it without a GPU; `UltralyticsEngine`
(in engine.py) is the real YOLOv11m implementation, loaded lazily so importing this
package never requires torch.
"""
from anpr.models.pipeline import (
    Bbox,
    PlateCandidate,
    PlatePipeline,
    MockEngine,
    build_pipeline,
)

__all__ = ["Bbox", "PlateCandidate", "PlatePipeline", "MockEngine", "build_pipeline"]
