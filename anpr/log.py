"""Logging for the ANPR service. Ported from v11_1 logger.py.

Colored output on a TTY; ASCII-stripped + plain on a non-TTY (systemd/journald).
Also provides a small throttle to suppress repetitive per-vehicle messages.
"""
import logging
import os
import sys
import time
import threading
from logging.handlers import TimedRotatingFileHandler

# Keep this many days of each log file. Rotates daily at midnight; older days are deleted.
# Overridable via env (the service sets it from config.retention.max_age_days at startup).
_LOG_RETENTION_DAYS = max(1, int(os.environ.get("ANPR_LOG_RETENTION_DAYS", "14")))

logger = logging.getLogger("cwc_anpr")
logger.setLevel(logging.INFO)
logger.propagate = False
logger.handlers.clear()

_IS_TTY = bool(getattr(sys.stdout, "isatty", lambda: False)())


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
        "RESET": "\033[0m",
    }

    def format(self, record):
        formatted = super().format(record)
        if _IS_TTY:
            color = self.COLORS.get(record.levelname, self.COLORS["RESET"])
            formatted = f"{color}{formatted}{self.COLORS['RESET']}"
        return formatted


_FMT = "[%(asctime)s] %(levelname)s - %(message)s"
_DATEFMT = "%d/%b/%Y %H:%M:%S"
_formatter = ColoredFormatter(_FMT, datefmt=_DATEFMT) if _IS_TTY else logging.Formatter(_FMT, datefmt=_DATEFMT)

_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_formatter)
_handler.setLevel(logging.INFO)
logger.addHandler(_handler)

if not _IS_TTY:
    class _AsciiFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            try:
                msg = record.getMessage()
                record.msg = msg.encode("ascii", "ignore").decode("ascii")
                record.args = ()
            except Exception:
                pass
            return True

    _handler.addFilter(_AsciiFilter())


# ---------------------------------------------------------------------------
# Per-point + camera-status log FILES.
# Each returns a child logger that writes to logs/<name>.log AND still bubbles to the
# console/journald handler above. Set ANPR_LOG_DIR to change the directory;
# set ANPR_DISABLE_FILE_LOGS=1 to turn files off (used by the test suite).
# ---------------------------------------------------------------------------
_POINTS = ("gate_in", "weighbridge", "gate_out")
# short log filenames: logs/gi.log, logs/wb.log, logs/go.log
_POINT_FILE = {"gate_in": "gi", "weighbridge": "wb", "gate_out": "go"}
_file_loggers: dict = {}


def _file_logger(name: str, propagate: bool = True) -> logging.Logger:
    if os.environ.get("ANPR_DISABLE_FILE_LOGS"):
        return logger
    cached = _file_loggers.get(name)
    if cached is not None:
        return cached
    lg = logging.getLogger(f"cwc_anpr.{name}")
    lg.setLevel(logging.INFO)
    lg.propagate = propagate  # True => also reach the console/journald handler on the parent
    try:
        log_dir = os.environ.get("ANPR_LOG_DIR", "logs")
        os.makedirs(log_dir, exist_ok=True)
        # Rotate daily at midnight; keep the last N days (older days auto-deleted). N comes
        # from config.retention.max_age_days via the ANPR_LOG_RETENTION_DAYS env (default 14).
        days = max(1, int(os.environ.get("ANPR_LOG_RETENTION_DAYS", _LOG_RETENTION_DAYS)))
        fh = TimedRotatingFileHandler(os.path.join(log_dir, f"{name}.log"),
                                      when="midnight", backupCount=days, encoding="utf-8")
        fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))
        lg.addHandler(fh)
    except Exception as e:
        logger.warning(f"file logging for {name!r} disabled: {e}")
        lg = logger
    _file_loggers[name] = lg
    return lg


def point_logger(point: str):
    """Per-point logger -> logs/gi.log | wb.log | go.log (else logs/system.log).
    Propagates to the console/journald handler too."""
    return _file_logger(_POINT_FILE.get(point, "system"))


def camera_logger():
    """Aggregate camera logger -> logs/cs.log only (does NOT hit the console; the console
    copy comes from the point logger via camera_event())."""
    return _file_logger("cs", propagate=False)


def camera_event(point: str, msg: str, level: str = "info") -> None:
    """A camera's connectivity event -> its point log (+ console) AND camera_status.log."""
    getattr(point_logger(point), level, point_logger(point).info)(msg)
    cam = camera_logger()
    if cam is not point_logger(point):  # avoid a double line when file logs are disabled
        getattr(cam, level, cam.info)(msg)


def camera_info(msg: str, level: str = "info") -> None:
    """Aggregate camera line (map, counts) -> console AND camera_status.log."""
    getattr(logger, level, logger.info)(msg)
    cam = camera_logger()
    if cam is not logger:
        getattr(cam, level, cam.info)(msg)


class LogThrottle:
    """Throttle repetitive log messages keyed by (message_key, vehicle)."""

    def __init__(self, default_interval: float = 5.0):
        self.default_interval = default_interval
        self.last_logged: dict = {}
        self.lock = threading.Lock()

    def should_log(self, message_key, vehicle=None, interval=None) -> bool:
        interval = self.default_interval if interval is None else interval
        cache_key = (message_key, vehicle or "")
        now = time.time()
        with self.lock:
            if now - self.last_logged.get(cache_key, 0) >= interval:
                self.last_logged[cache_key] = now
                return True
            return False

    def cleanup(self, max_age: float = 300):
        now = time.time()
        with self.lock:
            for k in [k for k, v in self.last_logged.items() if now - v > max_age]:
                del self.last_logged[k]


log_throttle = LogThrottle()


def throttled_log(level: str, message: str, message_key=None, vehicle=None, interval: float = 5.0):
    if message_key is None:
        message_key = message[:30]
    if log_throttle.should_log(message_key, vehicle, interval):
        getattr(logger, level, logger.info)(message)


__all__ = ["logger", "point_logger", "camera_logger", "camera_event", "camera_info",
           "log_throttle", "throttled_log"]
