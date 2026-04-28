import logging
import os
import sys
import ctypes
import shutil
import warnings
from datetime import datetime


_STAGE_PREFIX_MAP = (
    ("realization", "stage3_realization"),
    ("reasoning", "stage2_reasoning"),
    ("perception", "stage1_perception"),
)


class _StageNameFilter(logging.Filter):
    """
    Normalize logger names so that logs from any perception/reasoning/realization
    module appear as the corresponding stage logger in the terminal output.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name or ""
        if name in {"stage1_perception", "stage2_reasoning", "stage3_realization"}:
            return True
        for needle, stage_name in _STAGE_PREFIX_MAP:
            if needle in name:
                record.name = stage_name
                break
        return True


class _ColorFormatter(logging.Formatter):
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m\033[97m",
    }
    LEVEL_BADGES = {
        "DEBUG": "DEBUG",
        "INFO": "INFO ",
        "WARNING": "WARN ",
        "ERROR": "ERROR",
        "CRITICAL": "CRIT ",
    }

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S.%f")[:-3]
        lvl = record.levelname
        name = record.name
        process_name = getattr(record, "processName", "MainProcess")
        process_id = getattr(record, "process", 0)
        msg = record.getMessage()
        badge = self.LEVEL_BADGES.get(lvl, lvl[:5].ljust(5))
        proc = f"{process_name}:{process_id}"
        if self.use_color:
            lvl_col = self.COLORS.get(lvl, "")
            return (
                f"{self.DIM}{ts}{self.RESET} "
                f"{self.BLUE}[{name}]{self.RESET} "
                f"{self.DIM}({proc}){self.RESET} "
                f"{lvl_col}{self.BOLD}{badge}{self.RESET} "
                f"{self.MAGENTA}|{self.RESET} {msg}"
            )
        return f"{ts} [{name}] ({proc}) {badge} | {msg}"


def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        # Best-effort only; fallback is non-colored logs.
        return


def _supports_color() -> bool:
    force_color = str(
        os.getenv("FORCE_COLOR")
        or os.getenv("CLICOLOR_FORCE")
        or os.getenv("PY_COLORS")
        or ""
    ).strip().lower()
    if force_color in {"1", "true", "yes", "on"}:
        return True
    if os.getenv("NO_COLOR"):
        return False
    if os.getenv("TERM", "").lower() == "dumb":
        return False
    if os.getenv("CI", "").strip():
        return True
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def print_startup_logo() -> None:
    width = max(64, int(shutil.get_terminal_size(fallback=(100, 24)).columns))
    inner = width - 2
    top_bottom = "=" * width
    title = "IMAGE TRANSCREATION PIPELINE".center(inner)
    subtitle = "Perception -> Reasoning -> Realization".center(inner)

    if _supports_color():
        c1 = _ColorFormatter.CYAN + _ColorFormatter.BOLD
        c2 = _ColorFormatter.BLUE + _ColorFormatter.BOLD
        reset = _ColorFormatter.RESET
        print(f"{c1}{top_bottom}{reset}")
        print(f"{c2}|{title}|{reset}")
        print(f"{c2}|{subtitle}|{reset}")
        print(f"{c1}{top_bottom}{reset}")
        return

    print(top_bottom)
    print(f"|{title}|")
    print(f"|{subtitle}|")
    print(top_bottom)


_TARGET_STAGE_LOGGERS = (
    "pipeline_main",
    "stage1_perception",
    "stage2_reasoning",
    "stage3_realization",
    # Parent loggers for module hierarchies that emit stage-specific logs
    # (e.g. src.reasoning.engine, src.realization.engine). Handler + filter
    # will rewrite their record name to the corresponding stage logger.
    "src.perception",
    "src.reasoning",
    "src.realization",
)


def configure_terminal_logger(level: str = "INFO") -> None:
    _enable_windows_ansi()
    log_level = getattr(logging, str(level).upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    handler.setFormatter(_ColorFormatter(use_color=_supports_color()))
    handler.addFilter(_StageNameFilter())

    root = logging.getLogger()
    root.setLevel(log_level)
    root.handlers = []
    root.addHandler(handler)

    # Attach the SAME handler directly to each stage logger and disable
    # propagation. This guarantees our pipeline log lines reach the terminal
    # even if a third-party library (e.g. paddleocr) mutates the root logger
    # level or replaces its handlers during import.
    for lg_name in _TARGET_STAGE_LOGGERS:
        stage_logger = logging.getLogger(lg_name)
        stage_logger.setLevel(log_level)
        stage_logger.propagate = False
        stage_logger.handlers = [handler]

    _configure_third_party_loggers()


def _configure_third_party_loggers() -> None:
    """
    Keep terminal output readable by reducing noisy third-party debug logs.
    """
    for name, level in {
        "ppocr": logging.WARNING,
        "paddleocr": logging.WARNING,
        "transformers": logging.ERROR,
    }.items():
        ext_logger = logging.getLogger(name)
        ext_logger.setLevel(level)
    warnings.filterwarnings(
        "ignore",
        message=r"`text_config_dict` is provided which will be used to initialize `CLIPTextConfig`\..*",
    )

