from __future__ import annotations

from datetime import datetime
from typing import Any
import traceback


class SimpleLogger:
    """Minimal logger that prints messages to stdout.

    Mirrors a subset of the standard logging.Logger API used in this project.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def _emit(self, level: str, message: str, extra: dict[str, Any] | None, exc_info: Any) -> None:
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        extra_suffix = f" | extra={extra}" if extra else ""
        print(f"[{timestamp}] {level} {self.name}: {message}{extra_suffix}")

        if exc_info:
            if exc_info is True:
                traceback.print_exc()
            elif isinstance(exc_info, BaseException):
                traceback.print_exception(exc_info.__class__, exc_info, exc_info.__traceback__)
            else:
                print(exc_info)

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        if args:
            message = message % args
        self._emit("DEBUG", message, extra, exc_info)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        if args:
            message = message % args
        self._emit("INFO", message, extra, exc_info)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        if args:
            message = message % args
        self._emit("WARNING", message, extra, exc_info)

    def error(self, message: str, *args: Any, **kwargs: Any) -> None:
        extra = kwargs.pop("extra", None)
        exc_info = kwargs.pop("exc_info", None)
        if args:
            message = message % args
        self._emit("ERROR", message, extra, exc_info)

    def exception(self, message: str, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault("exc_info", True)
        self.error(message, *args, **kwargs)


def get_logger(name: str) -> SimpleLogger:
    """Factory matching logging.getLogger signature."""
    return SimpleLogger(name)


__all__ = ["SimpleLogger", "get_logger"]

