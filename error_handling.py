import time
from functools import wraps
from typing import Iterable, Type, Callable, List, Dict, Any


class DataSourceError(RuntimeError):
    """Raised when an external data source cannot be reached or parsed."""


class DataValidationError(ValueError):
    """Raised when a record or dataset violates validation rules."""


class SchemaError(DataValidationError):
    """Raised when a dataset schema is missing required fields."""


class ErrorCollector:
    """Collect validation issues from row-by-row checks."""

    def __init__(self):
        self._errors: List[Dict[str, Any]] = []

    def add(self, key: Any, error: Exception) -> None:
        self._errors.append({"key": key, "error": str(error)})

    def has_errors(self) -> bool:
        return bool(self._errors)

    def count(self) -> int:
        return len(self._errors)

    def as_dicts(self) -> List[Dict[str, Any]]:
        return [dict(item) for item in self._errors]

    def log_summary(self, logger) -> None:
        logger.warning("Collected %s validation errors.", len(self._errors))


def safe_run(retries: int = 3, delay_seconds: float = 0.0, exceptions: Iterable[Type[BaseException]] = (Exception,)):
    """Retry a function a fixed number of times for tolerated exception types."""

    allowed = tuple(exceptions)
    if retries < 1:
        retries = 1

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(retries + 1):
                try:
                    return func(*args, **kwargs)
                except allowed as exc:
                    last_exc = exc
                    if attempt >= retries:
                        raise
                    if delay_seconds:
                        time.sleep(delay_seconds)
            if last_exc is not None:
                raise last_exc
            raise RuntimeError("safe_run exhausted without an exception")

        return wrapper

    return decorator
