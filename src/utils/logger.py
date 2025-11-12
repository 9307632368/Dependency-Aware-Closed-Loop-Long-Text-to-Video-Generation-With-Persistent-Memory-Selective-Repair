# src/utils/logger.py
import os
from datetime import datetime
from typing import Optional


class SimpleLogger:
    """
    Lightweight logger for console + optional file logging.
    """

    def __init__(self, log_file: Optional[str] = None, also_print: bool = True):
        self.log_file = log_file
        self.also_print = also_print

        if self.log_file:
            parent = os.path.dirname(self.log_file)
            if parent:
                os.makedirs(parent, exist_ok=True)

    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write(self, level: str, message: str) -> None:
        line = f"[{self._timestamp()}] [{level.upper()}] {message}"

        if self.also_print:
            print(line)

        if self.log_file:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def info(self, message: str) -> None:
        self._write("INFO", message)

    def warning(self, message: str) -> None:
        self._write("WARNING", message)

    def error(self, message: str) -> None:
        self._write("ERROR", message)

    def debug(self, message: str) -> None:
        self._write("DEBUG", message)

    def exception(self, message: str) -> None:
        self._write("EXCEPTION", message)