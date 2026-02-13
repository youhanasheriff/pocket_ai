"""
Text-to-Speech Engine - macOS `say` backend with console fallback.

Uses the native macOS `say` command for reliable offline speech.
Runs each utterance as a subprocess so the pipeline is never blocked.

Design: "latest wins" single-slot. While speech is playing, new calls
to speak() just update the pending slot. When the current utterance
finishes, the worker picks up whatever is in the slot (latest only).

The speak() signature is stable across phases - callers never change.
"""

import logging
import platform
import subprocess
import threading

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech interface.

    Uses macOS `say` command for reliable non-blocking speech.
    Falls back to console-only on non-macOS platforms.
    """

    def __init__(self, backend: str = "auto"):
        self.backend = "console"
        self._process: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Single-slot: only the latest instruction matters
        self._pending_text: str | None = None
        self._current_text: str = ""

        if backend == "auto" or backend == "say":
            self._init_say()

    def _init_say(self) -> None:
        """Check if macOS `say` command is available."""
        if platform.system() != "Darwin":
            logger.info("Not macOS, TTS falling back to console")
            return

        try:
            subprocess.run(
                ["say", "--version"],
                capture_output=True,
                timeout=2,
            )
            self.backend = "say"

            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

            logger.info("TTS engine initialized (macOS say)")
        except Exception as e:
            logger.warning(f"macOS say not available: {e}")

    def _worker(self) -> None:
        """Background thread: speaks the latest pending text, one at a time."""
        while not self._stop_event.is_set():
            with self._lock:
                text = self._pending_text
                self._pending_text = None

            if text is None:
                self._stop_event.wait(timeout=0.15)
                continue

            self._current_text = text
            try:
                # `say` blocks until speech finishes -- no cutoff
                self._process = subprocess.Popen(
                    ["say", "-r", "190", text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._process.wait()
            except Exception as e:
                logger.warning(f"TTS speech failed: {e}")
            finally:
                self._process = None
                self._current_text = ""

    def speak(self, text: str, priority: str = "normal") -> None:
        """
        Speak or print a text instruction.

        Console output is always printed. For audio, the text is placed in a
        single slot. If the engine is busy, only the most recent call survives
        -- older pending text is silently replaced.

        Args:
            text: The instruction text
            priority: "urgent" | "normal" | "info"
        """
        prefix_map = {
            "urgent": "\033[91m[ALERT]\033[0m",
            "normal": "\033[93m[INFO]\033[0m",
            "info": "\033[94m[NOTE]\033[0m",
        }
        prefix = prefix_map.get(priority, "[INFO]")
        print(f"  {prefix} {text}")

        if self.backend != "say":
            return

        with self._lock:
            # Skip if already speaking the exact same text
            if self._current_text == text:
                return
            # Overwrite any stale pending text -- latest wins
            self._pending_text = text

    def stop(self) -> None:
        """Kill current speech immediately."""
        with self._lock:
            self._pending_text = None
        proc = self._process
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        """True if audio is currently playing or pending."""
        return self._process is not None or self._pending_text is not None

    def shutdown(self) -> None:
        """Clean shutdown."""
        self._stop_event.set()
        self.stop()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
