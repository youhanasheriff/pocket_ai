"""
Text-to-Speech Engine - pyttsx3 backend with console fallback.

Uses pyttsx3 for offline spoken output. Speech runs in a background thread
so the pipeline is never blocked waiting for audio to finish.

The speak() signature is stable across phases - callers never change.
"""

import logging
import threading
from queue import Queue, Empty

logger = logging.getLogger(__name__)


class TTSEngine:
    """
    Text-to-Speech interface.

    Tries pyttsx3 for real audio output; falls back to console printing
    if pyttsx3 fails to initialize (e.g. missing system TTS libs).
    """

    def __init__(self, backend: str = "pyttsx3"):
        self.backend = backend
        self._engine = None
        self._queue: Queue = Queue()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._speaking = False

        if backend == "pyttsx3":
            self._init_pyttsx3()

    def _init_pyttsx3(self) -> None:
        """Try to initialize pyttsx3 engine."""
        try:
            import pyttsx3

            self._engine = pyttsx3.init()
            # Slightly faster speech rate for quick alerts
            self._engine.setProperty("rate", 175)
            self.backend = "pyttsx3"

            # Start background worker thread
            self._thread = threading.Thread(target=self._worker, daemon=True)
            self._thread.start()

            logger.info("TTS engine initialized (pyttsx3)")
        except Exception as e:
            logger.warning(f"pyttsx3 init failed, falling back to console: {e}")
            self._engine = None
            self.backend = "console"

    def _worker(self) -> None:
        """Background thread that processes the speech queue."""
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.5)
            except Empty:
                continue

            if self._engine is None:
                continue

            try:
                self._speaking = True
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.warning(f"TTS speech failed: {e}")
            finally:
                self._speaking = False
                self._queue.task_done()

    def speak(self, text: str, priority: str = "normal") -> None:
        """
        Speak or print a text instruction.

        Args:
            text: The instruction text
            priority: "urgent" | "normal" | "info"
        """
        # Always print to console with color prefix
        prefix_map = {
            "urgent": "\033[91m[ALERT]\033[0m",   # red
            "normal": "\033[93m[INFO]\033[0m",     # yellow
            "info": "\033[94m[NOTE]\033[0m",       # blue
        }
        prefix = prefix_map.get(priority, "[INFO]")
        print(f"  {prefix} {text}")

        # Queue for audio if engine is available
        if self._engine is not None:
            # For urgent messages, clear any pending non-urgent speech
            if priority == "urgent":
                self._clear_queue()
            self._queue.put(text)

    def _clear_queue(self) -> None:
        """Drain pending items from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except Empty:
                break

    def stop(self) -> None:
        """Interrupt current speech and clear queue."""
        self._clear_queue()
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass

    def is_speaking(self) -> bool:
        """True if audio is currently playing."""
        return self._speaking or not self._queue.empty()

    def shutdown(self) -> None:
        """Clean shutdown of the background thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
