"""
Text-to-Speech Stub - Phase 1 console output.

Phase 1: Prints instructions to console with priority prefix.
Phase 1.4: Replace with pyttsx3, Kokoro-82M, or piper-TTS backend.

The speak() signature is stable across phases - callers never change.
"""


class TTSEngine:
    """
    Text-to-Speech interface.

    Phase 1: prints to console.
    Phase 1.4: plays audio via offline TTS engine.
    """

    def __init__(self, backend: str = "console"):
        self.backend = backend

    def speak(self, text: str, priority: str = "normal") -> None:
        """
        Speak or print a text instruction.

        Args:
            text: The instruction text
            priority: "urgent" | "normal" | "info"
        """
        prefix_map = {
            "urgent": "\033[91m[ALERT]\033[0m",   # red
            "normal": "\033[93m[INFO]\033[0m",     # yellow
            "info": "\033[94m[NOTE]\033[0m",       # blue
        }
        prefix = prefix_map.get(priority, "[INFO]")
        print(f"  {prefix} {text}")

    def stop(self) -> None:
        """Interrupt current speech. Phase 1: no-op."""
        pass

    def is_speaking(self) -> bool:
        """Phase 1: always False."""
        return False
