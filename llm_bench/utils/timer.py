import time
from typing import Optional

class Timer:
    """A simple timer class for measuring execution time."""
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.last_lap: Optional[float] = None
        self.is_running = False

    def start(self) -> None:
        """Start the timer."""
        if not self.is_running:
            self.start_time = self.last_lap = time.perf_counter()
            self.is_running = True

    def lap(self) -> float:
        """Return time since start or last lap without stopping the timer."""
        if not self.is_running:
            raise RuntimeError("Timer is not running")
        
        current_time = time.perf_counter()
        elapsed = current_time - self.last_lap
        self.last_lap = current_time
        return elapsed

    def stop(self) -> float:
        """Stop the timer and return the total elapsed time."""
        if not self.is_running:
            raise RuntimeError("Timer is not running")
        
        elapsed = time.perf_counter() - self.start_time
        self.is_running = False
        self.start_time = None
        self.last_lap = None
        return elapsed

    def reset(self) -> None:
        """Reset the timer."""
        self.start_time = None
        self.last_lap = None
        self.is_running = False