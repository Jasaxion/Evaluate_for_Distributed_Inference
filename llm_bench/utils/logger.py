import logging
import sys
from pathlib import Path
from typing import Optional

class BenchLogger:
    """Custom logger for the benchmark framework."""
    
    def __init__(self, name: str, log_file: Optional[Path] = None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (optional)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def info(self, msg: str) -> None:
        """Log info message."""
        self.logger.info(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message."""
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message."""
        self.logger.error(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message."""
        self.logger.debug(msg)

    def benchmark_start(self, framework: str, model: str) -> None:
        """Log benchmark start."""
        self.info(f"Starting benchmark for {framework} with model {model}")
    
    def benchmark_complete(self, framework: str, model: str) -> None:
        """Log benchmark completion."""
        self.info(f"Benchmark complete for {framework} with model {model}")
    
    def generation_metrics(self, metrics: dict) -> None:
        """Log generation metrics."""
        msg = "Generation metrics:\n"
        for key, value in metrics.items():
            msg += f"  {key}: {value}\n"
        self.info(msg)

def get_logger(name: str, log_file: Optional[Path] = None) -> BenchLogger:
    """Get a logger instance."""
    return BenchLogger(name, log_file)