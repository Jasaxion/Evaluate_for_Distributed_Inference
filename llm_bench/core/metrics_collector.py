from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path
import json

@dataclass
class GenerationMetrics:
    """Dataclass for storing generation metrics."""
    ttft: float  # Time to first token (ms)
    tpot: float  # Time per output token (ms)
    throughput: float  # Tokens per second
    memory_used: float  # GPU memory used (GB)
    total_time: float  # Total generation time (ms)
    num_tokens: int  # Number of tokens generated

class MetricsCollector:
    """Collect and analyze benchmark metrics."""
    
    def __init__(self, framework_name: str, model_name: str):
        self.framework_name = framework_name
        self.model_name = model_name
        self.metrics: List[GenerationMetrics] = []
    
    def add_metrics(self, metrics_dict: Dict[str, Any]) -> None:
        """Add metrics from a single generation run."""
        metrics = GenerationMetrics(
            ttft=metrics_dict['ttft'],
            tpot=metrics_dict['tpot'],
            throughput=metrics_dict['throughput'],
            memory_used=metrics_dict['memory_used'],
            total_time=metrics_dict['total_time'],
            num_tokens=metrics_dict['num_tokens']
        )
        self.metrics.append(metrics)
    
    def compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """Compute statistics for collected metrics."""
        if not self.metrics:
            raise ValueError("No metrics collected yet")
        
        # Convert metrics to numpy arrays
        metrics_dict = {
            field: np.array([getattr(m, field) for m in self.metrics])
            for field in GenerationMetrics.__dataclass_fields__
        }
        
        # Compute statistics for each metric
        stats = {}
        for field, values in metrics_dict.items():
            stats[field] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'p95': float(np.percentile(values, 95))
            }
        
        return stats
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        return pd.DataFrame([asdict(m) for m in self.metrics])
    
    def save_results(self, output_dir: Path) -> None:
        """Save metrics and statistics to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw metrics
        df = self.to_dataframe()
        df.to_csv(output_dir / f"{self.framework_name}_{self.model_name}_metrics.csv", index=False)
        
        # Save statistics
        stats = self.compute_statistics()
        with open(output_dir / f"{self.framework_name}_{self.model_name}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the metrics."""
        stats = self.compute_statistics()
        summary = f"Performance Summary for {self.framework_name} ({self.model_name})\n"
        summary += "=" * 50 + "\n\n"
        
        metrics_names = {
            'ttft': 'Time to First Token (ms)',
            'tpot': 'Time per Token (ms)',
            'throughput': 'Throughput (tokens/s)',
            'memory_used': 'Memory Usage (GB)',
            'total_time': 'Total Time (ms)',
            'num_tokens': 'Tokens Generated'
        }
        
        for metric, name in metrics_names.items():
            metric_stats = stats[metric]
            summary += f"{name}:\n"
            summary += f"  Mean ± Std: {metric_stats['mean']:.2f} ± {metric_stats['std']:.2f}\n"
            summary += f"  Range: [{metric_stats['min']:.2f}, {metric_stats['max']:.2f}]\n"
            summary += f"  95th Percentile: {metric_stats['p95']:.2f}\n\n"
        
        return summary