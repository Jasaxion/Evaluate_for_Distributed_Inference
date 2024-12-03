import json
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd

from llm_bench.core.base_framework import BaseFramework

class BenchmarkRunner:
    """Runs benchmarks for different frameworks and collects results."""
    
    def __init__(self, framework: BaseFramework, data_path: str):
        self.framework = framework
        self.data_path = data_path
        self.results = []

    def load_test_data(self) -> List[str]:
        """Load test prompts from data file."""
        with open(self.data_path, 'r') as f:
            data = json.load(f)
        return [item['prompt'] for item in data]

    def run_single_test(self, prompt: str) -> Dict[str, Any]:
        """Run a single test and collect metrics."""
        self.framework.clear_memory()
        return self.framework.generate(prompt)

    def run_benchmark(self, num_samples: int = 100) -> pd.DataFrame:
        """Run benchmark on multiple prompts and aggregate results."""
        prompts = self.load_test_data()[:num_samples]
        
        for prompt in prompts:
            result = self.run_single_test(prompt)
            self.results.append(result)

        # Aggregate results
        df = pd.DataFrame(self.results)
        aggregated = {
            'throughput_mean': df['throughput'].mean(),
            'throughput_std': df['throughput'].std(),
            'memory_used_max': df['memory_used'].max(),
            'ttft_mean': df['ttft'].mean(),
            'ttft_std': df['ttft'].std(),
            'tpot_mean': df['tpot'].mean(),
            'tpot_std': df['tpot'].std()
        }
        
        return pd.DataFrame([aggregated])

    def save_results(self, output_path: str):
        """Save benchmark results to file."""
        df = pd.DataFrame(self.results)
        df.to_csv(output_path, index=False)