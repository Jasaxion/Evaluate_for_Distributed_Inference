import argparse
import yaml
from pathlib import Path
from typing import Dict, Any

import torch
from llm_bench.frameworks.baseline.huggingface_pipeline import HuggingFacePipeline
from llm_bench.core.benchmark_runner import BenchmarkRunner
from llm_bench.core.metrics_collector import MetricsCollector

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_framework(model_config: Dict[str, Any], hardware_config: Dict[str, Any]) -> HuggingFacePipeline:
    """Initialize the framework with model configuration."""
    framework = HuggingFacePipeline(
        model_name=model_config['name'],
        device=hardware_config['device'],
        trust_remote_code=model_config.get('trust_remote_code', False),
        chat_template=model_config.get('chat_template', False)
    )
    framework.load_model()
    return framework

def run_benchmarks(config: Dict[str, Any]) -> None:
    """Run benchmarks for all configured models."""
    # Create base output directory
    results_dir = Path(config['output_config']['results_dir'])
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Run benchmarks for each model
    for model_config in config['models']:
        print(f"\nRunning benchmarks for {model_config['name']}")
        model_name = model_config['name'].replace('/', '_')
        
        # Initialize framework with generation config
        framework = setup_framework(model_config, config['hardware_config'])
        
        # Initialize benchmark runner
        runner = BenchmarkRunner(
            framework=framework,
            data_path=config['data_config']['dataset_path']
        )
        
        try:
            # Run benchmark and get aggregated results
            print("\nRunning benchmark tests...")
            results_df = runner.run_benchmark(
                num_samples=config['benchmark_config']['num_samples']
            )
            
            # Create output directory and save results
            model_results_dir = results_dir / f"{model_config['framework']}_{model_name}"
            model_results_dir.mkdir(parents=True, exist_ok=True)
            
            # Save raw results
            raw_results_path = model_results_dir / "raw_results.csv"
            runner.save_results(str(raw_results_path))
            
            # Save aggregated results
            agg_results_path = model_results_dir / "aggregated_results.csv"
            results_df.to_csv(agg_results_path, index=False)
            
            # Print summary
            print("\nBenchmark Results:")
            print("-" * 50)
            print(f"Model: {model_config['name']}")
            print(f"Framework: {model_config['framework']}")
            print("-" * 50)
            print("Aggregated Metrics:")
            for column in results_df.columns:
                value = results_df[column].iloc[0]
                print(f"{column}: {value:.4f}")
                
        except Exception as e:
            print(f"Error running benchmark for {model_config['name']}: {str(e)}")
            continue
            
        finally:
            # Clear GPU memory
            print("\nClearing GPU memory...")
            framework.clear_memory()
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description='Run LLM benchmarks')
    parser.add_argument('--config', type=str, required=True, help='Path to benchmark configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Run benchmarks
    run_benchmarks(config)

if __name__ == "__main__":
    main()