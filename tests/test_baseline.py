import pytest
import torch
from pathlib import Path
import sys
import yaml

# Add parent directory to path to import llm_bench
sys.path.append(str(Path(__file__).parent.parent))

from llm_bench.frameworks.baseline.huggingface_pipeline import HuggingFacePipeline
from llm_bench.utils.timer import Timer

@pytest.fixture
def config():
    """Load test configuration."""
    config_path = Path(__file__).parent.parent / "configs" / "benchmark_config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def model_name():
    """Return test model name."""
    return "THUDM/chatglm2-6b"  # Using smallest model for testing

@pytest.fixture
def baseline_framework(model_name):
    """Initialize baseline framework."""
    framework = HuggingFacePipeline(
        model_name=model_name,
        device="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True
    )
    return framework

def test_framework_initialization(baseline_framework):
    """Test framework initialization."""
    assert baseline_framework is not None
    assert baseline_framework.model_name == "THUDM/chatglm2-6b"
    assert baseline_framework.device in ["cuda", "cpu"]

def test_model_loading(baseline_framework):
    """Test model loading."""
    baseline_framework.load_model()
    assert baseline_framework.model is not None
    assert baseline_framework.tokenizer is not None

def test_text_generation(baseline_framework):
    """Test text generation."""
    baseline_framework.load_model()
    
    # Test with a simple prompt
    prompt = "Hello, how are you?"
    result = baseline_framework.generate(prompt, max_new_tokens=20)
    
    # Check result structure
    assert isinstance(result, dict)
    assert all(key in result for key in [
        'output', 'ttft', 'tpot', 'memory_used', 'throughput', 'num_tokens'
    ])
    
    # Check metric values
    assert result['ttft'] > 0  # TTFT should be positive
    assert result['tpot'] > 0  # TPOT should be positive
    assert result['memory_used'] >= 0  # Memory usage should be non-negative
    assert result['throughput'] > 0  # Throughput should be positive
    assert result['num_tokens'] > 0  # Should generate at least one token
    assert isinstance(result['output'], str)  # Output should be string

def test_memory_tracking(baseline_framework):
    """Test memory usage tracking."""
    if torch.cuda.is_available():
        baseline_framework.load_model()
        initial_memory = baseline_framework.get_memory_usage()
        
        # Generate text to use memory
        baseline_framework.generate("Test prompt", max_new_tokens=50)
        
        peak_memory = baseline_framework.get_memory_usage()
        assert peak_memory >= initial_memory
        
        # Test memory clearing
        baseline_framework.clear_memory()
        cleared_memory = baseline_framework.get_memory_usage()
        assert cleared_memory <= peak_memory

def test_timer_integration(baseline_framework):
    """Test timer integration."""
    timer = Timer()
    
    # Test basic timer functionality
    timer.start()
    result = timer.stop()
    assert result > 0
    
    # Test with generation
    baseline_framework.load_model()
    timer.start()
    baseline_framework.generate("Quick test.", max_new_tokens=10)
    total_time = timer.stop()
    assert total_time > 0

@pytest.mark.parametrize("max_tokens", [10, 50, 100])
def test_different_generation_lengths(baseline_framework, max_tokens):
    """Test generation with different lengths."""
    baseline_framework.load_model()
    result = baseline_framework.generate("Test prompt", max_new_tokens=max_tokens)
    assert result['num_tokens'] <= max_tokens + 1  # +1 for potential special tokens

if __name__ == "__main__":
    pytest.main([__file__])