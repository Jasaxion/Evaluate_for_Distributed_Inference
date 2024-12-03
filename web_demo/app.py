from flask import Flask, render_template, request, jsonify
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import sys
import time
import threading
from queue import Queue

sys.path.append('..')
from llm_bench.frameworks.baseline.huggingface_pipeline import HuggingFacePipeline

app = Flask(__name__)

# Global variables for model instances
model_instances = {}
model_locks = {}

def load_config():
    """Load benchmark configuration."""
    with open("../configs/benchmark_config.yaml", 'r') as f:
        return yaml.safe_load(f)

def initialize_models():
    """Initialize all models on startup."""
    config = load_config()
    for model_config in config['models']:
        model_name = model_config['name']
        try:
            model = HuggingFacePipeline(
                model_name=model_name,
                device="cuda",
                trust_remote_code=model_config.get('trust_remote_code', False)
            )
            model.load_model()
            model_instances[model_name] = model
            model_locks[model_name] = threading.Lock()
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")

def load_benchmark_results():
    """Load and process benchmark results."""
    results_path = Path("../results")
    frameworks = {}
    
    for result_file in results_path.glob("*.csv"):
        framework_name = result_file.stem
        df = pd.read_csv(result_file)
        frameworks[framework_name] = df
    
    return frameworks

def create_comparison_charts(frameworks):
    """Create comparison visualizations."""
    charts = {}
    # ... (之前的图表创建代码保持不变)
    return charts

@app.route('/')
def index():
    frameworks = load_benchmark_results()
    charts = create_comparison_charts(frameworks)
    
    # Add list of available models
    available_models = list(model_instances.keys())
    
    return render_template('index.html', 
                         charts=charts, 
                         models=available_models)

@app.route('/generate', methods=['POST'])
def generate():
    """Handle text generation request."""
    data = request.json
    model_name = data['model']
    prompt = data['prompt']
    max_tokens = int(data.get('max_tokens', 100))
    
    if model_name not in model_instances:
        return jsonify({
            'error': f'Model {model_name} not found'
        }), 404
    
    # Use threading lock to ensure sequential access
    with model_locks[model_name]:
        try:
            start_time = time.time()
            model = model_instances[model_name]
            result = model.generate(
                prompt=prompt,
                max_new_tokens=max_tokens
            )
            total_time = (time.time() - start_time) * 1000  # Convert to ms
            
            return jsonify({
                'output': result['output'],
                'metrics': {
                    'ttft': f"{result['ttft']:.2f} ms",
                    'tpot': f"{result['tpot']:.2f} ms",
                    'throughput': f"{result['throughput']:.2f} tokens/s",
                    'memory': f"{result['memory_used']:.2f} GB",
                    'total_time': f"{total_time:.2f} ms",
                    'num_tokens': result['num_tokens']
                }
            })
            
        except Exception as e:
            return jsonify({
                'error': str(e)
            }), 500

@app.route('/get_model_status')
def get_model_status():
    """Get the loading status of all models."""
    status = {name: True for name in model_instances.keys()}
    return jsonify(status)

if __name__ == '__main__':
    # Initialize models on startup
    initialize_models()
    app.run(debug=True, host='0.0.0.0', port=5000)