import time
from typing import Dict, Any, Optional
import torch
from transformers import AutoModel, AutoTokenizer, TextGenerationPipeline
import torch.nn as nn

from llm_bench.core.base_framework import BaseFramework
from llm_bench.utils.timer import Timer

class HuggingFacePipeline(BaseFramework):
    """Baseline implementation using HuggingFace Transformers."""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(model_name, device)
        self.trust_remote_code = kwargs.get('trust_remote_code', False)
        self.chat_template = kwargs.get('chat_template', False)

        self._preferred_device = device
        
    def load_model(self) -> None:
        """Load model and tokenizer using transformers."""
        # Load tokenizer with trust_remote_code for ChatGLM models
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=self.trust_remote_code
        )
        if "chatglm" in self.model_name.lower():
            self.tokenizer.eos_token = "</s>"
        
        # Configure model loading parameters
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16,
            "trust_remote_code": self.trust_remote_code
        }
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load the model
        self.model = AutoModel.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Enable gradient checkpointing if available
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()
        
        # Create pipeline
        self.pipeline = TextGenerationPipeline(
            model=self.model,
            tokenizer=self.tokenizer
            # device=self.device
        )
    
    def get_actual_device(self) -> torch.device:
        """Get the actual device where the model's first parameter resides."""
        try:
            return next(self.model.parameters()).device
        except StopIteration:
            return torch.device(self._preferred_device)

    def format_prompt(self, prompt: str) -> str:
        """Format prompt according to model's requirements."""
        if not self.chat_template:
            return prompt
            
        # Model specific chat templates
        if "chatglm" in self.model_name.lower():
            return f"[Round 1]\n\n问：{prompt}\n\n答："
        elif "llama-2" in self.model_name.lower():
            return f"[INST] {prompt} [/INST]"
        elif "mistral" in self.model_name.lower():
            return f"<s>[INST] {prompt} [/INST]"
        
        return prompt

    def generate(self, prompt: str, max_new_tokens: int = 100, **kwargs) -> Dict[str, Any]:
        """Generate text and collect metrics."""
        timer = Timer()
        formatted_prompt = self.format_prompt(prompt)
        
        # Tokenize input
        actual_device = self.get_actual_device()
        input_ids = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(actual_device)
        
        # Track memory before generation
        initial_memory = self.get_memory_usage()
        
        # Generation with timing
        timer.start()
        with torch.no_grad():
            # First token generation (for TTFT)
            first_token_output = self.model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            ttft = timer.lap()
            
            # Complete generation
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=kwargs.get('temperature', 0.7),
                top_p=kwargs.get('top_p', 0.9),
                pad_token_id=self.tokenizer.pad_token_id,
                **kwargs
            )
        total_time = timer.stop()
        
        # Calculate metrics
        if "chatglm" in self.model_name.lower():
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 移除可能的前缀
            output_text = output_text.replace("[gMASK] sop", "").strip()
        else:
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        num_new_tokens = len(outputs[0]) - len(input_ids[0])
        
        peak_memory = self.get_memory_usage()
        memory_used = peak_memory - initial_memory
        
        # Calculate TPOT (excluding first token time)
        remaining_time = total_time - ttft
        remaining_tokens = num_new_tokens - 1
        tpot = remaining_time / remaining_tokens if remaining_tokens > 0 else 0
        
        return {
            'output': output_text,
            'ttft': ttft * 1000,  # Convert to ms
            'tpot': tpot * 1000,  # Convert to ms
            'memory_used': memory_used,
            'throughput': num_new_tokens / total_time,
            'num_tokens': num_new_tokens,
            'total_time': total_time
        }

    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024**3
        return 0.0

    def clear_memory(self) -> None:
        """Clear GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # if hasattr(self.model, "cpu"):
            #     with torch.no_grad():
            #         self.model.cpu()
            # torch.cuda.empty_cache()
            # if hasattr(self.model, "to"):
            #     with torch.no_grad():
            #         self.model.to(self.device)