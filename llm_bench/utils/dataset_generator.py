import json
from typing import List, Dict
import random

class BenchmarkDatasetGenerator:
    """Generate benchmark dataset with various prompt types."""
    
    def __init__(self):
        self.templates = {
            'short': [
                "简单总结一下{topic}的主要内容。",
                "请简短描述{topic}是什么？",
                "用一句话解释{topic}。",
            ],
            'medium': [
                "详细分析{topic}的优点和缺点。",
                "请解释{topic}的工作原理和应用场景。",
                "比较{topic}和{alternative}的区别。",
            ],
            'long': [
                "请全面讨论{topic}在当前领域的发展状况、面临的挑战以及未来的发展方向。",
                "从技术、应用和影响三个方面深入分析{topic}。",
                "详细介绍{topic}的历史发展、现状和未来趋势。",
            ]
        }
        
        self.topics = [
            "人工智能", "机器学习", "深度学习", "自然语言处理",
            "计算机视觉", "强化学习", "神经网络", "大语言模型",
            "迁移学习", "联邦学习", "知识图谱", "注意力机制"
        ]
        
        self.alternatives = {
            "人工智能": "传统编程",
            "机器学习": "规则系统",
            "深度学习": "传统机器学习",
            "自然语言处理": "规则解析",
            "计算机视觉": "传统图像处理",
            "强化学习": "监督学习",
            "神经网络": "统计模型",
            "大语言模型": "传统NLP模型",
            "迁移学习": "从头训练",
            "联邦学习": "中心化学习",
            "知识图谱": "关系数据库",
            "注意力机制": "RNN结构"
        }

    def generate_prompt(self, prompt_type: str) -> str:
        """Generate a single prompt of specified type."""
        template = random.choice(self.templates[prompt_type])
        topic = random.choice(self.topics)
        
        if "{alternative}" in template:
            alternative = self.alternatives[topic]
            return template.format(topic=topic, alternative=alternative)
        return template.format(topic=topic)

    def generate_dataset(self, 
                        output_path: str,
                        num_short: int = 200,
                        num_medium: int = 200,
                        num_long: int = 100) -> None:
        """Generate and save benchmark dataset."""
        dataset = []
        
        # Generate prompts of different lengths
        for _ in range(num_short):
            dataset.append({
                "type": "short",
                "prompt": self.generate_prompt("short")
            })
            
        for _ in range(num_medium):
            dataset.append({
                "type": "medium",
                "prompt": self.generate_prompt("medium")
            })
            
        for _ in range(num_long):
            dataset.append({
                "type": "long",
                "prompt": self.generate_prompt("long")
            })
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)

def create_benchmark_dataset(output_path: str):
    """Convenience function to create benchmark dataset."""
    generator = BenchmarkDatasetGenerator()
    generator.generate_dataset(output_path)

if __name__ == "__main__":
    create_benchmark_dataset("data/sample_conversations.json")