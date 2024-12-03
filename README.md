# Evaluate_for_Distributed_Inference
The final project for the course "Big Data Fundamentals B" focuses on exploring the performance of distributed inference frameworks. 大数据基础 B 的结课大作业，着眼于探索分布式推理框架的性能

# 使用方法
1. 安装依赖: pip install -r requirements.txt

2. 运行基准测试 `python -m llm_bench.run_benchmark --config configs/benchmark_config.yaml`

3. 启动Web演示
```
cd web_demo
flask run
```

# 如何新增模型？
1. 在frameworks/目录下创建新的子目录

2. 实现相应的Framework类，继承BaseFramework

3. 在配置文件中添加新的框架配置