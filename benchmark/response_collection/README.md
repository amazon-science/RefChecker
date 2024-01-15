## Collection Responses of Your LLM on RefChecker Benchmark

Please take the following steps to collect responses of your LLM on the RefChecker benchmark data.

1. Write a class inherit from `ResponseCollectorBase` in [collector_base.py](collector_base.py). Please check the examples in [gpt4_turbo.py](gpt4_turbo.py), [chatglm3_6b.py](chatglm3_6b.py) and [mistral.py](mistral.py).


2. Modify [main.py](main.py) to add your model there.

3. Run the following command to collect responses from your model:

```bash
cd benchmark
python response_collection/main.py --model=<model name>
```

The file containing responses will be saved to folds of different settings in [benchmark/data](../../benchmark/data/).