#!/bin/bash
refchecker-cli extract-check \
  --input_path example/example_in.json \
  --output_path example/example_out.json \
  --extractor_name gpt4 \
  --extractor_max_new_tokens 1000 \
  --checker_name gpt4 \
  --aggregator_name soft \
  --openai_key "<path to your key here>"
