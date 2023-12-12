#!/bin/bash
bschecker-cli check \
  --input_path example/example_out_triplets.json \
  --output_path example/example_out.json \
  --checker_name gpt4 \
  --aggregator_name soft \
  --openai_key "<path to your key here>"
