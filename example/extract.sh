#!/bin/bash
refchecker-cli extract \
  --input_path example/example_in.json \
  --output_path example/example_out_triplets.json \
  --extractor_name gpt4 \
  --openai_key "<path to your key here>"
