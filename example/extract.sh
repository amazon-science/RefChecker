#!/bin/bash
refchecker-cli extract \
  --input_path example/example_in.json \
  --output_path example/example_out_claims.json \
  --extractor_name claude3-sonnet \
  --extractor_max_new_tokens 1000 \
  --openai_key "<path to your key here>"
