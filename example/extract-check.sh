#!/bin/bash
refchecker-cli extract-check \
  --input_path example/example_in.json \
  --output_path example/example_out.json \
  --extractor_name claude3 \
  --extractor_max_new_tokens 1000 \
  --checker_name claude3 \
  --aggregator_name soft