#!/bin/bash
refchecker-cli extract-check \
  --input_path example/example_in.json \
  --output_path example/example_out.json \
  --claim_format subsentence \
  --extractor_name claude3-sonnet \
  --extractor_max_new_tokens 1000 \
  --checker_name claude3-sonnet \
  --aggregator_name soft