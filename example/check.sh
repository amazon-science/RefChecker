#!/bin/bash
refchecker-cli check \
  --input_path example/example_out_claims.json \
  --output_path example/example_out.json \
  --checker_name claude3-sonnet \
  --aggregator_name soft \
  --openai_key "<path to your key here>"
