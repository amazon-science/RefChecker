#!/bin/bash
refchecker-cli check \
  --input_path example/example_out_claims.json \
  --output_path example/example_out.json \
  --checker_name bedrock/anthropic.claude-3-sonnet-20240229-v1:0 \
  --aggregator_name soft
