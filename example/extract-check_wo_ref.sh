#!/bin/bash
# using google retriever requires access to claude2 for query generation
refchecker-cli extract-check \
  --input_path example/example_in.json \
  --output_path example/example_out.json \
  --extractor_name gpt4 \
  --extractor_max_new_tokens 1000 \
  --checker_name gpt4 \
  --aggregator_name soft \
  --openai_key "<path to your key file>" \
  --use_retrieval \
  --serper_api_key "<path to your key file>"
