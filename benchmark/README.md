## Benchmark

This folder contains the script for downloading the benchmark dataset.

### Download Benchmark Data

Please take the following steps to download the benchmark dataset.

1. Install `gsutil` with instructions here: https://cloud.google.com/storage/docs/gsutil_install

2. Run the following script
```bash
cd benchmark/data
sh download_data.sh
```

After downloading and processing, the benchmark data will be saved to the three folders under [benchmark/data](../benchmark/data)

### Collect Responses for Your Model

Please refer to the instructions in [response_collection](response_collection/README.md).

### Run the Checking Pipeline and Evaluation Script

Run the checking pipeline:
```bash
python evaluation/autocheck.py --model=<model name> --extractor=<extractor> --checker=<checker>
```

Evaluate the results:
```bash
python evaluation/evaluate.py python evaluation/evaluate.py --model=<model name> --extractor=<extractor> --checker=<checker> --output_file=<path to the output json file>
```