mkdir zero_context/nq
gsutil -m cp -R gs://natural_questions/v1.0/dev zero_context/nq
gzip -d zero_context/nq/dev/nq-dev-00.jsonl.gz
gzip -d zero_context/nq/dev/nq-dev-01.jsonl.gz
gzip -d zero_context/nq/dev/nq-dev-02.jsonl.gz
gzip -d zero_context/nq/dev/nq-dev-03.jsonl.gz
gzip -d zero_context/nq/dev/nq-dev-04.jsonl.gz

python gather_benchmark_data.py --dataset=nq
python gather_benchmark_data.py --dataset=msmarco
python gather_benchmark_data.py --dataset=dolly

rm -r zero_context/nq
