import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from tqdm import tqdm

from .extractor import (
    Claude2Extractor, GPT4Extractor, MistralExtractor, MixtralExtractor
)
from .checker import (
    Claude2Checker, GPT4Checker, NLIChecker, AlignScoreChecker, RepCChecker
)
from .retriever import GoogleRetriever
from .aggregator import strict_agg, soft_agg, major_agg


def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "mode", nargs="?", choices=["extract", "check", "extract-check"],
        help="extract:       Extract triplets from provided responses.\n"
             "check:         Check whether the provided triplets are factual.\n"
             "extract-check: Extract triplets and check whether they are factual."
    )
    parser.add_argument(
        "--input_path", type=str, required=True,
        help="Input path to the json file."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Output path to the result json file."
    )
    parser.add_argument(
        "--cache_dir", type=str, default="./.cache",
        help="Path to the cache directory. Default: ./.cache"
    )
    parser.add_argument(
        '--extractor_name', type=str, default="claude2",
        choices=["gpt4", "claude2", "mistral", "mixtral"],
        help="Model used for extracting triplets. Default: claude2."
    )
    parser.add_argument(
        '--extractor_max_new_tokens', type=int, default=500,
        help="Max generated tokens of the extractor, set a larger value for longer documents. Default: 500"
    )
    parser.add_argument(
        "--checker_name", type=str, default="claude2",
        choices=["gpt4", "claude2", "nli", "alignscore", "repc"],
        help="Model used for checking whether the triplets are factual. "
        "Default: claude2."
    )
    parser.add_argument(
        "--retriever_name", type=str, default="google", choices=["google"],
        help="Model used for retrieving reference (currently only google is"
        " supported). Default: google."
    )
    parser.add_argument(
        "--aggregator_name", type=str, default="soft",
        choices=["strict", "soft", "major"],
        help="Aggregator used for aggregating the results from multiple "
             "triplets. Default: soft.\n"
             "*  strict: If any of the triplets is Contradiction, the response"
             " is Contradiction.\nIf all of the triplets are Entailment, the "
             "response is Entailment. Otherwise, the\nresponse is Neutral.\n"
             "*  soft:   The ratio of each category is calculated.\n"
             "*  major:  The category with the most votes is selected."
    )
    parser.add_argument(
        "--openai_key", type=str, default="",
        help="Path to the openai api key file. Required if openAI models are"
        " used."
    )
    parser.add_argument(
        "--anthropic_key", type=str, default="",
        help="Path to the Anthropic api key file. Required if the Anthropic "
        "Claude2 api is used."
    )
    parser.add_argument(
        "--aws_bedrock_region", type=str, default="",
        help="AWS region where the Amazon Bedrock api is deployed. Required if "
        "the Amazon Bedrock api is used."
    )
    parser.add_argument(
        "--use_retrieval", action="store_true",
        help="Whether to use retrieval to find the reference for checking. "
        "Required if the reference\nfield in input data is not provided."
    )
    parser.add_argument(
        "--serper_api_key", type=str, default="",
        help="Path to the serper api key file. Required if the google retriever"
        " is used."
    )

    return parser.parse_args()


def main():
    args = get_args()
    # set environment variables
    if args.openai_key:
        with open(args.openai_key, "r") as fp:
            os.environ["OPENAI_API_KEY"] = fp.read().strip()
    if args.anthropic_key:
        with open(args.anthropic_key, "r") as fp:
            os.environ["ANTHROPIC_API_KEY"] = fp.read().strip()
    if args.aws_bedrock_region:
        os.environ["aws_bedrock_region"] = args.aws_bedrock_region
    if args.serper_api_key:
        os.environ["SERPER_API_KEY"] = args.serper_api_key

    if args.mode == "extract":
        extract(args)
    elif args.mode == "check":
        check(args)
    elif args.mode == "extract-check":
        output_path = args.output_path
        args.output_path = output_path + ".temp"
        extract(args)
        args.input_path = args.output_path
        args.output_path = output_path
        check(args)
    else:
        raise NotImplementedError


def extract(args):
    # initialize models
    if args.extractor_name == "claude2":
        extractor = Claude2Extractor()
    elif args.extractor_name == "gpt4":
        extractor = GPT4Extractor()
    elif args.extractor_name == "mixtral":
        extractor = MixtralExtractor()
    elif args.extractor_name == "mistral":
        extractor = MistralExtractor()
    else:
        raise NotImplementedError

    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)
    
    # extract triplets
    print('Extracting')
    output_data = []
    for item in tqdm(input_data):
        assert "response" in item, "response field is required"
        response = item["response"]
        question = item.get("question", None)
        triplets = extractor.extract_claim_triplets(response, question, max_new_tokens=args.extractor_max_new_tokens)
        out_item = {**item, **{"triplets": triplets}}
        output_data.append(out_item)
    with open(args.output_path, "w") as fp:
        json.dump(output_data, fp, indent=2)


def check(args):
    # initialize models
    if args.checker_name == "claude2":
        checker = Claude2Checker()
    elif args.checker_name == "gpt4":
        checker = GPT4Checker()
    elif args.checker_name == "nli":
        checker = NLIChecker()
    elif args.checker_name == "alignscore":
        checker = AlignScoreChecker()
    elif args.checker_name == "repc":
        checker = RepCChecker()
    else:
        raise NotImplementedError
    
    retriever = None
    if args.use_retrieval:
        if args.retriever_name == "google":
            retriever = GoogleRetriever(args.cache_dir)
        else:
            raise NotImplementedError
    
    if args.aggregator_name == "strict":
        agg_fn = strict_agg
    elif args.aggregator_name == "soft":
        agg_fn = soft_agg
    elif args.aggregator_name == "major":
        agg_fn = major_agg
    else:
        raise NotImplementedError
    
    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)
    
    # check triplets
    print('Checking')
    output_data = []
    for item in tqdm(input_data):
        assert "triplets" in item, "triplets field is required"
        triplets = item["triplets"]
        if args.use_retrieval:
            reference = retriever.retrieve(item["response"])
            item["reference"] = reference
        else:
            assert "reference" in item, \
                "reference field is required if retriever is not used."
            reference = item["reference"]
        question = item.get("question", None)
        results = [
            checker.check(t, reference, question=question)
            for t in triplets
        ]
        agg_results = agg_fn(results)
        out_item = {
            **item,
            **{
                "Y": agg_results,
                "ys": results,
            }
        }
        output_data.append(out_item)
    with open(args.output_path, "w") as fp:
        json.dump(output_data, fp, indent=2)


if __name__ == "__main__":
    main()
