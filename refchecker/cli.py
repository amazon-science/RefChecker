import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter
from tqdm import tqdm

from .extractor import LLMExtractor
from .checker import (
    LLMChecker, NLIChecker, AlignScoreChecker, RepCChecker
)
from .retriever import GoogleRetriever
from .aggregator import strict_agg, soft_agg, major_agg
from .base import RCClaim


def get_args():
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument(
        "mode", nargs="?", choices=["extract", "check", "extract-check"],
        help="extract:       Extract claims from provided responses.\n"
             "check:         Check whether the provided claims are factual.\n"
             "extract-check: Extract claims and check whether they are factual."
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
        '--extractor_name', type=str, default="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        help="Model used for extracting triplets. Default: bedrock/anthropic.claude-3-sonnet-20240229-v1:0"
    )
    parser.add_argument(
        '--extractor_max_new_tokens', type=int, default=500,
        help="Max generated tokens of the extractor, set a larger value for longer documents. Default: 500"
    )
    parser.add_argument(
        '--claim_format', type=str, default='triplet',
        choices=['triplet', 'subsentence'],
        help='The format of the extracted claims. Default: subsentence'
    )
    parser.add_argument(
        "--checker_name", type=str, default="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        help="Model used for checking whether the triplets are factual. "
        "Default: Claude 3 Sonnet"
    )
    parser.add_argument(
        "--extractor_api_base", type=str,
        help="API base URL if using vllm for deploying the extractor"
    )
    parser.add_argument(
        "--checker_api_base", type=str,
        help="API base URL if using vllm for deploying the checker"
    )
    parser.add_argument(
        "--repc_classifier_name", type=str, default="nn_ensemble",
        choices=["svm", "svm_ensemble", "nn", "nn_ensemble"],
        help="Classifier Model used for RepC checker, only valid when RepC checker is used. "
        "Default: nn_ensemble, neural network classifier with layer ensemble."
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
        "--use_retrieval", action="store_true",
        help="Whether to use retrieval to find the reference for checking. "
        "Required if the reference\nfield in input data is not provided."
    )
    parser.add_argument(
        "--batch_size_extractor", type=int, default=16,
        help="Batch size for extractor."
    )
    parser.add_argument(
        "--batch_size_checker", type=int, default=16,
        help="Batch size for checker."
    )

    return parser.parse_args()


def main():
    args = get_args()
    # set environment variables
    # if args.openai_key:
    #     with open(args.openai_key, "r") as fp:
    #         os.environ["OPENAI_API_KEY"] = fp.read().strip()
    # if args.anthropic_key:
    #     with open(args.anthropic_key, "r") as fp:
    #         os.environ["ANTHROPIC_API_KEY"] = fp.read().strip()
    # if args.aws_bedrock_region:
    #     os.environ["AWS_REGION_NAME"] = args.aws_bedrock_region
    # if args.serper_api_key:
    #     os.environ["SERPER_API_KEY"] = args.serper_api_key

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
    extractor = LLMExtractor(
        claim_format=args.claim_format, 
        model=args.extractor_name, 
        api_base=args.extractor_api_base
    )

    # load data
    with open(args.input_path, "r") as fp:
        input_data = json.load(fp)
    
    # extract triplets
    print('Extracting')
    question_list = [d.get('question', None) for d in input_data]
    response_list = [d['response'] for d in input_data]
    
    extraction_results = extractor.extract(
        batch_responses=response_list, 
        batch_questions=question_list, 
        max_new_tokens=args.extractor_max_new_tokens
    )
    for res, d in zip(extraction_results, input_data):
        d['claims'] = [c.content for c in res.claims]
    
    with open(args.output_path, "w") as fp:
        json.dump(input_data, fp, indent=2)


def check(args):
    # initialize models
    if args.checker_name == "nli":
        checker = NLIChecker(batch_size=args.batch_size_checker)
    elif args.checker_name == "alignscore":
        checker = AlignScoreChecker(batch_size=args.batch_size_checker)
    elif args.checker_name == "repc":
        checker = RepCChecker(classifier=args.repc_classifier_name, batch_size=args.batch_size_checker)
    else:
        checker = LLMChecker(
            model=args.checker_name, 
            batch_size=args.batch_size_checker,
            api_base=args.checker_api_base
        )
    
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
    
    # check claims
    print('Checking')
    claim_list = []
    reference_list = []
    question_list = []
    for item in input_data:
        assert "claims" in item, "claims field is required"
        claims = item['claims']
        if args.use_retrieval:
            reference = retriever.retrieve(item["response"])
            item["reference"] = reference
        else:
            assert "reference" in item, \
                "reference field is required if retriever is not used."
            reference = item["reference"]
        question = item.get("question", None)
        claim_list.append(claims)
        reference_list.append(reference)
        question_list.append(question)

    results = checker.check(
        batch_claims=claim_list, 
        batch_references=reference_list, 
        batch_questions=question_list)
    agg_results = [agg_fn(r) for r in results]
    
    output_data = [{
        **input_data[i],
        **{
            "Y": agg_results[i],
            "ys": results[i],
        }
    } for i in range(len(input_data))]
    with open(args.output_path, "w") as fp:
        json.dump(output_data, fp, indent=2)


if __name__ == "__main__":
    main()
