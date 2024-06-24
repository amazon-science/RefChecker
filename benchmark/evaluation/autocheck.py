import argparse
import json
from tqdm import tqdm
import os

from refchecker import (
    LLMChecker,
    NLIChecker,
    AlignScoreChecker,
    RepCChecker
)

from refchecker.extractor import LLMExtractor
from refchecker.checker import LLMChecker


def _get_checker(checker_model):
    checker = None
    if checker_model == 'nli':
        checker = NLIChecker()
    elif checker_model == 'alignscore':
        checker = AlignScoreChecker(
            batch_size=args.batch_size
        )
    elif checker_model == 'repc':
        checker = RepCChecker()
    else:
        checker = LLMChecker(
            model=checker_model,
            api_base=args.api_base,
            batch_size=args.batch_size
        )
    return checker


def _get_extractor(extractor_model):
    claim_extractor = LLMExtractor(
        claim_format=args.claim_format,
        model=extractor_model,
        api_base=args.api_base,
        batch_size=args.batch_size
    )
    return claim_extractor


def autocheck(extractor_model, checker_model):    
    claim_extractor = None
    checker = None

    for setting, ds in zip(
        ["zero_context", "noisy_context", "accurate_context"],
        ["nq", "msmarco", "dolly"]
    ):
        print(f'Evaluating {args.model} on {setting} setting with {extractor_model} extractor and {checker_model} checker')
        
        input_dir = os.path.join('human_annotations_v1', setting)
        response_filename = f'{ds}_{args.model}_answers.json'
        output_dir = os.path.join(args.output_dir, setting)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_filename = os.path.join(output_dir, response_filename)
        
        if os.path.exists(output_filename):
            response_data = json.load(open(output_filename))
        else:
            response_data = json.load(open(os.path.join(input_dir, response_filename)))
        # in case the order of response data is not aligned with ours
        id_to_data = {d['id']: d for d in json.load(open(f'data/{setting}/{ds}.json'))}
        
        # === Extraction ===
        batch_questions = []
        batch_responses = []
        kg_key = f'{extractor_model}_response_kg'
        for r in response_data:
            d = id_to_data[r['id']]
            r['question'] = d['question']
            r['context'] = d['context']
            if kg_key not in r:
                batch_questions.append(r['question'])
                batch_responses.append(r['response'])
        
        if len(batch_responses):
            if claim_extractor is None:
                claim_extractor = _get_extractor(extractor_model)
            
            print(f'Running Claim Extraction on {len(batch_responses)} examples...')
            extraction_results = claim_extractor.extract(
                batch_responses=batch_responses,
                batch_questions=batch_questions,
                max_new_tokens=1000
            )

            assert len(extraction_results) == len(batch_responses)
            
            _i = 0
            for r in response_data:
                if kg_key not in r:
                    r[kg_key] = [{'claim': c.content, 'attributed_sent_ids': c.attributed_sent_ids} for c in extraction_results[_i].claims]
                    r['extraction_orig_response'] = extraction_results[_i].extractor_response
                    _i += 1

            json.dump(response_data, open(output_filename, 'w'), indent=4)

        # # === Checking ===
        batch_claims = []
        batch_references = []
        batch_questions = []
        batch_responses = []
        
        label_key = f'{checker_model}_label'
        for r in response_data:
            if kg_key in r:
                claims = [c['claim'] for c in r[kg_key] if label_key not in c]
                if len(claims):
                    batch_claims.append(claims)
                    _references = []
                    if len(r['context']) > 1:
                        for pi, psg in enumerate(r['context']):
                            _references.append(f'Passage {pi}: {psg}')
                    else:
                        for pi, psg in enumerate(r['context']):
                            _references.append(psg)
                    batch_references.append(_references)
                    
                    batch_questions.append(r['question'])
                    batch_responses.append(r['response'])

        if checker_model in ['nli', 'alignscore', 'repc'] and ds != 'nq':
            max_reference_segment_length = 200
        else:
            max_reference_segment_length = 0

        if len(batch_claims):
            print(f'Running Checking on {len(batch_claims)} examples...')
            if checker is None:
                checker = _get_checker(checker_model)
            
            checking_results = checker.check(
                batch_claims=batch_claims,
                batch_references=batch_references,
                batch_questions=batch_questions,
                batch_responses=batch_responses,
                max_reference_segment_length=max_reference_segment_length,
                merge_psg=True
            )
            _i = 0
            for r in response_data:
                d = id_to_data[r['id']]
                if kg_key in r:
                    claims = [c['claim'] for c in r[kg_key] if label_key not in c]
                    if len(claims):
                        labels = checking_results[_i]
                        _j = 0
                        for claim in r[kg_key]:
                            if label_key not in claim:
                                claim[label_key] = labels[_j]
                                _j += 1
                        _i += 1
            json.dump(response_data, open(output_filename, 'w'), indent=4)
        
        cnt = 0
        for r in response_data:
            if kg_key in r:
                is_checking_finished = True
                for t in r[kg_key]:
                    if label_key not in t:
                        is_checking_finished = False
                        break
                if is_checking_finished:
                    cnt += 1
        
        print(f'{setting}: {cnt} finished.')


def main():
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    autocheck(args.extractor, args.checker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--claim_format', type=str, choices=['triplet', 'subsentence'])
    parser.add_argument('--checker', type=str)
    parser.add_argument('--api_base', type=str)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()

    main()