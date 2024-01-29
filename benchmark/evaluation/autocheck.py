import argparse
import json
from tqdm import tqdm

from refchecker import (
    GPT4Extractor, 
    Claude2Extractor, 
    MixtralExtractor,
    GPT4Checker, 
    Claude2Checker, 
    NLIChecker,
    AlignScoreChecker,
    RepCChecker
)


def _get_checker(checker_model):
    checker = None
    if checker_model == 'gpt4':
        checker = GPT4Checker()
    elif checker_model == 'claude2':
        checker = Claude2Checker()
    elif checker_model == 'nli':
        checker = NLIChecker()
    elif checker_model == 'alignscore':
        checker = AlignScoreChecker()
    elif checker_model == 'repc':
        checker = RepCChecker()
    return checker


def _get_extractor(extractor_model):
    claim_extractor = None
    if extractor_model == 'gpt4':
        claim_extractor = GPT4Extractor()
    elif extractor_model == 'claude2':
        claim_extractor = Claude2Extractor()
    elif extractor_model == 'mixtral':
        claim_extractor = MixtralExtractor()
    return claim_extractor


def autocheck(extractor_model, checker_model):
    print(extractor_model, checker_model)
    
    claim_extractor = None
    checker = None

    for setting, ds in zip(
        ["zero_context", "noisy_context", "accurate_context"],
        ["nq", "msmarco", "dolly"]
    ):
        print(f'Evaluating {args.model} on {setting} setting with {extractor_model} extractor and {checker_model} checker')
        # response_file = f'data/{setting}/{setting}_{args.model}_answers.json'
        response_file = f'compare_other_approach/{ds}/{ds}_{args.model}_answers.json'
        response_data = json.load(open(response_file))
        # in case the order of response data is not aligned with ours
        # id_to_data = {d['id']: d for d in json.load(open(f'data/{setting}/{ds}.json'))}
        id_to_data = {d['id']: d for d in json.load(open(f'benchmark/data/{setting}/{ds}.json'))}
        
        cnt = 0
        for r in tqdm(response_data):
            d = id_to_data[r['id']]

            # claim extraction
            kg_key = f'{extractor_model}_response_kg'
            if kg_key not in r:
                if claim_extractor is None:
                    claim_extractor = _get_extractor(extractor_model)
                claims = claim_extractor.extract(
                    question=d['question'],
                    response=r['response']
                )
                if claims is not None:
                    r[kg_key] = [{'triplet': t} for t in claims]
                    json.dump(response_data, open(response_file, 'w'), indent=4)
            
            # checking
            label_key = f'{checker_model}_label'
            if kg_key in r and len(r[kg_key]):
                # reference
                if checker_model == 'nli':
                    reference = []
                    for pi, psg in enumerate(d['context']):
                        reference.append(f'Passage {pi}: {psg}')
                else:
                    reference = ''
                    for pi, psg in enumerate(d['context']):
                        reference += f'Passage {pi}: {psg}\n'

                for t in r[kg_key]:
                    if label_key not in t:
                        if checker_model == 'nli' and ds != 'nq':
                            max_reference_segment_length = 200
                        else:
                            max_reference_segment_length = 0
                            
                        if checker_model == 'nli':
                            claim_str = ' '.join(t['triplet'])
                        else:
                            claim_str = str(tuple(t['triplet']))

                        if checker is None:
                            checker = _get_checker(checker_model)
                        label = checker.check(
                            claim=claim_str,
                            reference=reference,
                            question=d['question'],
                            response=r['response'],
                            max_reference_segment_length=max_reference_segment_length
                        )
                        if label:
                            t[label_key] = label
                            json.dump(response_data, open(response_file, 'w'), indent=4)
            
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
    autocheck(args.extractor, args.checker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument(
        '--extractor', 
        type=str, 
        choices=['gpt4', 'claude2', 'mixtral']
    )
    parser.add_argument(
        '--checker', 
        type=str, 
        choices=['gpt4', 'claude2', 'nli', 'alignscore', 'repc']
    )

    args = parser.parse_args()

    main()