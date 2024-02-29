import argparse
import json
from tqdm import tqdm

from refchecker import (
    GPT4Extractor, 
    Claude2Extractor, 
    MistralExtractor,
    LLMChecker,
    NLIChecker,
    AlignScoreChecker,
    RepCChecker
)


def _get_checker(checker_model):
    checker = None
    if checker_model in ["gpt4", "claude2"]:
        checker = LLMChecker(model=checker_model)
    elif checker_model == 'nli':
        checker = NLIChecker()
    elif checker_model == 'alignscore':
        checker = AlignScoreChecker()
    elif checker_model == 'repc':
        checker = RepCChecker()
    else:
        raise NotImplementedError
    return checker


def _get_extractor(extractor_model):
    claim_extractor = None
    if extractor_model == 'gpt4':
        claim_extractor = GPT4Extractor()
    elif extractor_model == 'claude2':
        claim_extractor = Claude2Extractor()
    elif extractor_model == 'mistral-sft':
        claim_extractor = MistralExtractor()
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
        response_file = f'human_annotations_v1/{setting}/{ds}_{args.model}_answers.json'
        response_data = json.load(open(response_file))
        # in case the order of response data is not aligned with ours
        id_to_data = {d['id']: d for d in json.load(open(f'data/{setting}/{ds}.json'))}
        
        cnt = 0
        kg_key = f'{extractor_model}_response_kg'
        for r in tqdm(response_data):
            d = id_to_data[r['id']]

            # claim extraction
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

        claim_list = []
        reference_list = []
        question_list = []
        response_list = []
        idx_list = []
        label_key = f'{checker_model}_label'
        for r_idx, r in enumerate(response_data):
            d = id_to_data[r['id']]
            # checking
            if kg_key in r and len(r[kg_key]):
                triplets = []
                ids = []
                # reference
                if checker_model == 'nli':
                    reference = []
                    for pi, psg in enumerate(d['context']):
                        reference.append(f'Passage {pi}: {psg}')
                else:
                    reference = ''
                    for pi, psg in enumerate(d['context']):
                        reference += f'Passage {pi}: {psg}\n'

                for t_idx, t in enumerate(r[kg_key]):
                    if label_key not in t:
                        triplets.append(t['triplet'])
                        ids.append((r_idx, t_idx))
                if len(triplets) > 0:
                    claim_list.append(triplets)
                    reference_list.append(reference)
                    question_list.append(d['question'])
                    response_list.append(r['response'])
                    idx_list.append(ids)

        if checker_model in ['nli', 'alignscore', 'repc'] and ds != 'nq':
            max_reference_segment_length = 200
        else:
            max_reference_segment_length = 0

        if checker is None:
            checker = _get_checker(checker_model)
        labels = checker.check(
            claim=claim_list,
            reference=reference_list,
            question=question_list,
            response=response_list,
            max_reference_segment_length=max_reference_segment_length
        )
        for idx_example, labels_example in enumerate(labels):
            for l_idx, label in enumerate(labels_example):
                if label:
                    r_idx = idx_list[idx_example][l_idx][0]
                    t_idx = idx_list[idx_example][l_idx][1]
                    response_data[r_idx][kg_key][t_idx][label_key] = label
                    json.dump(response_data, open(response_file, 'w'), indent=4)

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
    autocheck(args.extractor, args.checker)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument(
        '--extractor', 
        type=str, 
        choices=['gpt4', 'claude2', 'mistral-sft']
    )
    parser.add_argument(
        '--checker', 
        type=str, 
        choices=['gpt4', 'claude2', 'nli', 'alignscore', 'repc']
    )

    args = parser.parse_args()

    main()