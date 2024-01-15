import json
import numpy as np
import argparse
from collections import Counter


def get_evaluation_results(model, extractor, checker):
    ret = dict()
    avg_abstain_list = []
    avg_contra_list = []
    avg_entail_list = []
    avg_neutral_list = []

    for setting in ['zero_context', 'noisy_context', 'accurate_context']:
        contra_list = []
        entail_list = []
        neutral_list = []
        
        abstain_cnt = 0
        response_data = json.load(open(f'data/{setting}/{setting}_{model}_answers.json'))
        for r in response_data:
            c, e, n = 0, 0, 0
            
            if f'{extractor}_response_kg' not in r:
                n_triplets = 0
            else:
                n_triplets = len(r[f'{extractor}_response_kg'])
            if n_triplets == 0:
                # abstain response
                abstain_cnt += 1
            else:
                # non abstain response
                if checker == 'ensemble':
                    for t in r[f'{extractor}_response_kg']:
                        v = Counter([t['gpt4_label'], t['claude2_label'], t['nli_label']]).most_common(1)[0][0]
                        if v == 'Entailment':
                            e += 1
                        elif v == 'Neutral':
                            n += 1
                        elif v == 'Contradiction':
                            c += 1
                        else:
                            n += 1
                else:
                    for v in [x[f'{checker}_label'] for x in r[f'{extractor}_response_kg']]:
                        if v == 'Entailment':
                            e += 1
                        elif v == 'Neutral':
                            n += 1
                        elif v == 'Contradiction':
                            c += 1
                        else:
                            n += 1
                assert e + n + c == n_triplets, r[f'{extractor}_response_kg']
                contra_list.append(c / n_triplets)
                entail_list.append(e / n_triplets)
                neutral_list.append(n / n_triplets)
        abstain_rate = abstain_cnt / len(response_data)
            
        ret[setting] = {
            'abstain': abstain_rate * 100,
            'entailment': np.mean(entail_list) * 100,
            'neutral': np.mean(neutral_list) * 100,
            'contradiction': np.mean(contra_list) * 100
        }
        avg_entail_list += entail_list
        avg_neutral_list += neutral_list
        avg_contra_list += contra_list
        avg_abstain_list.append(abstain_rate)
    
    ret['avg'] = {
        'abstain': np.mean(avg_abstain_list) * 100,
        'entailment': np.mean(avg_entail_list) * 100,
        'neutral': np.mean(avg_neutral_list) * 100,
        'contradiction': np.mean(avg_contra_list) * 100
    }
    
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--checker', type=str)
    parser.add_argument('--output_file', type=str)
    
    args = parser.parse_args()
    
    ret = get_evaluation_results(args.model, args.extractor, args.checker)
    json.dump(ret, open(args.output_file, 'w'), indent=4)