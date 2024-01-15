import argparse
import json
from datasets import load_dataset


def process_dolly():
    dolly_data = load_dataset('databricks/databricks-dolly-15k', split='train')
    example_ids = json.load(open('accurate_context/dolly_example_ids.json'))
    
    chosen_examples = dict()
    for i, d in enumerate(dolly_data):
        if str(i) in example_ids:
            ex = {'id': str(i)}
            ex['question'] = d['instruction']
            ex['context'] = [d['context']]
            ex['category'] = d['category']
            ex['human_response'] = d['response']
            
            chosen_examples[ex['id']] = ex

    ret = [chosen_examples[ex_id] for ex_id in example_ids]
    json.dump(ret, open('accurate_context/dolly.json', 'w'), indent=4)


def process_msmarco():
    ms_data = load_dataset('ms_marco', 'v2.1', split='validation')
    example_ids = json.load(open('noisy_context/msmarco_example_ids.json'))
    
    chosen_examples = dict()
    for d in ms_data:
        if str(d['query_id']) in example_ids:
            ex = {'id': str(d['query_id'])}
            ex['question'] = d['query']
            ex['context'] = d['passages']['passage_text']
            ex['query_type'] = d['query_type']
            ex['answers'] = d['answers']
            ex['wellFormedAnswers'] = d['wellFormedAnswers']
            ex['context_is_selected'] = d['passages']['is_selected']
            ex['context_ur'] = d['passages']['url']
            
            chosen_examples[ex['id']] = ex
    
    ret = [chosen_examples[ex_id] for ex_id in example_ids]
    json.dump(ret, open('noisy_context/msmarco.json', 'w'), indent=4)


def process_nq():
    example_ids = json.load(open('zero_context/nq_example_ids.json'))
    
    chosen_examples = dict()
    for i in range(5):
        with open(f'zero_context/nq/dev/nq-dev-0{i}.jsonl') as f:
            for l in f.readlines():
                l = json.loads(l)
                assert 'example_id' in l
                if str(l['example_id']) in example_ids:
                    ex = {'id': str(l['example_id'])}
                    ex['question'] = l['question_text']
                    
                    annotations = []
                    for anno in l['annotations']:
                        cleaned_anno = dict()
                        cleaned_anno['short_answers'] = []
                        for short_ans in anno['short_answers']:
                            short_ans_start = short_ans['start_token']
                            short_ans_end = short_ans['end_token']
                            short_answer = [tok['token'] for tok in l['document_tokens'][short_ans_start: short_ans_end] if
                                            not tok['html_token']]
                            cleaned_anno['short_answers'].append(' '.join(short_answer).strip())
                        long_ans_start = anno['long_answer']['start_token']
                        long_ans_end = anno['long_answer']['end_token']
                        long_answer = [tok['token'] for tok in l['document_tokens'][long_ans_start: long_ans_end] if
                                    not tok['html_token']]
                        cleaned_anno['long_answer'] = ' '.join(long_answer).strip()
                        
                        if len(cleaned_anno['short_answers']) > 0 and any([len(a) for a in cleaned_anno['short_answers']]) and len(cleaned_anno['long_answer']) > 0:
                            annotations.append(cleaned_anno)
                    assert len(annotations) > 0
                    ex['context'] = [annotations[0]['long_answer']]
                    ex['short_answers'] = annotations[0]['short_answers']
                    chosen_examples[ex['id']] = ex
    ret = [chosen_examples[ex_id] for ex_id in example_ids]
    json.dump(ret, open('zero_context/nq.json', 'w'), indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['nq', 'msmarco', 'dolly'])
    
    args = parser.parse_args()
    
    if args.dataset == 'nq':
        process_nq()
    elif args.dataset == 'msmarco':
        process_msmarco()
    elif args.dataset == 'dolly':
        process_dolly()
