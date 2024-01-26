import os
import json
from tqdm import tqdm


closed_qa_prompt = """Instruction: Provide a well-formed answer to the question using information from the given context.
Question: {question}
Context: {context}
"""

ie_prompt = """Instruction: {question}
Context: {context}
"""

sum_prompt = """Instruction: {question}
Context: {context}
"""


class ResponseCollectorBase:
    def __init__(
        self, 
        mname,
        device='cuda'
    ) -> None:
        self.model = None
        self.tokenizer = None
        
        self.mname = mname
        self.device = device
        
        self.max_contex_length = None
        self.max_new_tokens = 300
    
    def collect_response(self):
        assert self.max_contex_length is not None
        
        for setting, ds in zip(
            ['zero_context', 'noisy_context', 'accurate_context'],
            ["nq", "msmarco", "dolly"]
        ):
            examples = json.load(open(f'data/{setting}/{ds}.json'))
            response_file = f'data/{setting}/{setting}_{self.mname}_answers.json'
            if os.path.exists(response_file):
                response_data = json.load(open(response_file))
            else:
                response_data = [{'id': ex['id']} for ex in examples]
                json.dump(response_data, open(response_file, 'w'), indent=4)
            
            finish_cnt = 0
            for ex, r in tqdm(zip(examples, response_data), total=len(examples)):
                if 'response' in r:
                    finish_cnt += 1
                    continue
                input_prompt = self.get_input(ds, ex)
                res = self.get_response(input_prompt)
                if res and len(res):
                    r['input'] = input_prompt
                    r['response'] = res
                    json.dump(response_data, open(response_file, 'w'), indent=4)
                    finish_cnt += 1
            print(f'{setting}: {finish_cnt} responses collected.')
    
    def get_input(self, split, example):
        if split == 'nq':
            return example['question']
        elif split == 'msmarco':
            tail = '\nAnswer: \n'
            
            prompt = f'Please answer the following question based on the provided passages.\n\nQuestion: {example["question"]}?\n\nPassages:\n'
            for i, p in enumerate(example['context']):
                prompt += f'Passage {i}: {p}\n'
            prompt_encoded = self.tokenizer_encode(prompt)
            if prompt_encoded:
                tail_length = len(self.tokenizer_encode(tail))
                
                prompt_encoded = prompt_encoded[:self.max_contex_length - self.max_new_tokens - tail_length]
                prompt_truncated = self.tokenizer_decode(prompt_encoded)
                prompt_truncated += tail
                return prompt_truncated
            return prompt + tail
        elif split == 'dolly':
            if example['category'] == 'closed_qa':
                prompt = closed_qa_prompt.format(**{'question': example['question'], 'context': example['context'][0]})
            elif example['category'] == 'information_extraction':
                prompt = closed_qa_prompt.format(**{'question': example['question'], 'context': example['context'][0]})
            elif example['category'] == 'summarization':
                prompt = sum_prompt.format(**{'question': example['question'], 'context': example['context'][0]})
            prompt_encoded = self.tokenizer_encode(prompt)
            if prompt_encoded:
                prompt_encoded = prompt_encoded[:self.max_contex_length - self.max_new_tokens]
                prompt_truncated = self.tokenizer_decode(prompt_encoded)
                return prompt_truncated
            return prompt
    
    def tokenizer_encode(self, prompt):
        raise NotImplementedError

    def tokenizer_decode(self, encoded):
        raise NotImplementedError

    def get_response(self, prompt):
        raise NotImplementedError
