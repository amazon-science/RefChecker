from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

from collector_base import ResponseCollectorBase


class Mistral(ResponseCollectorBase):
    def __init__(
        self,
        mname,
        device='cuda'
    ) -> None:
        super().__init__(mname, device)
        
        self.tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        
        self.llm = LLM(
            model='mistralai/Mistral-7B-Instruct-v0.1',
            tokenizer='mistralai/Mistral-7B-Instruct-v0.1',
            trust_remote_code=True,
            tensor_parallel_size=8)
        
        self.max_contex_length = 8100
        
        self.sampling_params = SamplingParams(temperature=0, max_tokens=self.max_new_tokens)
    
    def tokenizer_encode(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False)
    
    def tokenizer_decode(self, encoded):
        return self.tokenizer.decode(encoded)
    
    def get_response(self, prompt):
        prompt = f'<s>[INST] {prompt} [/INST]'
        
        res = self.llm.generate([prompt], self.sampling_params, use_tqdm=False)[0].outputs[0].text
        res = res.strip()
        return res