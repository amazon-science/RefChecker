import tiktoken


from bschecker.utils import get_openai_model_response
from collector_base import ResponseCollectorBase

class GPT4Turbo(ResponseCollectorBase):
    def __init__(self, mname, device='cuda') -> None:
        super().__init__(mname, device)
        
        self.tokenizer = tiktoken.encoding_for_model("gpt-4-1106-preview")
        self.max_contex_length = 127000

    def tokenizer_encode(self, prompt):
        return self.tokenizer.encode(prompt)

    def tokenizer_decode(self, encoded):
        return self.tokenizer.decode(encoded)
    
    def get_response(self, prompt):
        res = get_openai_model_response(prompt, temperature=0, model='gpt-4-1106-preview')
        return res.strip()