from transformers import AutoModel, AutoTokenizer

from collector_base import ResponseCollectorBase


class ChatGLM3(ResponseCollectorBase):
    def __init__(
        self,
        mname,
        device='cuda'
    ) -> None:
        super().__init__(mname, device)
        
        self.model = AutoModel.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm3-6b", trust_remote_code=True)

        self.max_contex_length = 8100
            
    def tokenizer_encode(self, prompt):
        return self.tokenizer.encode(prompt, add_special_tokens=False)
    
    def tokenizer_decode(self, encoded):
        return self.tokenizer.decode(encoded)
    
    def get_response(self, prompt):        
        res, _ = self.model.chat(self.tokenizer, prompt, history=[], do_sample=False)
        res = res.strip()
        return res