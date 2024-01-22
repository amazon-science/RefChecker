from .inference import Inferencer
from typing import List

class AlignScore:
    def __init__(self, model: str, batch_size: int, device: int, ckpt_path: str, evaluation_mode='nli_sp', verbose=True) -> None:
        self.model = Inferencer(
            ckpt_path=ckpt_path, 
            model=model,
            batch_size=batch_size, 
            device=device,
            verbose=verbose
        )
        self.model.nlg_eval_mode = evaluation_mode

    def score(self, contexts: List[str], claims: List[str]) -> List[float]:
        return self.model.nlg_eval(contexts, claims)[1].tolist()