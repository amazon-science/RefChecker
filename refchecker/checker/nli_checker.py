from typing import Any, List
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)

from .checker_base import CheckerBase


LABELS = ["Entailment", "Neutral", "Contradiction"]


class NLIChecker(CheckerBase):
    def __init__(
        self, 
        model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
        device=0,
        batch_size=16
    ):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def _check(
        self,
        claims: List[List[str]],
        references: List[str],
        responses: List[str],
        questions: List[str],
    ):
        N1, N2 = len(references), len(claims)
        assert N1 == N2, f"Batches must be of the same length. {N1} != {N2}"
        if isinstance(claims[0], list):
            assert len(claims[0]) == 3
            claims = [f"{c[0]} {c[1]} {c[2]}" for c in claims]
        batch_preds = []
        for i in tqdm(range(0, len(claims), self.batch_size)):
            batch_claims = claims[i:i + self.batch_size]
            batch_references = references[i:i + self.batch_size]

            inputs = self.tokenizer(
                batch_references, batch_claims, max_length=512, truncation=True,
                return_tensors="pt", padding=True, return_token_type_ids=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            output = self.model(**inputs).logits.softmax(dim=-1).cpu()  # [batch_size, 3]
            preds = output.argmax(dim=-1)
            batch_preds.extend(preds)
        ret = [LABELS[p] for p in batch_preds]

        return ret
