from typing import Any, List, Union
from tqdm import tqdm

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification
)

from .checker_base import CheckerBase
from ..base import RCClaim


LABELS = ["Entailment", "Neutral", "Contradiction"]


class NLIChecker(CheckerBase):
    def __init__(
        self, 
        model='ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli',
        device=0,   # Todo: support distributed inference
        batch_size=16
    ):
        """
        Initializes the NLIChecker with the specified model, device, and batch size.

        Parameters
        ----------
        model : str, optional
            The name or identifier of the model to use, defaults to 'ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli'.
        device : int, optional
            The device to run inference on, defaults to 0.
        batch_size : int, optional
            The batch size for inference, defaults to 16.
        """

        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.device = device
        self.batch_size = batch_size

    @torch.no_grad()
    def _check(
        self,
        claims: List[RCClaim],
        references: List[str],
        responses: List[str],
        questions: List[str],
    ):
        """
        Batch checking claims against references.

        Parameters
        ----------
        claims : List[RCClaim]
            List of claims.
        references : List[str]
            List of reference passages (split according to 'max_reference_segment_length').
        responses : List[str]
            List of model response texts.
        questions : List[str]
            List of questions corresponding to each triplet.

        Returns
        -------
        ret : List[str]
            List of labels for the checking results.

        """

        N1, N2 = len(references), len(claims)
        assert N1 == N2, f"Batches must be of the same length. {N1} != {N2}"
        claims = [c.get_content() for c in claims]
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
