import os
import subprocess
from typing import List
from tqdm import tqdm

from ..checker_base import CheckerBase
from .inference import Inferencer

import torch


LABELS = ["Entailment", "Neutral", "Contradiction"]


class AlignScoreChecker(CheckerBase):
    def __init__(
        self,
        ckpt_path='alignscore.ckpt',
        device=0,
        batch_size=16
    ):
        super().__init__()
        self._download_ckpt(ckpt_path)
        self.scorer = Inferencer(
            ckpt_path, model="roberta-large", device=device, verbose=False
        )
        self.batch_size = batch_size
    
    def _download_ckpt(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            url = "https://huggingface.co/yzha/AlignScore/resolve/main/AlignScore-large.ckpt"
            command=["wget", "-O", ckpt_path, url]
            try:
                subprocess.call(command)
            except Exception as e:
                print(e)

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
            scores = self.scorer.inference(premise=batch_references, hypo=batch_claims)[-1]
            preds = scores.argmax(dim=-1)
            batch_preds.extend(preds)
        ret = [LABELS[p] for p in batch_preds]

        return ret


if __name__ == "__main__":
    checker = AlignScoreChecker()
    print(checker._check(
        claims=["The dog is cute.", "The dog is cute."],
        references=["The dog is cute.", "The dog is not cute."],
        response=None, question=None
    ))
