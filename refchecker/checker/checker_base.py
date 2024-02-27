from typing import List, Union
from itertools import groupby

from ..utils import split_text


def merge_ret(ret):
    """Merge results from multiple paragraphs"""
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"


def merge_multi_psg_ret(ret):
    """Merge results from multiple passages
    TODO: consider possible cases where the results are inconsistent.
    """
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"


class CheckerBase:
    def __init__(self) -> None:
        self.label_entailment = 'Entailment'
        self.label_neutral = 'Neutral'
        self.label_contradiction = 'Contradiction'
        self.labels = ["Entailment", "Neutral", "Contradiction"]

    def check(
        self, 
        claim: List[List[List[str]]],
        reference: Union[List[str], List[List[str]]],
        response: List[str] = None,
        question: List[str] = None,
        max_reference_segment_length: int = 200, 
    ):
        if response == None:
            response = [None] * len(claim)
        if question == None:
            question = [None] * len(claim)
        input_flattened = []
        input_ids = []
        for idx, (c, ref, res, q) in enumerate(zip(claim, reference, response, question)):
            if isinstance(ref, str):
                ref = [ref]
            segments_all_psg = []
            for psg in ref:
                if max_reference_segment_length > 0:
                    segments = split_text(psg, max_reference_segment_length)
                else:
                    segments = [psg]
                segments_all_psg.append(segments)
            for c_idx, t in enumerate(c):
                for idx_psg, seg_psg in enumerate(segments_all_psg):
                    for seg in seg_psg:
                        input_flattened.append([t, seg, res, q])
                        input_ids.append([idx, c_idx, idx_psg])
        ret = self._check(
                claims=[inp[0] for inp in input_flattened],
                references=[inp[1] for inp in input_flattened],
                responses=[inp[2] for inp in input_flattened],
                questions=[inp[3] for inp in input_flattened],
            )

        ret = [[x] + y for x, y in zip(ret, input_ids)]
        ret_merge_seg = [[merge_ret([item[0] for item in group])] + key[:-1] for key, group in groupby(ret, key=lambda x: x[1:])]
        ret_merge_psg = [[merge_multi_psg_ret([item[0] for item in group])] + key[:-1] for key, group in groupby(ret_merge_seg, key=lambda x: x[1:])]
        ret_group_triplet = [[item[0] for item in group] for key, group in groupby(ret_merge_psg, key=lambda x: x[1:])]

        return ret_group_triplet

    def _check(
        self,
        claims: List[str],
        references: List[str],
        responses: List[str],
        questions: List[str]
    ):
        raise NotImplementedError
