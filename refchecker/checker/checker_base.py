from typing import List, Union
from itertools import groupby

from ..utils import split_text
from ..base import RCClaim


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
        """
        Initializer for the CheckerBase class.

        Initialize labels for 'Entailment', 'Neutral', and 'Contradiction'.
        Also initializes a list of all labels.
        """

        self.label_entailment = 'Entailment'
        self.label_neutral = 'Neutral'
        self.label_contradiction = 'Contradiction'
        self.labels = ["Entailment", "Neutral", "Contradiction"]

    def check(
        self, 
        batch_claims: List[List[Union[str, List[str]]]],
        batch_references: Union[List[str], List[List[str]]],
        batch_questions: List[str] = None,
        max_reference_segment_length: int = 0,
        merge_psg: bool = False,
        is_joint: bool = True,
        joint_check_num: int = 5,
        sagemaker_client=None,
        sagemaker_params=None,
        sagemaker_get_response_func=None,
        custom_llm_api_func=None,
        **kwargs
    ):
        """
        Check claims against references.

        Parameters
        ----------
        batch_claims : List[List[Union[str, List[str]]]]
            List consists of the claims extracted from each given example.
        batch_references : Union[List[str], List[List[str]]]
            List of reference passages for each given example.
        batch_questions : List[str], optional
            List of questions for each given example, defaults to None.
        max_reference_segment_length : int, optional
            Maximum length of each reference segment, defaults to 0.
        merge_psg : bool, optional
            Whether to merge results from multiple passages, defaults to False.
        is_joint: bool, optional
            Whether perform joint checking for claims to accelerate the checking process.
        joint_check_num: int, optional
            Number of claims to check jointly in one prompt. Defaults to 5.
        
        Returns
        -------
        results : List[List[str]]
            Grouped triplet checking results corresponding to each given example.

        """
        assert len(batch_claims) == len(batch_references)
        if batch_questions is None:
            batch_questions = [None] * len(batch_claims)
        
        # check whether the claims or references are empty
        valid_batch_claims = []
        valid_batch_references = []
        valid_batch_questions = []
        
        empty_claim_indices = set()
        empty_ref_indices = set()
        for index, (claims, references, questions) in enumerate(zip(batch_claims, batch_references, batch_questions)):
            if len(claims) == 0:
                empty_claim_indices.add(index)
            if isinstance(references, list) and len(references) == 0:
                empty_ref_indices.add(index)

            if index not in empty_claim_indices and index not in empty_ref_indices:
                valid_batch_claims.append(claims)
                valid_batch_references.append(references)
                valid_batch_questions.append(questions)
        
        if is_joint:
            # joint checking is for LLM-based checkers only, and it doesn't need merge_psg
            labels = self._check(
                claims=valid_batch_claims, 
                references=valid_batch_references, 
                questions=valid_batch_questions,
                is_joint=True,
                joint_check_num=joint_check_num,
                sagemaker_client=sagemaker_client,
                sagemaker_params=sagemaker_params,
                sagemaker_get_response_func=sagemaker_get_response_func,
                custom_llm_api_func=custom_llm_api_func,
                **kwargs
            )
            if merge_psg:
                labels = [
                    [merge_multi_psg_ret(claim_labels) for claim_labels in item_labels]
                    for item_labels in labels
                ]
        else:
            input_flattened = []
            input_ids = []
            for idx, (claims, references, questions) in enumerate(zip(valid_batch_claims, valid_batch_references, valid_batch_questions)):
                if isinstance(references, str):
                    references = [references]
                segments_all_psg = []
                for psg in references:
                    if max_reference_segment_length > 0:
                        segments = split_text(psg, max_reference_segment_length)
                    else:
                        segments = [psg]
                    segments_all_psg.append(segments)
                for c_idx, claim in enumerate(claims):
                    for idx_psg, seg_psg in enumerate(segments_all_psg):
                        for seg in seg_psg:
                            input_flattened.append([claim, seg, questions])
                            input_ids.append([idx, c_idx, idx_psg])
            ret = self._check(
                    claims=[inp[0] for inp in input_flattened],
                    references=[inp[1] for inp in input_flattened],
                    questions=[inp[2] for inp in input_flattened],
                    is_joint=False,
                    sagemaker_client=sagemaker_client,
                    sagemaker_params=sagemaker_params,
                    sagemaker_get_response_func=sagemaker_get_response_func,
                    custom_llm_api_func=custom_llm_api_func,
                )

            ret = [[x] + y for x, y in zip(ret, input_ids)]
            ret_merge_seg = [[merge_ret([item[0] for item in group])] + key[:-1] for key, group in groupby(ret, key=lambda x: x[1:])]
            if merge_psg:
                ret_merge_psg = [
                    [merge_multi_psg_ret([item[0] for item in group])] + key[:-1] 
                    for key, group in groupby(ret_merge_seg, key=lambda x: x[1:])
                ]
            else:
                ret_merge_psg = [
                    [[item[0] for item in group]] + key[:-1]
                    for key, group in groupby(ret_merge_seg, key=lambda x: x[1:])
                ]
            labels = [[item[0] for item in group] for key, group in groupby(ret_merge_psg, key=lambda x: x[1:])]

        # filling the results with empty claims or references
        final_labels = []
        cur_i = 0
        for index, (claims, references) in enumerate(zip(batch_claims, batch_references)):
            if index in empty_claim_indices:
                final_labels.append([])
            elif index in empty_ref_indices:
                final_labels.append([[]] * len(claims))
            else:
                final_labels.append(labels[cur_i])
                cur_i += 1
        
        return final_labels # [batch_size, claim_num, reference_num]


    def _check(
        self,
        claims: List[RCClaim],
        references: List[str],
        responses: List[str],
        questions: List[str],
        **kwargs
    ):
        """
        Internal method for checking claims against references.

        This method should be implemented by subclasses.

        Parameters
        ----------
        claims : List[RCClaim]
            List of claims to be checked.
        references : List[str]
            List of reference passages.
        responses : List[str]
            List of model response texts.
        questions : List[str]
            List of questions.

        Returns
        -------
        List[str]
            List of checking results.
        """

        raise NotImplementedError
