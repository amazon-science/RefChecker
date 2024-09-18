import re
from typing import List, Union
from tqdm import tqdm
import numpy as np

from .checker_base import CheckerBase
from ..utils import get_model_batch_response, get_llm_full_name
from .checker_prompts import *


class LLMChecker(CheckerBase):
    def __init__(
        self,
        model: str = 'bedrock/anthropic.claude-3-sonnet-20240229-v1:0',
        batch_size: int = 16,
        api_base: str = None
    ) -> None:
        """
        Initializer for the LLMChecker class.

        Initializes LLMChecker with the provided model and batch size.

        Parameters:
        -----------
        model : str
            The name or identifier of the language model to use.
        batch_size : int, optional
            Batch size for checking, defaults to 16.
        """

        super().__init__()
        self.prompt_temp = LLM_CHECKING_PROMPT
        self.prompt_temp_wq = LLM_CHECKING_PROMPT_Q
        self.prompt_temp_subsent = SUBSENTENCE_CLAIM_CHECKING_PROMPT
        
        self.batch_size = batch_size
        
        self.model = get_llm_full_name(model)
        self.api_base = api_base

    def _check(
        self,
        claims: List[Union[str, List[str], List[List[str]]]],
        references: List[Union[str, List[str]]],
        responses: List[str] = None,
        questions: List[str] = None,
        is_joint: bool = False,
        joint_check_num: int = 5,
        custom_llm_api_func=None,
        **kwargs
    ):
        """
        Batch checking claims against references.

        Parameters
        ----------
        claims : List[Union[str, List[str]]]
            List of claims.
        references : List[str]
            List of reference passages (split according to 'max_reference_segment_length').
        responses : List[str]
            List of model response texts.
        questions : List[str]
            List of questions corresponding to each triplet.
        is_joint: bool, optional
            Whether perform joint checking for claims to accelerate the checking process.
        joint_check_num: int, optional
            Number of claims to check jointly in one prompt. Defaults to 5.
        Returns
        -------
        ret : List[str]
            List of labels for the checking results.

        """
        if is_joint:
            batch_claim_nums = [len(claims_per_batch) for claims_per_batch in claims]
            batch_ref_nums = []
            for ref_per_batch in references:
                if isinstance(ref_per_batch, str):
                    batch_ref_nums.append(1)
                else:
                    assert isinstance(ref_per_batch, list)
                    batch_ref_nums.append(len(ref_per_batch))
            
            prompt_template = JOINT_CHECKING_PROMPT_Q
            
            prompt_list = []
            prompt_ids = [] # for setting the limit of max num of claims
            claim_nums = []
            p_id = 0
            for claims_per_batch, references_per_batch, question_per_batch in zip(claims, references, questions):
                if len(claims_per_batch) == 0:
                    continue
                    
                if isinstance(references_per_batch, str):
                    references_per_batch = [references_per_batch]
                
                for ref in references_per_batch:
                    _claim_cnt = 0
                    claims_text = ''
                    
                    for _ci, c in enumerate(claims_per_batch):
                        claims_text += f'("{c[0]}", "{c[1]}", "{c[2]}")\n'
                        _claim_cnt += 1
                        if _claim_cnt >= joint_check_num or _ci == len(claims_per_batch) - 1:
                            prompt = prompt_template.replace('[QUESTION]', question_per_batch)
                            prompt = prompt.replace('[REFERENCE]', ref)
                            prompt = prompt.replace('[CLAIMS]', claims_text.strip())
                            prompt_list.append(prompt)
                            
                            prompt_ids.append(p_id)
                            claim_nums.append(_claim_cnt)
                            _claim_cnt = 0
                            claims_text = ''
                            
                    p_id += 1
            
            labels_list = []
            for i in tqdm(range(0, len(prompt_list), self.batch_size)):
                batch_prompts = prompt_list[i:i + self.batch_size]

                llm_responses = get_model_batch_response(
                    prompts=batch_prompts,
                    temperature=0,
                    model=self.model,
                    max_new_tokens=joint_check_num * 10 + 100,
                    api_base=self.api_base,
                    custom_llm_api_func=custom_llm_api_func,
                    **kwargs
                )
                
                for llm_response in llm_responses:
                    if llm_response is not None:
                        labels = self._parse_joint_checking_labels(llm_response)
                        labels_list.append(labels)
                    else:
                        raise 'API returns None or empty string'
            
            # pad labels with Neutral
            assert len(claim_nums) == len(labels_list)
            for _i, claim_n in enumerate(claim_nums):
                if len(labels_list[_i]) < claim_n:
                    labels_list[_i] = labels_list[_i] + ['Neutral'] * (claim_n - len(labels_list[_i]))
                elif len(labels_list[_i]) > claim_n:
                    labels_list[_i] = labels_list[_i][:claim_n]
            # merge labels
            merged_label_list = []
            for _i, _pid in enumerate(prompt_ids):
                if _i > 0 and _pid == prompt_ids[_i - 1]:
                    merged_label_list[-1] += labels_list[_i]
                else:
                    merged_label_list.append(labels_list[_i])
            
            ret_labels = []
            _index = 0
            for _i, claim_num in enumerate(batch_claim_nums):
                if claim_num > 0:
                    one_batch_labels = merged_label_list[_index: _index + batch_ref_nums[_i]] # [ref_num, claim_num]

                    _index += batch_ref_nums[_i]
                    
                    one_batch_labels = np.array(one_batch_labels).transpose(1, 0)
                    # if batch_ref_nums[_i] == 1:
                    #     one_batch_labels = one_batch_labels.squeeze(-1)
                    ret_labels.append(one_batch_labels.tolist())
                else:
                    ret_labels.append([])
            return ret_labels
        else:
            ret_labels = []
            prompt_list = []
            for claim, reference, question in zip(claims, references, questions):
                claim_text = str(claim)
                
                if isinstance(claim, list) and len(claim) == 3:
                    if question is None:
                        prompt = self.prompt_temp.format(
                            reference=reference,
                            claim=claim_text
                        )
                    else:
                        prompt = self.prompt_temp_wq.format(
                            question=question,
                            reference=reference,
                            claim=claim_text
                        )
                elif isinstance(claim, str):
                    if question and len(question):
                        reference = question + ' ' + reference
                    prompt = self.prompt_temp_subsent.format(
                        reference=reference,
                        claim=claim_text
                    )
                else:
                    raise f'Unknown claim format: {type(claim)}'
                prompt_list.append(prompt)

            for i in tqdm(range(0, len(prompt_list), self.batch_size)):
                batch_prompts = prompt_list[i:i + self.batch_size]

                llm_responses = get_model_batch_response(
                    prompts=batch_prompts,
                    temperature=0,
                    model=self.model,
                    max_new_tokens=10,
                    api_base=self.api_base,
                    custom_llm_api_func=custom_llm_api_func,
                    **kwargs
                )
                
                for llm_response in llm_responses:
                    if llm_response and len(llm_response):
                        label = None
                        if self.label_contradiction.lower() in llm_response.lower():
                            label = self.label_contradiction
                        elif self.label_entailment.lower() in llm_response.lower():
                            label = self.label_entailment
                        else:
                            label = self.label_neutral
                        ret_labels.append(label)
                    else:
                        raise 'API returns None or empty string'
            return ret_labels

    def _parse_joint_checking_labels(self, text):
        pattern = r'\b(Entailment|Neutral|Contradiction)\b'
        matches = re.findall(pattern, text, re.IGNORECASE)
        parsed_labels = [label.title() for label in matches]
        return parsed_labels