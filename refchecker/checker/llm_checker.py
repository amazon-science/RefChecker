import os
from typing import List, Union
from tqdm import tqdm

from .checker_base import CheckerBase
from ..utils import get_model_batch_response


LLM_CHECKING_PROMPT_Q = \
"""I have a claim that made by a language model to a question, please help me for checking whether the claim can be entailed according to the provided reference which is related to the question. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Question:
{question}

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""

LLM_CHECKING_PROMPT = \
"""I have a claim that made by a language model, please help me for checking whether the claim can be entailed according to the provided reference. 
The reference is a list of passages, and the claim is represented as a triplet formatted with ("subject", "predicate", "object").

If the claim is supported by ANY passage in the reference, answer 'Entailment'. 
If NO passage in the reference entail the claim, and the claim is contradicted with some passage in the reference, answer 'Contradiction'.
If NO passage entail or contradict with claim, or DOES NOT contain information to verify the claim, answer 'Neutral'. 

Please DO NOT use your own knowledge for the judgement, just compare the reference and the claim to get the answer.

### Reference:
{reference}

### Claim:
{claim}

Your answer should always be only a single word in ['Entailment', 'Neutral', 'Contradiction']. DO NOT add explanations or you own reasoning to the output.
"""


class LLMChecker(CheckerBase):
    def __init__(
        self,
        model,
        batch_size=16
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
        self.batch_size = batch_size
        if model not in ['gpt4', 'claude2', 'claude3-sonnet', 'claude3-haiku']:
            self.model = model
        elif model == 'gpt4':
            self.model = 'gpt-4'
        elif model == 'claude2':
            self.model = 'bedrock/anthropic.claude-v2' if os.environ.get('AWS_REGION_NAME') else 'claude-2'
        elif model == 'claude3-sonnet':
            self.model = 'anthropic.claude-3-sonnet-20240229-v1:0' if os.environ.get('AWS_REGION_NAME') else 'claude-3-sonnet-20240229'
        elif model == 'claude3-haiku':
            self.model = 'anthropic.claude-3-haiku-20240307-v1:0' if os.environ.get('AWS_REGION_NAME') else 'claude-3-haiku-20240307'
        else:
            raise ValueError('The model you specified is not supported.')

    def _check(
        self,
        claims: List[Union[str, List[str]]],
        references: List[str],
        responses: List[str],
        questions: List[str],
    ):
        """
        Batch checking claims against references.

        Parameters
        ----------
        claims : List[Union[str, List[str]]]
            List of claim triplets.
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

        ret_labels = []
        prompt_list = []
        for claim, reference, question in zip(claims, references, questions):
            if isinstance(claim, list):
                assert len(claim) == 3
                claim = f"({claim[0]}, {claim[1]}, {claim[2]})"
            if question is None:
                prompt = self.prompt_temp.format(
                    reference=reference,
                    claim=claim
                )
            else:
                prompt = self.prompt_temp_wq.format(
                    question=question,
                    reference=reference,
                    claim=claim
                )
            prompt_list.append(prompt)

        for i in tqdm(range(0, len(prompt_list), self.batch_size)):
            batch_prompts = prompt_list[i:i + self.batch_size]
            llm_responses = get_model_batch_response(
                prompts=batch_prompts,
                temperature=0,
                model=self.model,
                max_new_tokens=10,
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
