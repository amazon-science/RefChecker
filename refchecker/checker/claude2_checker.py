from typing import List

from .checker_base import CheckerBase
from ..utils import get_claude2_response


CLAUDE2_CHECKING_PROMPT_Q = \
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

CLAUDE2_CHECKING_PROMPT = \
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


class Claude2Checker(CheckerBase):
    def __init__(self) -> None:
        super().__init__()
        self.prompt_temp = CLAUDE2_CHECKING_PROMPT
        self.prompt_temp_wq = CLAUDE2_CHECKING_PROMPT_Q
    
    def _check(
        self, 
        claims: List, 
        references: List,
        response: str,
        question: str,
    ):
        ret_labels = []
        for claim, reference in zip(claims, references):
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
            claude2_response = get_claude2_response(
                prompt=prompt,
                temperature=0,
                max_new_tokens=6
            )
            if claude2_response and len(claude2_response):
                label = None
                if self.label_contradiction.lower() in claude2_response.lower():
                    label = self.label_contradiction                    
                elif self.label_entailment.lower() in claude2_response.lower():
                    label = self.label_entailment
                else:
                    label = self.label_neutral
                ret_labels.append(label)
            else:
                raise 'Claude 2 API returns None or empty string'
        return ret_labels
