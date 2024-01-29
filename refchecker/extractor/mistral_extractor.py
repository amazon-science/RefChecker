from .extractor_base import ExtractorBase

import torch
from vllm import LLM, SamplingParams


one_digit_tensor = torch.ones((1, 1), dtype=torch.long)

MISTRAL_KG_EXTRACTION_PROMPT_Q = """Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("head", "relation", "tail"), each triplet in a line.
Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.
Now generate the KG for the following candidate answer based on the provided question:

### Question:
{q}

### Candidate Answer:
{a}

### KG:
"""


class MistralExtractor(ExtractorBase):
    def __init__(
        self,
        claim_format: str = "triplet",
        tensor_parallel_size=-1,
    ) -> None:
        super().__init__(claim_format=claim_format)
        self.llm = LLM(
            "dongyru/Mistral-7B-Claim-Extractor",
            load_format="safetensors",
            tensor_parallel_size=torch.cuda.device_count()
            if tensor_parallel_size < 0
            else tensor_parallel_size,
            trust_remote_code=True,
        )
        self.tokenizer = self.llm.get_tokenizer()
        if self.claim_format == "triplet":
            self.prompt_temp_wq = MISTRAL_KG_EXTRACTION_PROMPT_Q
            self.prompt_temp = None


    # TODO move this to utils for general LLMs based on vllm interface
    def _get_response_from_mistral(
        self, raw_prompt: str, max_new_tokens: int = 2048
    ):
        llm = self.llm
        tokenizer = self.tokenizer
        prompt_token_ids = tokenizer.encode(f"[INST] {raw_prompt} [/INST]")
        outputs = llm.generate(
            prompt_token_ids=[prompt_token_ids],
            sampling_params=SamplingParams(
                temperature=0., max_tokens=max_new_tokens
            ),
            use_tqdm=False,
        )
        llm_output = outputs[0].outputs[0].text
        return llm_output

    def extract_claim_triplets(
        self, response, question=None, max_new_tokens=2048
    ):
        if question is None:
            raise NotImplementedError(
                "Question is required for MistralExtractor"
            )
        else:
            prompt = self.prompt_temp_wq.format(q=question, a=response)
        mistral_response = self._get_response_from_mistral(
            prompt, max_new_tokens=max_new_tokens
        )
        if mistral_response and len(mistral_response):
            kg_str = None
            if "###" in mistral_response:
                kg_str = mistral_response[: mistral_response.index("###")]
            else:
                kg_str = mistral_response
            triplets = self._parse_claim_triplets(kg_str)
            return triplets
        return []


if __name__ == "__main__":
    import json

    extractor = MistralExtractor()
    example_for_test = {
        "question": "full time student how many hours",
        "response": 'Based on the provided passages, the number of hours required to be considered a full-time student can vary depending on the context. However, some common requirements mentioned are:\n\n- Passage 0: Full-time status is usually considered as a schedule of 12 or more semester or quarter hours, but there is no data on how many credits full-time students are actually taking.\n\n- Passage 1: For undergraduate students in the summer, 12 hours is considered full-time.\n\n- Passage 2: For fall and spring semesters, a full-time college student completes at least 12 semester hours. Some schools may require 15 semester hours. In the summer, completing 6 semester hours is considered full-time.\n\n- Passage 3: For graduate students, a normal full-time load is nine graduate-level semester hours. But for those holding graduate assistantships, full-time status is six semester hours.\n\n- Passage 4: According to the University Bulletin, graduate students taking 9 or more credit hours per semester (6 credits in the summer) are considered full-time.\n\n- Passage 5: The normal load for full-time students is 3 courses (9 credits).\n\n- Passage 8: For graduate students, 9 hours is considered full-time.\n\nBased on the information provided, the answer to the question "full time student how many hours?" is not explicitly stated or consistent across all passages. It can range from 12 hours for undergraduate students in the summer to 9 or more credit hours for graduate students. Some schools may require 15 semester hours for full-time status.',
    }

    print(
        json.dumps(
            extractor.extract_claim_triplets(
                response=example_for_test["response"],
                question=example_for_test["question"]
            ),
            indent=4,
        )
    )
