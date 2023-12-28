from .extractor_base import ExtractorBase

import torch
from vllm import LLM, SamplingParams
from typing import List, Tuple, Union

one_digit_tensor = torch.ones((1, 1), dtype=torch.long)

MISTRAL_KG_EXTRACTION_PROMPT_Q = """Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("head", "relation", "tail"), each triplet in a line.
Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples of extraction:

### Question:
Given these paragraphs about the Tesla bot, what is its alias?

### Candidate Answer:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occured at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")

### Question:
here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?

### Candidate Answer:
11 years

### KG:
("Andre Weiss at University of Dijon in Paris", "duration", "11 years")

Now generate the KG for the following candidate answer based on the provided question:

### Question:
{q}

### Candidate Answer:
{a}

### KG:
"""

MISTRAL_KG_EXTRACTION_PROMPT = """Given an input text, please extract a KG from the text and represent the KG with triples formatted with ("head", "relation", "tail"), each triplet in a line. Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples of extraction:

### Input:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occured at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")

### Input:
Question: here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris?
Answer: 11 years

### KG:
("Andre Weiss at University of Dijon in Paris", "duration", "11 years")

Now generate the KG for the following input text based on the provided question:

### Input Text:
{input_text}

### KG:
"""


class MistralClaimExtractor(ExtractorBase):
    def __init__(
        self,
        claim_format: str = "triplet",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        load_format="safetensors",
        tensor_parallel_size=-1,
        temperature=0.0,
        max_new_tokens=2048,
    ) -> None:
        self.llm = LLM(
            model,
            load_format=load_format,
            tensor_parallel_size=torch.cuda.device_count()
            if tensor_parallel_size < 0
            else tensor_parallel_size,
            trust_remote_code=True,
        )
        self.sampling_params = SamplingParams(
            temperature=temperature, max_tokens=max_new_tokens
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.llm.set_tokenizer(self.tokenizer)
        self.claim_format = claim_format

        if self.claim_format == "triplet":
            self.prompt_temp_wq = MISTRAL_KG_EXTRACTION_PROMPT_Q
            self.prompt_temp = MISTRAL_KG_EXTRACTION_PROMPT

    def _mistral_encode_conversation(
        self, prompt, answer_history: List[Tuple[str, str]] = []
    ):
        """Encode conversations for Mixtral.
        answer_hisory: [(u_1, a_1), (u_2, a_2), ... (u_k, a_k)]
        Assume all strings are .strip()-ed and does not include anything like [INST][/INST]
        """
        tokenizer = self.tokenizer
        final_prompted_tensors = []

        for i in range(len(answer_history)):
            user_prompt, model_output = answer_history[i]
            for_tokenize_prompt = f"[INST] {user_prompt} [/INST] {model_output}"
            tokenized_id = tokenizer.encode(
                for_tokenize_prompt, add_special_tokens=False, return_tensors="pt"
            )
            final_prompted_tensors.append(tokenized_id)
            final_prompted_tensors[-1] = torch.cat(
                (final_prompted_tensors[-1], one_digit_tensor * tokenizer.eos_token_id),
                dim=-1,
            )

        # last prompt
        for_tokenize_prompt = f"[INST] {prompt} [/INST]"
        tokenized_id = tokenizer.encode(
            for_tokenize_prompt, add_special_tokens=False, return_tensors="pt"
        )
        final_prompted_tensors.append(tokenized_id)

        # BOS
        final_prompted_tensors[0] = torch.cat(
            (one_digit_tensor * tokenizer.bos_token_id, final_prompted_tensors[0]),
            dim=-1,
        )

        final_prompted_tensor = torch.cat(final_prompted_tensors, dim=-1)
        return final_prompted_tensor

    def _get_response_from_mistral(
        self, raw_prompt: str, conversation_history: List[str] = []
    ):
        llm = self.llm
        outputs = llm.generate(
            prompt_token_ids=self._mistral_encode_conversation(
                raw_prompt, conversation_history
            ).tolist(),
            sampling_params=self.sampling_params,
            use_tqdm=False,
        )
        llm_output = outputs[0].outputs[0].text
        conversation_history = conversation_history.copy()
        conversation_history.append((raw_prompt, llm_output))
        return llm_output, conversation_history

    def extract_claim_triplets(self, response, question=None):
        if question is None:
            prompt = self.prompt_temp.format(input_text=response)
        else:
            prompt = self.prompt_temp_wq.format(q=question, a=response)
        mistral_response, _ = self._get_response_from_mistral(prompt)
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

    extractor = MistralClaimExtractor()
    example_for_test_q = {
        "id": "192218",
        "src": "msmarco/msmarco_chatgpt_answers.json",
        "question": "full time student how many hours",
        "response": 'Based on the provided passages, the number of hours required to be considered a full-time student can vary depending on the context. However, some common requirements mentioned are:\n\n- Passage 0: Full-time status is usually considered as a schedule of 12 or more semester or quarter hours, but there is no data on how many credits full-time students are actually taking.\n\n- Passage 1: For undergraduate students in the summer, 12 hours is considered full-time.\n\n- Passage 2: For fall and spring semesters, a full-time college student completes at least 12 semester hours. Some schools may require 15 semester hours. In the summer, completing 6 semester hours is considered full-time.\n\n- Passage 3: For graduate students, a normal full-time load is nine graduate-level semester hours. But for those holding graduate assistantships, full-time status is six semester hours.\n\n- Passage 4: According to the University Bulletin, graduate students taking 9 or more credit hours per semester (6 credits in the summer) are considered full-time.\n\n- Passage 5: The normal load for full-time students is 3 courses (9 credits).\n\n- Passage 8: For graduate students, 9 hours is considered full-time.\n\nBased on the information provided, the answer to the question "full time student how many hours?" is not explicitly stated or consistent across all passages. It can range from 12 hours for undergraduate students in the summer to 9 or more credit hours for graduate students. Some schools may require 15 semester hours for full-time status.',
    }

    example_for_test = example_for_test_q.copy()
    example_for_test[
        "input"
    ] = f"Question: {example_for_test_q['question']}\nAnswer: {example_for_test_q['response']}"

    print(
        json.dumps(
            extractor.extract_claim_triplets(
                example_for_test_q["response"], example_for_test_q["question"]
            ),
            ensure_ascii=False,
            indent=4,
        )
    )

    print(
        json.dumps(
            extractor.extract_claim_triplets(example_for_test["input"]),
            ensure_ascii=False,
            indent=4,
        )
    )
