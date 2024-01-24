from .extractor_base import ExtractorBase

import torch
from ..utils import get_response_from_mistral
from typing import Optional, Literal

MISTRAL_KG_EXTRACTION_PROMPT_Q = """Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("head", "relation", "tail"), each triplet in a line.
Please note that this is an extraction task, so do not care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

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

MISTRAL_KG_EXTRACTION_PROMPT = """Given an input text, please extract a KG from the text and represent the KG with triples formatted with ("head", "relation", "tail"), each triplet in a line. Please note that this is an extraction task, so do not care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

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


class MistralExtractor(ExtractorBase):
    def __init__(
        self,
        claim_format: str = "triplet",
        model_name: Literal["mixtral", "mistral"] = "mixtral",
        model_path: Optional[str] = None,
        use_gpu_num: Optional[int] = None,
    ) -> None:
        super().__init__(claim_format=claim_format)
        if self.claim_format == "triplet":
            self.prompt_temp_wq = MISTRAL_KG_EXTRACTION_PROMPT_Q
            self.prompt_temp = MISTRAL_KG_EXTRACTION_PROMPT

        self.model_name = model_name
        self.model_path = model_path
        self.use_gpu_num = use_gpu_num

    def extract_claim_triplets(self, response, question=None, max_new_tokens=500):
        if question is None:
            prompt = self.prompt_temp.format(input_text=response)
        else:
            prompt = self.prompt_temp_wq.format(q=question, a=response)
        mistral_response = get_response_from_mistral(
            prompt,
            max_new_tokens=max_new_tokens,
            model_name=self.model_name,
            model_path=self.model_path,
            use_gpu_num=self.use_gpu_num,
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
                question=example_for_test["question"],
            ),
            indent=4,
        )
    )

    print(
        json.dumps(
            extractor.extract_claim_triplets(response=example_for_test["response"]),
            indent=4,
        )
    )
