import os

from .extractor_base import ExtractorBase
from ..utils import get_model_batch_response


LLM_TRIPLET_EXTRACTION_PROMPT_Q = \
"""Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("subject", "predicate", "object").
Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples:

### Question:
Given these paragraphs about the Tesla bot, what is its alias?

### Candidate Answer:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occurred at", "Artificial Intelligence (AI) Day event")
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

LLM_TRIPLET_EXTRACTION_PROMPT = \
"""Given an input text, please extract a KG from the text and represent the KG with triples formatted with ("subject", "predicate", "object"), each triplet in a line. Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the candidate answer is factual or not, just extract the triplets from it.

Here are some in-context examples:

### Input:
Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.

### KG:
("Optimus", "is", "robotic humanoid")
("Optimus", "under development by", "Tesla, Inc.")
("Optimus", "also known as", "Tesla Bot")
("Tesla, Inc.", "announced", "Optimus")
("Announcement of Optimus", "occurred at", "Artificial Intelligence (AI) Day event")
("Artificial Intelligence (AI) Day event", "held on", "August 19, 2021")
("Artificial Intelligence (AI) Day event", "organized by", "Tesla, Inc.")

### Input:
The song "Here Comes the Boom" was originally released by American rock band Nelly in 2002 for the soundtrack of the film "The Longest Yard."

### KG:
("The song 'Here Comes the Boom'", "originally released by", "American rock band Nelly")
("The song 'Here Comes the Boom'", "released in", "2002")
("The song 'Here Comes the Boom'", "featured in", "soundtrack of the film 'The Longest Yard'")
("American rock band Nelly", "released", "The song 'Here Comes the Boom'")
("The Longest Yard", "had soundtrack featuring", "The song 'Here Comes the Boom'")


Now generate the KG for the provided input text:

### Input:
{input_text}

### KG:
"""


class LLMExtractor(ExtractorBase):
    def __init__(
        self,
        claim_format:str='triplet',
        model='claude3-sonnet'
    ) -> None:
        super().__init__(claim_format=claim_format)
        if self.claim_format == 'triplet':
            self.prompt_temp_wq = LLM_TRIPLET_EXTRACTION_PROMPT_Q
            self.prompt_temp = LLM_TRIPLET_EXTRACTION_PROMPT
            
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
    
    def extract_claim_triplets(self, response, question=None, max_new_tokens=500):
        if question is None:
            prompt = self.prompt_temp.format(
                input_text=response
            )
        else:
            prompt = self.prompt_temp_wq.format(
                q=question,
                a=response
            )
        response = get_model_batch_response(
            [prompt],
            temperature=0,
            model=self.model,
            max_new_tokens=max_new_tokens
        )[0]
        if response and len(response):
            kg_str = None
            if '###' in response:
                kg_str = response[:response.index('###')]
            else:
                kg_str = response
            triplets = self._parse_claim_triplets(kg_str)
            return triplets
        return []