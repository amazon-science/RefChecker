from typing import List

from .extractor_base import ExtractorBase
from ..utils import (
    get_model_batch_response, 
    get_llm_full_name
)
from ..base import RCText, ExtractionResult


LLM_CLAIM_EXTRACTION_PROMPT = """You are an AI assistant, you can help to extract claims from a given text. In addition, you should attribute the claims to the sentences followed by the sentence ids.
Each claim should satisfy the following criteria:
* A claim is a piece of `knowledge` in the text, the basic element of a `knowledge` is a triplet `(head entity, relation, tail entity)` used in Knowledge Graph, or `(subject, predicate, object)` used in Semantics.
* A claim should be fine-grained. One claim should not contain more than one pieces of knowledge.
* A claim should be self-contained, it is not dependent on other claims.
* Each claim should truly reflect the meaning to be expressed in the text, and the information in the claim should be complete and unambiguous, and necessary conditions and attributes should not be missed. For example, for the text "Donald Trump won the presidential election in 2016.", the claim "Donald Trump won the presidential election" is a bad claim where it misses necessary information "in 2016", so a complete claim should be "Donald Trump won the presidential election in 2016".
* Some sentence in the text may not contain claims. Some sentence may contain one or more claims. A claim may occur across multiple sentences.
* Opinions, speculations or questions in the text are not claims.

We have added sentence IDs at the beginning of the sentences in the text, you should output a claim followed by a list of IDs, and these IDs stand for the sentences this claim attribute to. The extraction process:
1. You should first identify whether there are claims in the text. If the text says something like "I don't know", or refuse to answer the question, then you can consider the text is abstaining from answering some question, you should output "Abstain".
2. If there are claims in the text, for each claim, you should identify a (subject, predicate, object) structure as the basic elements of a claim. The subject and object are entities, and the predicate is the relationship or event happened between them.
3. Note that the (subject, predicate, object) structure may just reflect partial information in the text, so if we take it in isolation, it may not reflect the original meaning in the text, you should fill in the missing information to make it complete and clear. The information in a claim may come across multiple sentences in the text.
4. Formulate the claim into one sentence, and followed by the sentence indices as the attributed sentences in the text.
5. Make sure you have generated all the claims in the text.

Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the text is factual or not, just extract the claims from it.

Here are some examples:

### Text
[1] Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. [2] It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.
### Claims
Optimus is a robotic humanoid. [1]
Optimus is under development by Tesla, Inc. [1]
Optimus is also known as Tesla Bot. [1]
Tesla, Inc. announced Optimus at the Artificial Intelligence (AI) Day event August 19, 2021. [2]

### Text
here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris? [1] 11 years
### Claims
Andre Weiss was at University of Dijon in Paris for 11 years. [1]

### Text
who sang i ran all the way home? [1] I'm sorry, I cannot provide an accurate answer without additional context. [2] Could you please provide more information?
### Claims
Abstain

### Text
does jimmy jazz sell fake shoes [1] Based on the provided passages, it appears that Jimmy Jazz may sell fake shoes. [2] Passage 0 and Passage 1 both mention that the shoes purchased from Jimmy Jazz were found to be fake, and Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [3] Additionally, Passage 4 mentions that Kanye West, who has collaborated with Jimmy Jazz in the past, has signed fake sneakers, which could suggest that Jimmy Jazz may be involved in selling counterfeit products. 
[4] However, it's important to note that there are also passages that do not mention anything about fake shoes, such as Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9. [5] These passages do not provide any information that would confirm or deny the suspicion of Jimmy Jazz selling fake shoes. 
[6] Therefore, based on the provided passages, it is possible that Jimmy Jazz may sell fake shoes, but there is not enough conclusive evidence to confirm this suspicion.
### Claims
Passage 0 and Passage 1 mention that shoes purchased from Jimmy Jazz were found to be fake. [2]
Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [2]
Passage 4 mentions that Kanye West has collaborated with Jimmy Jazz in the past. [3]
Passage 4 mentions that Kanye West has signed fake sneakers. [3]
Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9 do not mention anything about fake shoes. [4]

### Text
calories in ground beef stroganoff [1] Sure! [2] Based on the provided passages, the answer to the question "calories in ground beef stroganoff?" [3] is 227 calories. [4] This information can be found in Passage 9.
### Claims
Passage 9 states that ground beef stroganoff contains 227 calories. [2][3][4]

Now please generate the claims from the following text. You should always follow the output format shown in the above examples and just ouput the claims without any extra information:
### Text
{text}
### Claims
"""

LLM_TRIPLET_EXTRACTION_PROMPT_Q = \
"""Given a question and a candidate answer to the question, please extract a KG from the candidate answer condition on the question and represent the KG with triples formatted with ("subject", "predicate", "object"), each triplet in a line.
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
        claim_format: str = 'subsentence',
        model: str = 'claude3-sonnet'
    ) -> None:
        super().__init__(claim_format)
        
        self.model = get_llm_full_name(model)
    
    def extract_subsentence_claims(
        self, 
        response, 
        question=None, 
        max_new_tokens=500
    ):
        rc_response = RCText(response)
        indexed_response_text = rc_response.get_indexed_response(condense_newlines=True)
        if question and len(question):
            indexed_response_text = question + ' ' + indexed_response_text
        prompt = LLM_CLAIM_EXTRACTION_PROMPT.format(text=indexed_response_text)
        
        claude_response = get_model_batch_response(
            prompts=[prompt],
            temperature=0,
            model=self.model,
            n_choices=1,
            max_new_tokens=max_new_tokens
        )[0]
        
        claims = None
        error_info = None
        if claude_response and len(claude_response):
            claims = self.parse_claims(claude_response, excluded_content_prefix='### Text')
        else:
            error_info = 'Something wrong with the Extractor API'
        result = ExtractionResult(
            claims=claims,
            question=question,
            response=rc_response,
            extractor_response=claude_response,
            error_info=error_info
        )
        return result

    def extract_claim_triplets(
        self, 
        response, 
        question=None, 
        max_new_tokens=500
    ):
        if question is None:
            prompt = LLM_TRIPLET_EXTRACTION_PROMPT.format(
                input_text=response
            )
        else:
            prompt = LLM_TRIPLET_EXTRACTION_PROMPT_Q.format(
                q=question,
                a=response
            )

        claude_response = get_model_batch_response(
            prompts=[prompt],
            temperature=0,
            model=self.model,
            n_choices=1,
            max_new_tokens=max_new_tokens
        )[0]

        error_info = None
        if claude_response and len(claude_response):
            claims = self.parse_claims(claude_response, '###')
        else:
            error_info = 'Something wrong with the Extractor API'
        result = ExtractionResult(
            claims=claims,
            question=question,
            response=response,
            extractor_response=claude_response,
            error_info=error_info
        )
        return result