from typing import List
from tqdm import tqdm

from .extractor_base import ExtractorBase
from ..utils import (
    get_model_batch_response, 
    get_llm_full_name
)
from ..base import RCText, ExtractionResult
from .extractor_prompts import *


# LLM_CLAIM_EXTRACTION_PROMPT = """You are an AI assistant, you can help to extract claims from a given text. In addition, you should attribute the claims to the sentences followed by the sentence ids.
# Each claim should satisfy the following criteria:
# * A claim is a piece of `knowledge` in the text, the basic element of a `knowledge` is a triplet `(head entity, relation, tail entity)` used in Knowledge Graph, or `(subject, predicate, object)` used in Semantics.
# * A claim should be fine-grained. One claim should not contain more than one pieces of knowledge.
# * A claim should be self-contained, it is not dependent on other claims.
# * Each claim should truly reflect the meaning to be expressed in the text, and the information in the claim should be complete and unambiguous, and necessary conditions and attributes should not be missed. For example, for the text "Donald Trump won the presidential election in 2016.", the claim "Donald Trump won the presidential election" is a bad claim where it misses necessary information "in 2016", so a complete claim should be "Donald Trump won the presidential election in 2016".
# * Some sentence in the text may not contain claims. Some sentence may contain one or more claims. A claim may occur across multiple sentences.
# * Opinions, speculations or questions in the text are not claims.

# We have added sentence IDs at the beginning of the sentences in the text, you should output a claim followed by a list of IDs, and these IDs stand for the sentences this claim attribute to. The extraction process:
# 1. You should first identify whether there are claims in the text. If the text says something like "I don't know", or refuse to answer the question, then you can consider the text is abstaining from answering some question, you should output "Abstain".
# 2. If there are claims in the text, for each claim, you should identify a (subject, predicate, object) structure as the basic elements of a claim. The subject and object are entities, and the predicate is the relationship or event happened between them.
# 3. Note that the (subject, predicate, object) structure may just reflect partial information in the text, so if we take it in isolation, it may not reflect the original meaning in the text, you should fill in the missing information to make it complete and clear. The information in a claim may come across multiple sentences in the text.
# 4. Formulate the claim into one sentence, and followed by the sentence indices as the attributed sentences in the text.
# 5. Make sure you have generated all the claims in the text.

# Please note that this is an EXTRACTION task, so DO NOT care about whether the content of the text is factual or not, just extract the claims from it.

# Here are some examples:

# ### Text
# [1] Optimus (or Tesla Bot) is a robotic humanoid under development by Tesla, Inc. [2] It was announced at the company's Artificial Intelligence (AI) Day event on August 19, 2021.
# ### Claims
# Optimus is a robotic humanoid. [1]
# Optimus is under development by Tesla, Inc. [1]
# Optimus is also known as Tesla Bot. [1]
# Tesla, Inc. announced Optimus at the Artificial Intelligence (AI) Day event August 19, 2021. [2]

# ### Text
# here is some text about Andre Weiss, how many years was Andre at University of Dijon in Paris? [1] 11 years
# ### Claims
# Andre Weiss was at University of Dijon in Paris for 11 years. [1]

# ### Text
# who sang i ran all the way home? [1] I'm sorry, I cannot provide an accurate answer without additional context. [2] Could you please provide more information?
# ### Claims
# Abstain

# ### Text
# does jimmy jazz sell fake shoes [1] Based on the provided passages, it appears that Jimmy Jazz may sell fake shoes. [2] Passage 0 and Passage 1 both mention that the shoes purchased from Jimmy Jazz were found to be fake, and Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [3] Additionally, Passage 4 mentions that Kanye West, who has collaborated with Jimmy Jazz in the past, has signed fake sneakers, which could suggest that Jimmy Jazz may be involved in selling counterfeit products. 
# [4] However, it's important to note that there are also passages that do not mention anything about fake shoes, such as Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9. [5] These passages do not provide any information that would confirm or deny the suspicion of Jimmy Jazz selling fake shoes. 
# [6] Therefore, based on the provided passages, it is possible that Jimmy Jazz may sell fake shoes, but there is not enough conclusive evidence to confirm this suspicion.
# ### Claims
# Passage 0 and Passage 1 mention that shoes purchased from Jimmy Jazz were found to be fake. [2]
# Passage 3 is a Ripoff Report stating that Jimmy Jazz E-Commerce LLC has a history of late shipping and lies on their website. [2]
# Passage 4 mentions that Kanye West has collaborated with Jimmy Jazz in the past. [3]
# Passage 4 mentions that Kanye West has signed fake sneakers. [3]
# Passage 2, Passage 5, Passage 6, Passage 7, Passage 8, and Passage 9 do not mention anything about fake shoes. [4]

# ### Text
# calories in ground beef stroganoff [1] Sure! [2] Based on the provided passages, the answer to the question "calories in ground beef stroganoff?" [3] is 227 calories. [4] This information can be found in Passage 9.
# ### Claims
# Passage 9 states that ground beef stroganoff contains 227 calories. [2][3][4]

# Now please generate the claims from the following text. You should always follow the output format shown in the above examples and just ouput the claims without any extra information:
# ### Text
# {text}
# ### Claims
# """


class LLMExtractor(ExtractorBase):
    def __init__(
        self, 
        claim_format: str = 'subsentence',
        model: str = 'claude3-sonnet',
        batch_size=16
    ) -> None:
        super().__init__(claim_format)
        
        self.model = get_llm_full_name(model)
        self.batch_size = batch_size
    
    def extract_subsentence_claims(
        self, 
        response, 
        question=None, 
        max_new_tokens=500
    ):
        """Extract subsentence claims from the response text.
        Parameters
        ----------
        response : List[str]
            List of model response texts.
        question : List[str|None] | None
            List of questions corresponding to each response.
        max_new_tokens : int, optional
            Maximum number of tokens to generate, defaults to 500.
        Returns
        -------
        List[ExtractionResult]
            List of extracted claims for each response.
        """
        
        prompt_list = []
        result_list = []
        rc_responses = []
        for _i, r in enumerate(response):
            rc_r = RCText(r)
            indexed_r_text = rc_r.get_indexed_response(condense_newlines=True)
            q = None
            if question:
                q = question[_i]
            if q and len(q):
                prompt = LLM_CLAIM_EXTRACTION_PROMPT_Q.format(q=q, r=indexed_r_text)
            else:
                raise NotImplementedError
            prompt_list.append(prompt)
            rc_responses.append(rc_r)
        
        for _i in tqdm(range(0, len(prompt_list), self.batch_size)):
            batch_prompts = prompt_list[_i:_i+self.batch_size]
            llm_responses = get_model_batch_response(
                prompts=batch_prompts,
                temperature=0,
                model=self.model,
                n_choices=1,
                max_new_tokens=max_new_tokens
            )

            if llm_responses and len(llm_responses):
                for _j, res in enumerate(llm_responses):
                    claims = self.parse_claims(res, excluded_content_prefix='### Text')
                    result = ExtractionResult(
                        claims=claims,
                        question=question,
                        response=rc_responses[_i + _j],
                        extractor_response=res,
                    )
                    result_list.append(result)
            else:
                return None
        return result_list

    def extract_claim_triplets(
        self, 
        response, 
        question=None, 
        max_new_tokens=500
    ):
        """Extract KG triplets from the response text.
        Parameters
        ----------
        response : List[str]
            List of model response texts.
        question : List[str|None] | None
            List of questions corresponding to each response.
        max_new_tokens : int, optional
            Maximum number of tokens to generate, defaults to 500.
        Returns
        -------
        List[ExtractionResult]
            List of extracted claims for each response.
        """
        
        prompt_list = []
        result_list = []
        
        for _i, r in enumerate(response):
            q = None
            if question:
                q = question[_i]
            if q is None:
                prompt = LLM_TRIPLET_EXTRACTION_PROMPT.format(
                    input_text=response
                )
            else:
                prompt = LLM_TRIPLET_EXTRACTION_PROMPT_Q.format(
                    q=q,
                    a=response
                )
            prompt_list.append(prompt)
        
        for _i in tqdm(range(0, len(prompt_list), self.batch_size)):
            batch_prompts = prompt_list[_i:_i+self.batch_size]
            llm_responses = get_model_batch_response(
                prompts=batch_prompts,
                temperature=0,
                model=self.model,
                n_choices=1,
                max_new_tokens=max_new_tokens
            )

            error_info = None
            if llm_responses and len(llm_responses):
                for res in llm_responses:
                    claims = self.parse_claims(res, '###')
                    result = ExtractionResult(
                        claims=claims,
                        question=question,
                        response=response,
                        extractor_response=res,
                    )
                    result_list.append(result)
            else:
                return None
        return result_list