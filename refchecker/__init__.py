from .checker import (
    LLMChecker,
    NLIChecker, 
    AlignScoreChecker,
    RepCChecker
)
from .extractor import (
    Claude2Extractor, 
    GPT4Extractor, 
    MixtralExtractor,
    MistralExtractor
)


class RefChecker:
    def __init__(
        self,
        claim_extractor_model:str='claude2',
        claim_format:str='triplet',
        checker_model:str='claude2'
    ) -> None:
        self.claim_format = claim_format
        if self.claim_format != 'triplet':
            raise 'We currently only support claim-triplet'
        
        if claim_extractor_model == 'claude2':
            self.claim_extractor = Claude2Extractor(claim_format=claim_format)
        elif claim_extractor_model == 'gpt4':
            self.claim_extractor = GPT4Extractor(claim_format=claim_format)
        
        if checker_model == 'claude2':
            self.bc = Claude2Checker()
            self.max_reference_segment_length = 0
        elif checker_model == 'gpt4':
            self.bc = GPT4Checker()
            self.max_reference_segment_length = 0
        elif checker_model == 'nli':
            self.bc = NLIChecker()
            self.max_reference_segment_length = 200
        
    def check(
        self,
        question,
        response,
        reference
    ):
        claims = self.claim_extractor.extract(
            question=question,
            response=response
        )
        
        results = []
        for claim in claims:
            claim_str = claim
            if self.claim_format == 'triplet':
                claim_str = ' '.join(claim)
                
            label = self.bc.check(
                claim=claim_str,
                reference=reference,
                question=question,
                response=response,
                max_reference_segment_length=self.max_reference_segment_length
            )
            results.append({
                'claim': claim,
                'label': label
            })
        return results