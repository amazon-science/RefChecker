from .checker import (
    LLMChecker,
    NLIChecker, 
    AlignScoreChecker,
    RepCChecker
)
from .extractor import (
    LLMExtractor, 
    MixtralExtractor,
    MistralExtractor
)


class RefChecker:
    def __init__(
        self,
        extractor_name:str='claude3-sonnet',
        claim_format:str='subsentence',
        checker_name:str='claude3-sonnet'
    ) -> None:
        if claim_format not in ['triplet', 'subsentence']:
            raise 'We currently only support claim formats of \'triplet\' and \'subsentence\''
        
        if extractor_name.startswith('claude3') or extractor_name.startswith('gpt'):
            self.extractor = LLMExtractor(claim_format=claim_format, model=extractor_name)
        elif extractor_name == 'mistral':
            assert claim_format == 'triplet', 'The Mistral extractor currently only supports triplet claims'
            self.extractor = MistralExtractor(claim_format=claim_format)
        elif extractor_name == 'mixtral':
            assert claim_format == 'triplet', 'The Mixtral extractor currently only supports triplet claims'
            self.extractor = MixtralExtractor(claim_format=claim_format)
        
        if checker_name.startswith('claude3') or checker_name.startswith('gpt'):
            self.checker = LLMChecker(model=checker_name)
            self.max_reference_segment_length = 0
        elif checker_name == 'nli':
            self.checker = NLIChecker()
            self.max_reference_segment_length = 200
        elif checker_name == 'alignscore':
            self.checker = AlignScoreChecker()
            self.max_reference_segment_length = 200
        
    def check(
        self,
        response,
        reference,
        question=None,
        max_new_tokens=500
    ):
        extraction_result = self.extractor.extract(
            question=question,
            response=response,
            max_new_tokens=max_new_tokens
        )
        
        checking_result = self.checker.check(
            claim=[extraction_result.claims],
            reference=[reference],
            question=[question],
            max_reference_segment_length=self.max_reference_segment_length
        )
        
        return extraction_result.claims, checking_result[0]