import re
from typing import List, Union
from ..base import RCText, RCClaim


class ExtractorBase:
    def __init__(
        self,
        claim_format:str='triplet'
    ) -> None:
        assert claim_format in ['triplet', 'subsentence']
        self.claim_format = claim_format

    def extract(
        self, 
        batch_responses, 
        batch_questions=None, 
        max_new_tokens=500,
        custom_llm_api_func=None,
        **kwargs
    ):
        if self.claim_format == 'triplet':
            result = self.extract_claim_triplets(
                batch_responses=batch_responses,
                batch_questions=batch_questions,
                max_new_tokens=max_new_tokens,
                custom_llm_api_func=custom_llm_api_func,
                **kwargs
            )
        elif self.claim_format == 'subsentence':
            result = self.extract_subsentence_claims(
                batch_responses=batch_responses,
                batch_questions=batch_questions,
                max_new_tokens=max_new_tokens,
                custom_llm_api_func=custom_llm_api_func,
                **kwargs
            )
        return result

    def extract_claim_triplets(
        self,
        batch_responses,
        batch_questions=None, 
        max_new_tokens=500,
        custom_llm_api_func=None,
        **kwargs
    ):
        raise NotImplementedError

    def extract_subsentence_claims(
        self,
        batch_responses,
        batch_questions=None,
        max_new_tokens=500,
        custom_llm_api_func=None,
        **kwargs
    ):
        raise NotImplementedError

    def parse_claims(
        self,
        response, 
        claim_starting_prefix=None,
        excluded_content_prefix=None,
        response_sentence_ids=None
    ):
        response = response.strip()
        if excluded_content_prefix and excluded_content_prefix in response:
            response = response[:response.index(excluded_content_prefix)]
        
        if claim_starting_prefix and claim_starting_prefix in response:
            response = response[response.index(claim_starting_prefix) + len(claim_starting_prefix):]
        
        if self.claim_format == 'triplet':
            return self._parse_claim_triplets(response)
        elif self.claim_format == 'subsentence':
            claims = []
            # for c in re.findall(r'.*[\[\d+\]]+', response):
            for c in re.findall(r'.*[\[(\d+(?:,\s*\d+)*)\]]', response):
                sent_ids = []
                first_sid_index = None
                for sid in re.finditer(r'\[(\d+(?:,\s*\d+)*)\]', c):
                    if first_sid_index is None:
                        first_sid_index = sid.start()
                    sent_id_str = sid.group()[1:-1]
                    if ',' in sent_id_str:
                        for _id in sent_id_str.split(','):
                            _id = _id.strip()
                            sent_ids.append(_id)
                    else:
                        sent_ids.append(sid.group()[1:-1])
                sent_ids = [_id for _id in sent_ids if _id in response_sentence_ids]
                if len(sent_ids):
                    claims.append(RCClaim(
                        format=self.claim_format,
                        content=c[:first_sid_index].strip(), 
                        attributed_sent_ids=sent_ids
                    ))
            return claims
        else:
            raise ValueError(f'Unknown Claim Format: {self.format}')

    def _parse_claim_triplets(self, text):
        ret = []
        patterns = [
            r'\(".*", ".*", ".*"\)',
            r'\(".*", ".*", \'.*\'\)',
            r'\(".*", \'.*\', ".*"\)',
            r'\(\'.*\', ".*", ".*"\)',
            r'\(".*", \'.*\', \'.*\'\)',
            r'\(\'.*\', ".*", \'.*\'\)',
            r'\(\'.*\', \'.*\', ".*"\)',
            r'\(\'.*\', \'.*\', \'.*\'\)'
        ]
        for p in patterns:
            triplets = self._parse_triplets(p, text, triple_length=3)
            if triplets:
                ret += triplets

        # deduplication
        final_triple_set = []
        for t in ret:
            if tuple(t) not in final_triple_set:
                final_triple_set.append(tuple(t))
        
        # return [list(t) for t in final_triple_set]
        return [RCClaim('triplet', list(t), None) for t in final_triple_set]

    def _parse_triplets(self, pattern, text, triple_length=3):
        triplets = []
        matches = re.findall(pattern, text)
        for m in matches:
            try:
                t = eval(m)
            except:
                t = m.split(', ')
                if t[0].startswith('('):
                    t[0] = t[0][1:]
                if t[-1].endswith(')'):
                    t[-1] = t[-1][:-1]
            if len(t) != triple_length:
                continue
            if any([not isinstance(e, str) for e in t]):
                continue
            if any([len(e) == 0 for e in t]):
                continue
            triplets.append(list(t))
        return triplets