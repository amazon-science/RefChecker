import re


class ExtractorBase:
    def __init__(
        self,
        claim_format:str='triplet'
    ) -> None:
        self.claim_format = claim_format

    def extract(self, response, question=None, max_new_tokens=500):
        claims = None
        if self.claim_format == 'triplet':
            claims = self.extract_claim_triplets(
                response=response,
                question=question,
            )
        return claims

    def extract_claim_triplets(
        self,
        response,
        question=None, 
        max_new_tokens=500
    ):
        raise NotImplementedError

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
        
        return [list(t) for t in final_triple_set]

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
