import re

from spacy.lang.en import English


class Claim:
    def __init__(self, 
                 text,
                 attributed_sent_ids
                 ) -> None:
        self.text = text
        self.attributed_sent_ids = attributed_sent_ids

    def __repr__(self) -> str:
        ret = self.text + ' '
        for sid in self.attributed_sent_ids:
            ret += f'[{sid}]'
        return ret 


class Sentence:
    def __init__(self, sentence_text, is_blank) -> None:
        self.text = sentence_text
        self.is_blank = is_blank

    def __repr__(self) -> str:
        return self.text


class Response:
    def __init__(self, response_text) -> None:
        self.orig_text = response_text
        
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        
        self.sentences = None
        self._sent_id_to_index = dict()
        self.sentencize()
        
        self.indexed_response = None
    
    def sentencize(self):
        blanks = [' ', '\n', '\r']
        
        sents = []
        for s in self.nlp(self.orig_text).sents:
            s_text = s.text
            prefix = ''
            start_idx = 0
            while start_idx < len(s_text):
                if s_text[start_idx] in blanks:
                    prefix += s_text[start_idx]
                    start_idx += 1
                else:
                    start_idx = start_idx
                    break
            if len(prefix):
                sents.append(Sentence(prefix, is_blank=True))
            
            surfix = ''
            end_idx = len(s_text) - 1
            while end_idx > start_idx:
                if s_text[end_idx] in blanks:
                    surfix = s_text[end_idx] + surfix
                    end_idx -= 1
                else:
                    break
            if len(s_text[start_idx: end_idx+1]):
                sents.append(Sentence(s_text[start_idx: end_idx+1], is_blank=False))
            if len(surfix):
                sents.append(Sentence(surfix, is_blank=True))
        
        self.sentences = sents
        sent_id = 1
        for index, sent in enumerate(self.sentences):
            if not sent.is_blank:
                self._sent_id_to_index[str(sent_id)] = index
                sent_id += 1
    
    def get_indexed_response(self, condense_newlines: bool):
        if self.indexed_response is None:
            sent_id = 1
            res = ''
            for i, s in enumerate(self.sentences):
                sent_text = s.text
                if condense_newlines: 
                    sent_text = re.sub(r'(\n\s*)+\n', '\n', sent_text)
                    sent_text = re.sub(r'(\r\s*)+\r', '\r', sent_text)
                if s.is_blank:
                    res += sent_text
                else:
                    res += f'[{sent_id}] {sent_text}'
                    sent_id += 1
                    if i < len(self.sentences) - 1:
                        res += ' '
            self.indexed_response = res
        return self.indexed_response

    def get_sentence_by_id(self, sent_id: str):
        assert sent_id in self._sent_id_to_index, "Invalid sentence ID"
        assert self._sent_id_to_index[sent_id] < len(self.sentences)
        return self.sentences[self._sent_id_to_index[sent_id]]


def process_extraction_response(
    response, 
    excluded_content_prefix='### Text'
):
    response = response.strip()
    if excluded_content_prefix and excluded_content_prefix in response:
        response = response[:response.index(excluded_content_prefix)]
    
    claims = []
    for c in re.findall(r'.*[\[\d+\]]+', response):
        sent_ids = []
        first_sid_index = None
        for sid in re.finditer(r'\[\d+\]', c):
            if first_sid_index is None:
                first_sid_index = sid.start()
            sent_ids.append(sid.group()[1:-1])
        claims.append(Claim(text=c[:first_sid_index].strip(), attributed_sent_ids=sent_ids))
    return claims
