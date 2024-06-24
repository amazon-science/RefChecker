import re
from typing import List, Union

from spacy.lang.en import English


class RCSentence:
    def __init__(self, sentence_text, is_blank, start=None, end=None) -> None:
        self.text = sentence_text
        self.is_blank = is_blank
        self.start = start
        self.end = end

    def __repr__(self) -> str:
        return self.text

    def to_dict(self):
        return {'text': self.text, 'is_blank': self.is_blank, 'start': self.start, 'end': self.end}
    
    @classmethod
    def from_dict(cls, sent_dict: dict):
        return cls(text=sent_dict['text'], is_blank=sent_dict['is_blank'], start=sent_dict['start'], end=sent_dict['end'])
        

class RCText:
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
                sents.append(RCSentence(prefix, is_blank=True))
            
            surfix = ''
            end_idx = len(s_text) - 1
            while end_idx > start_idx:
                if s_text[end_idx] in blanks:
                    surfix = s_text[end_idx] + surfix
                    end_idx -= 1
                else:
                    break
            if len(s_text[start_idx: end_idx+1]):
                sents.append(RCSentence(s_text[start_idx: end_idx+1], is_blank=False, start=s.start, end=s.end))
            if len(surfix):
                sents.append(RCSentence(surfix, is_blank=True))
        
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

    def get_sentence_ids(self):
        return list(self._sent_id_to_index.keys())

    def to_dict(self):
        return {
            'sents': [s.to_dict() for s in self.sentences],
            'sent_id_to_index': self._sent_id_to_index
        }


class RCClaim:
    def __init__(
        self, 
        format: str,
        content: Union[str, list],
        attributed_sent_ids: List[str]
    ) -> None:
        self.format = format
        self.content = content
        self.attributed_sent_ids = attributed_sent_ids

    def __repr__(self) -> str:
        if self.format == 'triplet':
            return f'("{self.content[0]}", "{self.content[1]}", "{self.content[2]}")'
        elif self.format == 'subsentence':
            ret = self.content + ' '
            for sid in self.attributed_sent_ids:
                ret += f'[{sid}]'
            return ret
        else:
            raise ValueError(f'Unknown Claim Format: {self.format}')        

    def get_content(self, preserve_triplet_form=False):
        if self.format == 'triplet':
            if preserve_triplet_form:
                return f'("{self.content[0]}", "{self.content[1]}", "{self.content[2]}")'
            else:
                return f'{self.content[0]} {self.content[1]} {self.content[2]}'
        else:
            return self.content

    def to_dict(self):
        ret = {
            'format': self.format,
            'content': self.content,
            'attributed_sent_ids': self.attributed_sent_ids
        }
        return ret

    @classmethod
    def from_dict(cls, claim_dict: dict):
        return cls(
            format=claim_dict['format'],
            content=claim_dict['content'],
            attributed_sent_ids=claim_dict['attributed_sent_ids']
        )


class ExtractionResult:
    def __init__(
        self, 
        claims: List[RCClaim],
        response: Union[str, RCText],
        extractor_response: str = None,
    ) -> None:
        self.claims = claims
        self.response = response
        self.extractor_response = extractor_response