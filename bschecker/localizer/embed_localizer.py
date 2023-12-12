import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from ..utils import split_text


class NaiveEmbedLocalizer(object):
    """aligning the text and triplets"""
    def __init__(
        self, device: int = 0, segment_len: int = 256
    ):
        path_or_name = "princeton-nlp/sup-simcse-roberta-large"
        self.model = AutoModelForSequenceClassification.from_pretrained(
            path_or_name
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path_or_name)
        self.device = device
        self.segment_len = segment_len

    @torch.no_grad()
    def _encode_text(self, text, avg_pooling=False):
        """encode text into embeddings"""
        inputs = self.tokenizer(
            text, max_length=512, truncation=True, return_tensors="pt",
            padding=True, return_token_type_ids=True
        )
        _tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        _output = self.model(**inputs, output_hidden_states=True)
        _hid = _output.hidden_states[0][0][1: -1]
        if avg_pooling:
            _hid = _hid.mean(0)
        return _tokens[1: -1], _hid
    
    @staticmethod
    def cosine_dist(emb1, emb2):
        return float(
            (emb1 * emb2).sum() / (torch.norm(emb1, 2) * torch.norm(emb2, 2))
        )

    @staticmethod
    def normalize(text):
        return text.lower().replace("Ġ", " ").replace("▁", " ")

    @staticmethod
    def decorate(text, color, bgcolor):
        return f"""<span style="background-color:{bgcolor};color:{color};"""+\
            f"""border-radius:2%;">{text}</span>"""

    def locate(self, text, triplet, threshold=[0.65, 0.6, 0.65]):
        assert len(triplet) == 3, "triplet should have 3 elements"
        tokens = []
        segments = split_text(text, self.segment_len)
        text_emb = [] # embeddings for text [L, d]
        triplet_emb = [] # embeddings for triplet [3, d]
        for seg in segments:
            token, emb = self._encode_text(seg, avg_pooling=False)
            tokens.extend(token)
            text_emb.append(emb)
        text_emb = torch.cat(text_emb, 0)
        mask = np.zeros(len(tokens))
        lens = []
        for element in triplet:
            if len(element)>0:
                _, emb = self._encode_text(element, avg_pooling=True)
                triplet_emb.append(emb)
                lens.append(len(token))
        for i in range(3):
            if len(triplet[i]) > 0:
                bounds = []
                # varing window size between 0.8 and 1.2 times of the number of
                # the triplet element's tokens
                len_lb = max(1, int(0.8 * lens[i]))
                len_ub = min(len(text_emb) - 1, int(1.2 * lens[i]))
                for length in range(len_lb, len_ub):
                    for j in range(len(text_emb) - length):
                        emb1 = text_emb[j: j + length].mean(0)
                        emb2 = triplet_emb[i]
                        sim = self.cosine_dist(emb1, emb2)
                        _phrase = self.normalize("".join(tokens[j: j + length]))
                        if (_phrase == triplet[i].strip().lower()):
                            sim = threshold[i] + 0.01
                        if (len(_phrase) - len(triplet[i].strip().lower()) < 5) and (_phrase.startswith(triplet[i].strip().lower())) or (triplet[i].strip().lower().startswith(_phrase)):
                            sim = threshold[i] + 0.01
                        if sim > threshold[i]:
                            bounds.append([j, j + length, sim])
                for j in range(len(bounds) - 1, -1, -1):
                    if bounds[j][2] < threshold[i] * 1.2 and any([((x[0] >= bounds[j][0] and x[0] < bounds[j][1]) or (x[1] > bounds[j][0] and x[1] <= bounds[j][1])) and x[2] > bounds[j][2] * 1.05 for x in bounds[:j]]):
                        del bounds[j]
                for b in bounds:
                    mask[b[0]: b[1]] = i + 1
        vs = [int(x) for x in mask]
        ret = ''
        cmap = ['black', 'red', 'blue', 'green']
        for i in range(len(tokens)):
            ret += self.decorate(tokens[i], cmap[vs[i]], "#F1CEF3")
        return ret.replace("Ġ", " ").replace("▁", " ")


if __name__ == "__main__":
    localizer = NaiveEmbedLocalizer()
    text = """Eleanor Arnason (born 1945) is an American science fiction and fantasy writer. She is best known for her novel A Woman of the Iron People (1991), which won the James Tiptree, Jr. Award and was a finalist for the Nebula Award for Best Novel. Her other works include Ring of Swords (1993), The Sword Smith (1998), and The Hound of Merin (2002). She has also written several short stories, including "Dapple" (1991), which won the Nebula Award for Best Novelette. """
    sents = [
        "Eleanor Arnason (born 1945) is an American science fiction and fantasy writer.",
        "She is best known for her novel A Woman of the Iron People (1991), which won the James Tiptree, Jr. Award and was a finalist for the Nebula Award for Best Novel.",
        "Her other works include Ring of Swords (1993), The Sword Smith (1998), and The Hound of Merin (2002).",
    ]
    triplets = [
        ["Eleanor Arnason", "born", "1945"],
        ["Eleanor Arnason", "is", "American science fiction and fantasy writer"],
        ["A Woman of the Iron People (1991)", "won", "James Tiptree, Jr. Award"]
    ]
    for triplet in triplets:
        for sent in sents:
            print(localizer.locate(sent, triplet))
        print()
