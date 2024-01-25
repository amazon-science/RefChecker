import os
import subprocess
from typing import Any, List

from transformers import (
    AutoTokenizer, AutoModelForCausalLM
)

from ..checker_base import CheckerBase
from .ml_models import *

LABELS = ["Entailment", "Neutral", "Contradiction"]

prompt_template_dict = {
    "chatml":
        {
            "system_begin": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_begin": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_begin": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>\n"
        }
}

class RepCChecker(CheckerBase):
    def __init__(
        self,
        model='teknium/OpenHermes-2.5-Mistral-7B',
        classifier='svm',
        classifier_path='svm_n300_l32.pkl',
        prompt_style='chatml',
        selected_layer=-1,
        selected_token=-1,
        device=0
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device,
            torch_dtype=torch.float16,
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.prompt_style = prompt_style
        self.selected_layer = selected_layer
        self.selected_token = selected_token
        self.device = device
        self._download_classifier(classifier_path)
        if classifier == "svm":
            self.classifier = SVM(kernel="rbf")
            self.classifier.load(classifier_path)

    def _download_classifier(self, classifier_path):
        if not os.path.exists(classifier_path):
            url = "https://huggingface.co/zthang/repe/resolve/main/svm_n300_l32.pkl"
            command=["wget", "-O", classifier_path, url]
            try:
                download_state=subprocess.call(command)
            except Exception as e:
                print(e)

    def get_prompt(self, prompt_style, question, premise, hypothesis):
        return f"{prompt_template_dict[prompt_style]['system_begin']}Consider the NLI label between the user given premise and hypothesis.{prompt_template_dict[prompt_style]['system_end']}" \
               f"{prompt_template_dict[prompt_style]['user_begin']}Premise: {question}\n{premise}\nHypothesis: {hypothesis}{prompt_template_dict[prompt_style]['user_end']}" \
               f"{prompt_template_dict[prompt_style]['assistant_begin']}The NLI label (Entailment, Neutral, Contradiction) is"

    @torch.no_grad()
    def _check(
        self,
        claims: List,
        references: List,
        response: str,
        question: str,
    ):
        N1, N2 = len(references), len(claims)
        assert N1 == N2, f"Batches must be of the same length. {N1} != {N2}"
        if isinstance(claims[0], list):
            assert len(claims[0]) == 3
            claims = [f"{c[0]} {c[1]} {c[2]}" for c in claims]
        preds = []
        for i in range(N1):
            prompt = self.get_prompt(prompt_style=self.prompt_style, question=question, premise=references[i], hypothesis=claims[i])
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            res = self.model(input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = res["hidden_states"][1:][self.selected_layer].cpu()
            hidden_states = hidden_states[self.selected_token, :]
            pred = self.classifier.predict(hidden_states)[0]
            preds.append(pred)
        ret = [LABELS[p] for p in preds]
        return ret

if __name__ == "__main__":
    claims = ["H&R Block Online time to process tax return 1-2 days", "H&R Block Online time to process tax return 1-2 days"]
    references = ["I can’t imagine how it would take 2 hours. What record keeping does someone with a 1040ez generally need to do? I used to do all my taxes by hand. I figured about 8 hours total for federal and state and my tax situation was not simple. ",
                  "I did a full 1040 with stock sales, itemized deductions and rental properties. They list “1 hour for form submission”. How does that take 1 hour? Most people hit ‘submit’ on their tax software and other people shove them in an envelope and put them in the mail. Where do they get 1 hour from?"]
    question = "how long does it usually take to get taxes back from h & r block online?"
    checker = RepCChecker()
    ret = checker._check(claims=claims, references=references, question=question, response="")
    print(ret)
