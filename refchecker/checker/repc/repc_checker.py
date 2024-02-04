import os
import subprocess
import tarfile
from huggingface_hub import hf_hub_download
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
        classifier='nn_ensemble',
        classifier_dir='saved_models/repc',
        prompt_style='chatml',
        selected_token=-1,
        device=0,
        **kwargs
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device,
            torch_dtype=torch.float16,
            cache_dir="/home/ubuntu/huggingface_models",
            trust_remote_code=True,
            use_auth_token="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",
            use_safetensors=False
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.prompt_style = prompt_style
        self.selected_token = selected_token
        self.device = device
        self.classifier_str = classifier
        self.classifier_dir = classifier_dir
        if classifier == "nn_ensemble":
            self.n_train = 2000
            expert_paths = [f"{self.classifier_dir}/upload/nn_anli_n{self.n_train}_l{i}" for i in range(self.model.config.num_hidden_layers)]
            if not os.path.exists(f"{self.classifier_dir}/upload/nn_anli_n{self.n_train}_l31"):
                hf_hub_download(repo_id="zthang/repe", filename="nn.tar.gz", local_dir=self.classifier_dir)
                tar = tarfile.open(os.path.join(self.classifier_dir, "nn.tar.gz"), "r:gz")
                tar.extractall(path=self.classifier_dir)
                tar.close()
            self.classifier = EnsembleClassifier(input_size=(self.model.config.num_hidden_layers) * 3,
                                       output_size=3,
                                       num_experts=self.model.config.num_hidden_layers,
                                       expert_paths=expert_paths,
                                       expert_type="nn",
                                       classifier_type="mlp")
            self.classifier_path = os.path.join(self.classifier_dir, "ensemble_mlp_nn_2000_anli_n2000_l0")
            if not os.path.exists(self.classifier_path):
                hf_hub_download(repo_id="zthang/repe", filename="ensemble_mlp_nn_2000_anli_n2000_l0", local_dir=self.classifier_dir)
        elif classifier == "nn":
            self.selected_layer = 17
            self.n_train = 2000
            self.input_size = 4096
            self.hidden_size = 4096 // 4
            self.classifier = PyTorchClassifier(input_size=self.input_size, hidden_size=self.hidden_size)
            self.classifier_path = f"{self.classifier_dir}/nn/nn_anli_n{self.n_train}_l{self.selected_layer}"
            if not os.path.exists(self.classifier_path):
                hf_hub_download(repo_id="zthang/repe", filename=f"nn/nn_anli_n{self.n_train}_l{self.selected_layer}", local_dir=self.classifier_dir)
        elif classifier == "svm_ensemble":
            self.n_train = 1000
            expert_paths = [f"{self.classifier_dir}/svm/svm_anli_n{self.n_train}_l{i}" for i in range(self.model.config.num_hidden_layers)]
            if not os.path.exists(f"{self.classifier_dir}/svm/svm_anli_n{self.n_train}_l31"):
                hf_hub_download(repo_id="zthang/repe", filename="svm.tar.gz", local_dir=self.classifier_dir)
                tar = tarfile.open(os.path.join(self.classifier_dir, "svm.tar.gz"), "r:gz")
                tar.extractall(path=self.classifier_dir)
                tar.close()
            self.classifier = EnsembleClassifier(input_size=(self.model.config.num_hidden_layers) * 3,
                                       output_size=3,
                                       num_experts=self.model.config.num_hidden_layers,
                                       expert_paths=expert_paths,
                                       expert_type="svm",
                                       classifier_type="mlp")
            self.classifier_path = os.path.join(self.classifier_dir, "ensemble_mlp_svm_1000_anli_n1000_l0")
            if not os.path.exists(self.classifier_path):
                hf_hub_download(repo_id="zthang/repe", filename="ensemble_mlp_svm_1000_anli_n1000_l0", local_dir=self.classifier_dir)
        elif classifier == "svm":
            self.selected_layer = 15
            self.n_train = 1000
            self.classifier = SVM(kernel="rbf")
            self.classifier_path = f"{self.classifier_dir}/svm/svm_anli_n{self.n_train}_l{self.selected_layer}"
            if not os.path.exists(self.classifier_path):
                hf_hub_download(repo_id="zthang/repe", filename=f"svm/svm_anli_n{self.n_train}_l{self.selected_layer}", local_dir=self.classifier_dir)
        else:
            raise ValueError("classifier must in [svm, nn, svm_ensemble, nn_ensemble.")

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
            if self.classifier_str in ["svm", "nn"]:
                hidden_states = res["hidden_states"][1:][self.selected_layer].cpu().numpy()
                hidden_states = hidden_states[:, self.selected_token, :]
            else:
                hidden_states = torch.cat(res["hidden_states"][1:]).transpose(0, 1)[-1:].cpu().numpy()
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
