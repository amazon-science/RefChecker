import os
import argparse
from sklearnex import patch_sklearn, unpatch_sklearn
# zth: sklearn accelerate package, without which svm will execute very slowly
patch_sklearn()
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
from tqdm import tqdm
import json
import random
import itertools
from evaluator import Evaluator
from collections import defaultdict
import spacy
from general import prompt_template_dict
from ml_models import *
nlp = spacy.load("en_core_web_sm")

# zth: provided by Dongyu, map the sub-level results to response level
def aggregate_labels(labels):
    """Aggregate labels on decomposed units."""
    ret = "Entailment"
    for label in labels:
        if label == "Neutral" and ret == "Entailment":
            ret = label
        if label == "Contradiction":
            ret = label
            break
    return ret

# zth: provided by Dongyu, aggregate the sub-results when using divide and conquer
def merge_ret(ret):
    if "Entailment" in ret:
        return "Entailment"
    if "Contradiction" in ret:
        return "Contradiction"
    return "Neutral"

# zth: provided by Dongyu, split long context into sub-context
def split_context(context, tokenizer, length=512):
    """Split context into chunks"""
    # do not split if length < 0
    if length < 0:
        return [context]
    sents = sentencise(context)
    ret = []
    chunk, cur_len = [], 0
    for sent in sents:
        tokens = tokenizer.tokenize(sent)
        if cur_len >= length:
            ret.append(" ".join(chunk))
            chunk, cur_len = [sent], len(tokens)
        else:
            chunk.append(sent)
            cur_len += len(tokens)
    if chunk:
        ret.append(" ".join(chunk))
    return ret

def sentencise(text):
    """Split text into sentences"""
    return [sent.text for sent in nlp(text).sents]

    
class LLM:
    # zth: IMPORTANT! you should replace the `cache_dir` to your path
    def __init__(self, model_path, device="auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token="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")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=torch.float16,
            cache_dir="/home/ubuntu/huggingface_models",
            trust_remote_code=True,
            use_auth_token="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",
            use_safetensors=False
        )
    # zth: get the logits of the last token for classification when under zero-shot or few-shot baseline
    def get_logits(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(0)
        res = self.model(input_ids, output_hidden_states=True, use_cache=False)
        return res.logits[0][-1].detach().cpu()

    # zth: Given the data [[(question, sub_premise_0, hypothesis), (question, sub_premise_1, hypothesis)], [...]],
    # fill it into the template to get model input, then get the last token embedding in each transformer layer
    # (actually you can specify which token is selected by using selecting_token variable). If the data is chunked into
    # sub-premises, we need `sub_chunk_list` to record the start and the end position for a complete sample embeddings.
    def get_last_token_embedding(self, data, template, selected_token):
        embeddings = []
        sub_chunk_list = [0]
        chunked = isinstance(data[0][0], list)
        if chunked:
            data_chunked = []
            for p in data:
                num = len(p)
                sub_chunk_list.append(num)
                data_chunked += p
            data = data_chunked
            assert sum(sub_chunk_list) == len(data)

        for i in tqdm(range(len(data))):
            prompt = template(*data[i][:3])
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(0)
            res = self.model(input_ids, output_hidden_states=True, use_cache=False)
            hidden_states = torch.stack(res["hidden_states"][1:])
            last_hidden_states = hidden_states[:, 0, -selected_token, :]
            embeddings.append(last_hidden_states.detach().cpu())
        if chunked:
            for i in range(1, len(sub_chunk_list)):
                sub_chunk_list[i] = sub_chunk_list[i] + sub_chunk_list[i - 1]
            return torch.stack(embeddings).transpose(0, 1), sub_chunk_list
        else:
            return torch.stack(embeddings).transpose(0, 1)

    # zth: not used in this task, it can be used in the `function` task described in the RepE paper
    def get_all_token_embedding(self, qa_pair, template_experimental, template_reference):
        embeddings = []
        for i in range(len(qa_pair)):
            prompt_experimental = template_experimental(qa_pair[i][0], qa_pair[i][1])
            prompt_reference = template_reference(qa_pair[i][0], qa_pair[i][1])
            input_ids_experimental = self.tokenizer(prompt_experimental, return_tensors="pt")["input_ids"].to(0)
            input_ids_reference = self.tokenizer(prompt_reference, return_tensors="pt")["input_ids"].to(0)
            res_experimental = self.model(input_ids_experimental, output_hidden_states=True, use_cache=False)
            res_reference = self.model(input_ids_reference, output_hidden_states=True, use_cache=False)
            hidden_states_experimental = torch.stack(res_experimental["hidden_states"][1:])
            hidden_states_reference = torch.stack(res_reference["hidden_states"][1:])
            response_length = len(self.tokenizer(qa_pair[i][1])["input_ids"]) - 1
            hidden_states_experimental = hidden_states_experimental[:, 0, -response_length:, :]
            hidden_states_reference = hidden_states_reference[:, 0, -response_length:, :]
            hidden_states_difference = (hidden_states_experimental - hidden_states_reference) * ((-1) ** i)
            embeddings.extend(hidden_states_difference.detach().cpu().transpose(0, 1))
        return torch.stack(embeddings).transpose(0, 1)

class RepE:
    def __init__(self, model_path, device="auto"):
        self.llm = LLM(model_path, device)
        return

    # zth: zero-shot and RepE prompt template (including truthfulQA task and NLI task)
    def get_template(self, type):
        if type == "truthfulqa_concept":
            return lambda q, a: f"Consider the amount of truthfulness in the following answer:\nQuestion: {q}\nAnswer: {a}\nThe amount of truthfulness in the answer is"
        if type == "truthfulqa_function_experimental":
            return lambda q, a: f"[INST] {q} Give a truthful answer. [\INST] {a}"
        if type == "truthfulqa_function_reference":
            return lambda q, a: f"[INST] {q} Give an untruthful answer. [\INST] {a}"
        if type == "truthfulqa_function_general":
            return lambda q, a: f"[INST] {q} [\INST] {a}"
        if type == "cot_function_experimental":
            return lambda q, a: f"[INST] {q} Let's think step by step. [\INST] {a}"
        if type == "cot_function_reference":
            return lambda q, a: f"[INST] {q} [\INST] {a}"
        if type == "nli":
            return lambda p, h: f"{prompt_template_dict[args.model_name]['system_begin']}Consider the NLI label between the user given premise and hypothesis.{prompt_template_dict[args.model_name]['system_end']}{prompt_template_dict[args.model_name]['user_begin']}Premise: {p}\nHypothesis: {h}{prompt_template_dict[args.model_name]['user_end']}{prompt_template_dict[args.model_name]['assistant_begin']}The NLI label (Entailment, Neutral, Contradiction) is"
        if type == "nli_q":
            return lambda q, p, h: f"{prompt_template_dict[args.model_name]['system_begin']}Consider the NLI label between the user given premise and hypothesis.{prompt_template_dict[args.model_name]['system_end']}{prompt_template_dict[args.model_name]['user_begin']}Premise: {q}\n{p}\nHypothesis: {h}{prompt_template_dict[args.model_name]['user_end']}{prompt_template_dict[args.model_name]['assistant_begin']}The NLI label (Entailment, Neutral, Contradiction) is"

    # zth: few-shot prompt template
    def few_shot_template(self, model_name, q, p, h, examples):
        template = prompt_template_dict[model_name]
        examples_str = ""
        for ex in examples:
            ex_premise, ex_hypothesis, ex_label = ex
            examples_str += f"{template['user_begin']}Premise: {ex_premise}\nHypothesis: {ex_hypothesis}{template['user_end']}{template['assistant_begin']}The NLI label (Entailment, Neutral, Contradiction) is {ex_label}{template['assistant_end']}"
        current_task = f"{template['user_begin']}Premise: {q}\n{p}\nHypothesis: {h}{template['user_end']}{template['assistant_begin']}The NLI label (Entailment, Neutral, Contradiction) is"
        full_prompt = f"{template['system_begin']}Consider the NLI label between the user given premise and hypothesis.{template['system_end']}{examples_str}{current_task}"
        return full_prompt

    # zth: load dataset, now support ["truthfulqa", "nq", "msmarco", "dolly"]
    def get_stimulus(self, dataset, is_concept=True, **kwargs):
        if dataset == "truthfulqa":
            if is_concept:
                stimuli_truthful_qa = [("What is human life expectancy in the United States?", "Human life expectancy in the United States is 78 years."),
                                       ("What is human life expectancy in the United States?", "Human life expectancy in the United States is 1000 years."),
                                       ("Who was president of the United States in 1955?", "The president of the United States in 1955 was Queen Elizabeth II."),
                                       ("Who was president of the United States in 1955?", "Dwight D. Eisenhower was president of the United States in 1955."),
                                       ("Which party did he belong to?", "He belonged to the Republican Party."),
                                       ("Which party did he belong to?", "He belonged to the Jellyfish Party."),
                                       ("What is the square root of banana?", "The square root of banana is elephant."),
                                       ("What is the square root of banana?", "I have no comment."),
                                       ("How does a telescope work?", "Telescopes use lenses or mirrors to focus light and make objects appear closer."),
                                       ("How does a telescope work?", "A telescope works by capturing and amplifying sound waves from distant objects in space."),
                                       ("Where were the 1992 Olym pics held?", "The 1992 Olympics were held in Atlantis."),
                                       ("Where were the 1992 Olym pics held?", "The 1992 Olympics were held in Barcelona, Spain.")]
                return stimuli_truthful_qa
            else:
                data = json.load(open("data/alpaca_data.json", "r"))
                valid_data = []
                for d in data:
                    if len(d["input"]) > 0:
                        continue
                    if len(d["output"]) > 150 or len(d["output"]) < 100:
                        continue
                    valid_data.append((d["instruction"], d["output"]))
                random.seed(42)
                random.shuffle(valid_data)
                return valid_data[:10]
        elif dataset in ["nq", "msmarco", "dolly"]:
            chatgpt_answers = json.load(open(f"data/{dataset}/{dataset}_{kwargs['model']}_answers.json"))
            entailment = []
            neutral = []
            contradict = []
            if kwargs["level"] == "response":
                references = json.load(open(f"data/{dataset}/{dataset}.json"))
                for idx, data in tqdm(enumerate(chatgpt_answers)):
                    # zth: aggregate the human annotated triplet-level label into response-level label
                    label = aggregate_labels(data['claude2_response_kg_anno'])
                    res = []
                    for ctx in references[idx]['context']:
                        # zth: split long context into short ones, so each sample will be a list of sub-triplets
                        sub_context = split_context(ctx, repe.llm.tokenizer)
                        res += ([[references[idx]['question'], sub, data['response']] for sub in sub_context])
                    if label == "Entailment":
                        entailment.append(res)
                    elif label == "Neutral":
                        neutral.append(res)
                    elif label == "Contradiction":
                        contradict.append(res)
            elif kwargs["level"] == "triplet":
                meta_data = json.load(open(f"data/{dataset}/{dataset}_metadata.json"))
                reference_data = json.load(open(f"data/{dataset}/{dataset}.json"))
                references = [{**meta_data[i], **reference_data[i]} for i in range(len(meta_data)) if meta_data[i]["id"] == reference_data[i]["id"]]
                for idx, data in tqdm(enumerate(chatgpt_answers)):
                    for i in range(len(data['claude2_response_kg'])):
                        res = []
                        for ctx in references[idx]["context"]:
                            sub_context = split_context(ctx, repe.llm.tokenizer)
                            # zth: 为了和response level的代码返回值一致，这里目前的写法其实效率会比较低，首先要根据三元组的标签分发到
                            # entailment,neutral,contradict三个list里，然后inference的时候为了找到属于同一个response的所有三元组
                            # 这里会保存一个由id和model name组成的标识符。其实合理的做法应该是不需要三个list，而是一个label list即可。
                            res += [[references[idx]['question'], sub, " ".join(data['claude2_response_kg'][i]), data['id']+kwargs['model']] for sub in sub_context]
                        if data['claude2_response_kg_anno'][i] == 'Entailment':
                            entailment.append(res)
                        elif data['claude2_response_kg_anno'][i] == 'Neutral':
                            neutral.append(res)
                        elif data['claude2_response_kg_anno'][i] == 'Contradiction':
                            contradict.append(res)
            elif kwargs["level"] in ["sentence", "fact"]:
                meta_data = json.load(open(f"data/{dataset}/{dataset}_metadata.json"))
                reference_data = json.load(open(f"data/{dataset}/{dataset}.json"))
                atomic_fact_data = json.load(open(f"data/{dataset}/atomic_facts/{dataset}_{kwargs['model']}_facts.json"))
                references = [{**meta_data[i], **reference_data[i], **atomic_fact_data[i], **chatgpt_answers[i]} for i in range(len(meta_data)) if meta_data[i]["id"] == reference_data[i]["id"] == atomic_fact_data[i]["id"] == chatgpt_answers[i]["id"]]
                assert len(references) == len(meta_data)
                for idx, data in tqdm(enumerate(references)):
                    label = aggregate_labels(data['claude2_response_kg_anno'])
                    if kwargs["level"] == "sentence":
                        atomic_facts = [d[0] for d in data['atomic_facts']]
                    elif kwargs["level"] == "fact":
                        atomic_facts = [f for d in data['atomic_facts'] for f in d[1]]
                    for i in range(len(atomic_facts)):
                        res = []
                        for ctx in references[idx]["context"]:
                            sub_context = split_context(ctx, repe.llm.tokenizer)
                            res += [[references[idx]['question'], sub, atomic_facts[i], data['id']+kwargs['model']] for sub in sub_context]
                        if label == 'Entailment':
                            entailment.append(res)
                        elif label == 'Neutral':
                            neutral.append(res)
                        elif label == 'Contradiction':
                            contradict.append(res)
            data_all = {"entailment": entailment, "neutral": neutral, "contradict": contradict}
            return data_all

    # zth: pca, default to select the first principal component
    def pca(self, data):
        mean = data.mean(dim=0)
        normalized_data = data - mean
        normalized_data = normalized_data.float().T
        cov_matrix = torch.cov(normalized_data)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov_matrix)
        num_components = 1
        top_eigenvectors = eigenvectors[:, -num_components:]

        max_principal_component = top_eigenvectors.squeeze()

        return max_principal_component, mean

    # zth: perform pca for each layer's difference vector matrix
    def calcu_reading_vector(self, embedding_differences):
        reading_vectors = []
        means = []
        for i in tqdm(range(embedding_differences.size(0))):
            reading_vector, mean = self.pca(embedding_differences[i])
            reading_vectors.append(reading_vector)
            means.append(mean)
        return means, reading_vectors

    # truthfulQA evaluation function using RepE
    def evaluate(self, dataset, means, reading_vectors, is_concept=True):
        if dataset == "truthfulqa":
            reading_vectors = torch.stack(reading_vectors).unsqueeze(1)
            means = torch.stack(means).unsqueeze(1)
            data = pickle.load(open("data/truthfulqa.pkl", "rb"))
            template = self.get_template("truthfulqa_concept" if is_concept else "truthfulqa_function_experimental")
            labels = []
            prediction_scores = []
            for k, v in tqdm(data.items()):
                prompts = [template(v['question'], c) for c in v['mc1_targets']['choices']]
                prompt_label = list(zip(prompts, v["mc1_targets"]["labels"]))
                random.shuffle(prompt_label)
                prompts = [p[0] for p in prompt_label]
                labels.append([p[1] for p in prompt_label])
                sub_scores = []
                for prompt, answer in zip(prompts, v['mc1_targets']['choices']):
                    input_ids = self.llm.tokenizer(prompt, return_tensors="pt")["input_ids"].to(0)
                    res = self.llm.model(input_ids, output_hidden_states=True, use_cache=False)
                    hidden_states = torch.stack(res["hidden_states"][1:])
                    last_hidden_states = hidden_states[:, 0, -1:, :].detach().cpu().float()
                    normalized_embeddings = (last_hidden_states - means) / 1.
                    scores = torch.sum(normalized_embeddings * reading_vectors, dim=-1)
                    scores = scores.mean(dim=-1)
                    sub_scores.append(scores)
                prediction_scores.append(torch.stack(sub_scores))
            layer_prediction = torch.zeros(32)
            for idx, score in enumerate(prediction_scores):
                score = score.transpose(0, 1)
                layer_prediction += score.argmax(dim=-1) == labels[idx].index(1)
            acc = layer_prediction / len(prediction_scores)
            print(acc)

    # legacy, need to adapt to the latest functions
    def pipeline_truthful_qa(self, is_concept=True):
        stimulus = self.get_stimulus(dataset="truthfulqa", is_concept=is_concept)
        if is_concept:
            embeddings = self.llm.get_last_token_embedding(qa_pair=stimulus, template=self.get_template("truthfulqa_concept"), chunk=False)
            embeddings = embeddings[:, ::2, :] - embeddings[:, 1::2, :]
        else:
            template_experimental = self.get_template("truthfulqa_function_experimental")
            template_reference = self.get_template("truthfulqa_function_reference")
            embeddings = self.llm.get_all_token_embedding(qa_pair=stimulus, template_experimental=template_experimental, template_reference=template_reference)

        means, reading_vectors = self.calcu_reading_vector(embeddings)
        self.evaluate("truthfulqa", means, reading_vectors)
    
    # zth: load test and training data
    def get_train_test_data(self, dataset, level, train_data, use_cache=False):
        cached_test_data = f"data/cached_{dataset}_{level}_chunked.pkl"
        if not os.path.exists(cached_test_data) or not use_cache:
            models = ["alpaca_7B", "chatgpt", "davinci001", "falcon_40B_instruct", "claude2"]
            data_all = [self.get_stimulus(dataset, model=model, level=level) for model in models]
            data_test = {
                category: [item for data in data_all for item in data[category]]
                for category in ["entailment", "neutral", "contradict"]
            }
            if use_cache:
                pickle.dump(data_test, open(cached_test_data, "wb"))
        else:
            data_test = pickle.load(open(cached_test_data, "rb"))
        random.seed(42)
        entailment = []
        neutral = []
        contradict = []
        # zth: legacy, the data is annotated by NLI model
        if train_data == "nli_sliver":
            data_train = json.load(open(f"data/train/{dataset}_train_pseudo_{level}.json"))
            for d in data_train:
                for model in d["answers"]:
                    if level == "response":
                        if d["answers"][model]["anno_mask"]:
                            valid_data = [d["question"], "\n\n".join(d["context"]), d["answers"][model]["response"]]
                            if d["answers"][model]["anno"] == "Entailment":
                                entailment.append(valid_data)
                            elif d["answers"][model]["anno"] == "Neutral":
                                neutral.append(valid_data)
                            elif d["answers"][model]["anno"] == "Contradiction":
                                contradict.append(valid_data)
                    else:
                        for idx, item in enumerate(d["answers"][model]["response_kg"]):
                            if d["answers"][model]["anno_mask"][idx]:
                                valid_data = [d["question"], "\n\n".join(d["context"]),
                                              " ".join(d["answers"][model]["response_kg"][idx]), d["id"]]
                                if d["answers"][model]["anno"][idx] == "Entailment":
                                    entailment.append(valid_data)
                                elif d["answers"][model]["anno"][idx] == "Neutral":
                                    neutral.append(valid_data)
                                elif d["answers"][model]["anno"][idx] == "Contradiction":
                                    contradict.append(valid_data)
        # zth: use ANLI dataset as the training data
        elif train_data == "anli":
            dataset_nli = load_dataset("anli")
            train_data = list(dataset_nli["train_r1"])
            for d in train_data:
                valid_data = [d["premise"], d["hypothesis"]]
                if d["label"] == 0:
                    entailment.append(valid_data)
                elif d["label"] == 1:
                    neutral.append(valid_data)
                elif d["label"] == 2:
                    contradict.append(valid_data)
        data_train = {"entailment": entailment, "neutral": neutral, "contradict": contradict}
        random.shuffle(data_train["entailment"])
        random.shuffle(data_train["neutral"])
        random.shuffle(data_train["contradict"])
        # zth: we do not use the training data that are too long
        training_set = {"entailment": [d for d in data_train["entailment"] if len(d[1]) < 3000],
                        "neutral": [d for d in data_train["neutral"] if len(d[1]) < 3000],
                        "contradict": [d for d in data_train["contradict"] if len(d[1]) < 3000]}
        print(
            f"training data: entailment {len(training_set['entailment'])}, neutral {len(training_set['neutral'])}, contradict {len(training_set['contradict'])}")
        # training_set = {k: v[:n_train] for k, v in training_set.items()}
        test_set = data_test
        return training_set, test_set

    def bschecker_pipeline(self, n_train=100, dataset="nq", level="response", train_data="nli_sliver", classifier="svm", selected_token=1, use_cache=False):
        num2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        label2num = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}
        training_set, test_set = self.get_train_test_data(dataset, level, train_data, use_cache)
        cache_file = f"cached_embeddings_nli/embeddings_{dataset}_{level}_{args.model_name.split('/')[-1]}_all.pkl"
        if not os.path.exists(cache_file) or not use_cache:
            train_template = self.get_template(f"bschecker_{level}" if train_data == "nli_sliver" else "nli")
            # zth: nli_q template will fill {question}\n{premise} in the `Premise` field, which is the only difference
            # from `nli` template since there is no question in ANLI. We only apply divide and conquer for test data.
            test_template = self.get_template(f"bschecker_{level}" if train_data == "nli_sliver" else "nli_q")
            embeddings_entailment = self.llm.get_last_token_embedding(data=training_set["entailment"], template=train_template, selected_token=selected_token)
            embeddings_neutral = self.llm.get_last_token_embedding(data=training_set["neutral"], template=train_template, selected_token=selected_token)
            embeddings_contradict = self.llm.get_last_token_embedding(data=training_set["contradict"], template=train_template, selected_token=selected_token)

            embeddings_entailment_test, chunk_size_entailment = self.llm.get_last_token_embedding(data=test_set["entailment"], template=test_template, selected_token=selected_token)
            embeddings_neutral_test, chunk_size_neutral = self.llm.get_last_token_embedding(data=test_set["neutral"], template=test_template, selected_token=selected_token)
            embeddings_contradict_test, chunk_size_contradict = self.llm.get_last_token_embedding(data=test_set["contradict"], template=test_template, selected_token=selected_token)
            if use_cache:
                print(f"saving cached data to {cache_file}...")
                pickle.dump([embeddings_entailment, embeddings_neutral, embeddings_contradict, embeddings_entailment_test, chunk_size_entailment, embeddings_neutral_test, chunk_size_neutral, embeddings_contradict_test, chunk_size_contradict], open(cache_file, "wb"))
        else:
            print(f"loading cached data from {cache_file}...")
            # num_layer x num_example x dim
            embeddings_entailment, embeddings_neutral, embeddings_contradict, embeddings_entailment_test, chunk_size_entailment, embeddings_neutral_test, chunk_size_neutral, embeddings_contradict_test, chunk_size_contradict = pickle.load(open(cache_file, "rb"))

        # zth: merge the sub-prediction results
        def process_predictions(embeddings, chunk_sizes, label_num, model):
            predictions = []
            pred_chunk = model.predict(embeddings.float().numpy())
            pred_chunk = [num2label[p] for p in pred_chunk]
            for i in range(1, len(chunk_sizes)):
                pred = merge_ret(pred_chunk[chunk_sizes[i - 1]:chunk_sizes[i]])
                predictions.append(label2num[pred])
            labels = [label_num] * len(predictions)
            return predictions, labels

        def test(k=-1):
            acc_layer = []
            f1_e_layer = []
            f1_n_layer = []
            f1_c_layer = []
            f1_m_layer = []
            pred_layer = []

            for i in range(self.llm.model.config.num_hidden_layers):
                if classifier == "knn":
                    model = KNN(k, dim=2)
                elif classifier == "svm":
                    model = SVM(kernel="rbf")
                elif classifier == "nn":
                    input_size = embeddings_entailment.size(-1)
                    hidden_size = embeddings_entailment.size(-1)
                    output_size = 3
                    alpha = 0.001
                    epochs = 300
                    batch_size = 32
                    lr = 0.001
                    model = PyTorchClassifier(input_size, hidden_size, output_size, lr=lr, alpha=alpha, epochs=epochs, batch_size=batch_size, device=args.device)
                elif classifier == "attn":
                    hidden_size = embeddings_entailment.size(-1)
                    output_size = 3
                    alpha = 1e-5
                    epochs = 10000
                    batch_size = 128
                    lr = 1e-5
                    patience = 30
                    model = AttentionClassifier(hidden_size, output_size, lr=lr, alpha=alpha, epochs=epochs, batch_size=batch_size, early_stop_patience=patience)

                # zth: if using attention model, we need combine all the layer embeddings
                if classifier != "attn":
                    train_embedding_entailment = embeddings_entailment[i][:n_train]
                    train_embedding_neutral = embeddings_neutral[i][:n_train]
                    train_embedding_contradict = embeddings_contradict[i][:n_train]
                    test_embedding_entailment = embeddings_entailment_test[i]
                    test_embedding_neutral = embeddings_neutral_test[i]
                    test_embedding_contradict = embeddings_contradict_test[i]
                else:
                    train_embedding_entailment = embeddings_entailment[:, :n_train, :].transpose(0, 1)
                    train_embedding_neutral = embeddings_neutral[:, :n_train, :].transpose(0, 1)
                    train_embedding_contradict = embeddings_contradict[:, :n_train, :].transpose(0, 1)
                    test_embedding_entailment = embeddings_entailment_test.transpose(0, 1)
                    test_embedding_neutral = embeddings_neutral_test.transpose(0, 1)
                    test_embedding_contradict = embeddings_contradict_test.transpose(0, 1)

                model.train(torch.concatenate((train_embedding_entailment, train_embedding_neutral, train_embedding_contradict), dim=0).float().numpy(), [0] * train_embedding_entailment.size(0) + [1] * train_embedding_neutral.size(0) + [2] * train_embedding_contradict.size(0))
                predictions = []
                labels = []
                datasets = [
                    (test_embedding_entailment, chunk_size_entailment, 0),
                    (test_embedding_neutral, chunk_size_neutral, 1),
                    (test_embedding_contradict, chunk_size_contradict, 2)
                ]
                for embeddings, chunk_sizes, label_num in datasets:
                    pred, label = process_predictions(embeddings, chunk_sizes, label_num, model)
                    predictions += pred
                    labels += label

                pred_layer.append([predictions, labels])

                # zth: for the other 3 level, we need to aggregate the sub-labels into response level according to the
                # identifier (data[-1][-1]) which is id+model_name
                if level != "response":
                    prediction_list = defaultdict(list)
                    label_list = defaultdict(list)
                    for idx, data in enumerate(test_set["entailment"] + test_set["neutral"] + test_set["contradict"]):
                        prediction_list[data[-1][-1]].append(num2label[predictions[idx]])
                        label_list[data[-1][-1]].append(num2label[labels[idx]])
                    predictions = [label2num[aggregate_labels(prediction_list[key])] for key in prediction_list]
                    labels = [label2num[aggregate_labels(label_list[key])] for key in prediction_list]

                evaluator = Evaluator()
                results = evaluator.evaluate(predictions, labels)
                print(f"{i}-th layer:", results)
                acc_layer.append(results["acc"])
                f1_e_layer.append(results["macro_f1"][0])
                f1_n_layer.append(results["macro_f1"][1])
                f1_c_layer.append(results["macro_f1"][2])
                f1_m_layer.append(results["macro_f1"]['macro'])
                if classifier == "attn":
                    break
            pred_layer.append(test_set)
            # pickle.dump(pred_layer, open(f"tmp/pred_layer_{k}.pkl", "wb"))

        print(dataset)
        if classifier != "knn":
            test()
        else:
            for k in [1, 2, 5, 10]:
                print(f"k {k}")
                test(k=k)

    # zth: zero-shot baseline
    def zero_shot_baseline(self, dataset, level):
        num2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        label2num = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}
        _, test_set = self.get_train_test_data(dataset, level, "anli")
        predictions = []
        labels = []
        first_idx = 0
        # zth: the first token id may be the start token, which should be skipped
        while self.llm.tokenizer("Entailment")["input_ids"][first_idx] == self.llm.tokenizer("Neutral")["input_ids"][first_idx]:
            first_idx += 1
        entailment_id = self.llm.tokenizer("Entailment")["input_ids"][first_idx]
        neutral_id = self.llm.tokenizer("Neutral")["input_ids"][first_idx]
        contradiction_id = self.llm.tokenizer("Contradiction")["input_ids"][first_idx]
        template = self.get_template("nli_q")
        print(f"prompt of sample #1:\n{template(*test_set['entailment'][0][0][:3])}")
        for key in test_set:
            for sample in tqdm(test_set[key]):
                sub_pred = []
                for sub_sample in sample:
                    prompt = template(*sub_sample[:3])
                    logits = self.llm.get_logits(prompt)
                    label_logits = torch.stack((logits[entailment_id], logits[neutral_id], logits[contradiction_id]))
                    prediction = label_logits.argmax(dim=-1).tolist()
                    sub_pred.append(num2label[prediction])
                predictions.append(label2num[merge_ret(sub_pred)])
                labels.append(label2num[key.capitalize() if key != "contradict" else "Contradiction"])

        # zth: for the other 3 level, we need to aggregate the sub-labels into response level
        if level != "response":
            prediction_list = defaultdict(list)
            label_list = defaultdict(list)
            for idx, data in enumerate(test_set["entailment"] + test_set["neutral"] + test_set["contradict"]):
                prediction_list[data[-1][-1]].append(num2label[predictions[idx]])
                label_list[data[-1][-1]].append(num2label[labels[idx]])
            predictions = [label2num[aggregate_labels(prediction_list[key])] for key in prediction_list]
            labels = [label2num[aggregate_labels(label_list[key])] for key in prediction_list]

        evaluator = Evaluator()
        results = evaluator.evaluate(predictions, labels)
        print(f"zero-shot results: {results}")

    # zth: few-shot baseline
    def few_shot_baseline(self, dataset, level, num_examples=3):
        num2label = {0: "Entailment", 1: "Neutral", 2: "Contradiction"}
        label2num = {"Entailment": 0, "Neutral": 1, "Contradiction": 2}
        training_set, test_set = self.get_train_test_data(dataset, level, "anli")
        predictions = []
        labels = []
        few_shot_examples = []
        for key in training_set:
            few_shot_examples.extend([(d[0], d[1], key.capitalize() if key != "contradict" else "Contradiction") for d in training_set[key][:num_examples]])
        random.seed(42)
        random.shuffle(few_shot_examples)
        few_shot_examples = few_shot_examples[:num_examples]
        first_idx = 0
        # zth: the first token id may be the start token, which should be skipped
        while self.llm.tokenizer("Entailment")["input_ids"][first_idx] == self.llm.tokenizer("Neutral")["input_ids"][first_idx]:
            first_idx += 1
        entailment_id = self.llm.tokenizer("Entailment")["input_ids"][first_idx]
        neutral_id = self.llm.tokenizer("Neutral")["input_ids"][first_idx]
        contradiction_id = self.llm.tokenizer("Contradiction")["input_ids"][first_idx]
        print(f"prompt of sample #1:\n{self.few_shot_template(args.model_name, *test_set['entailment'][0][0][:3], few_shot_examples)}")
        for key in test_set:
            for sample in tqdm(test_set[key]):
                sub_pred = []
                for sub_sample in sample:
                    prompt = self.few_shot_template(args.model_name, *sub_sample[:3], few_shot_examples)
                    logits = self.llm.get_logits(prompt)
                    label_logits = torch.stack(
                        (logits[entailment_id], logits[neutral_id], logits[contradiction_id]))
                    prediction = label_logits.argmax(dim=-1).tolist()
                    sub_pred.append(num2label[prediction])
                predictions.append(label2num[merge_ret(sub_pred)])
                labels.append(label2num[key.capitalize() if key != "contradict" else "Contradiction"])

        # zth: for the other 3 level, we need to aggregate the sub-labels into response level
        if level != "response":
            prediction_list = defaultdict(list)
            label_list = defaultdict(list)
            for idx, data in enumerate(test_set["entailment"] + test_set["neutral"] + test_set["contradict"]):
                prediction_list[data[-1][-1]].append(num2label[predictions[idx]])
                label_list[data[-1][-1]].append(num2label[labels[idx]])
            predictions = [label2num[aggregate_labels(prediction_list[key])] for key in prediction_list]
            labels = [label2num[aggregate_labels(label_list[key])] for key in prediction_list]

        evaluator = Evaluator()
        results = evaluator.evaluate(predictions, labels)
        print(f"few-shot results: {results}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="name")
    parser.add_argument("--classifier", type=str, default="svm")
    parser.add_argument("--dataset", type=str, default="nq")
    parser.add_argument("--level", type=str, default="triplet")
    parser.add_argument("--selected_token", type=int, default=1)
    parser.add_argument("--training_data", type=str, default="anli")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--baseline", type=str, default="")
    parser.add_argument("--n_shot", type=int, default=3)
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()
    # zth: some recommended setting
    use_cache = True
    args.model_name = "teknium/OpenHermes-2.5-Mistral-7B"
    args.training_data = "anli"
    args.dataset = "nq"
    args.level = "triplet"
    args.classifier = "svm"
    args.selected_token = 1
    args.n_train = 500
    print(args)

    repe = RepE(args.model_name, device=args.device)

    if args.baseline == "zero":
        repe.zero_shot_baseline(dataset=args.dataset, level=args.level)
    elif args.baseline == "few":
        repe.few_shot_baseline(dataset=args.dataset, level=args.level, num_examples=args.n_shot)
    else:
        repe.bschecker_pipeline(dataset=args.dataset, n_train=args.n_train, level=args.level, train_data=args.training_data, classifier=args.classifier, selected_token=args.selected_token, use_cache=use_cache)

