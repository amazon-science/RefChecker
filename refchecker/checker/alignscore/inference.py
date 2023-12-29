from logging import warning
import spacy
from nltk.tokenize import sent_tokenize
import torch
from .model import BERTAlignModel
from transformers import AutoConfig, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm

class Inferencer():
    def __init__(self, ckpt_path, model='bert-base-uncased', batch_size=32, device='cuda', verbose=True) -> None:
        self.device = device
        if ckpt_path is not None:
            self.model = BERTAlignModel(model=model).load_from_checkpoint(checkpoint_path=ckpt_path, strict=False).to(self.device)
        else:
            warning('loading UNTRAINED model!')
            self.model = BERTAlignModel(model=model).to(self.device)
        self.model.eval()
        self.batch_size = batch_size

        self.config = AutoConfig.from_pretrained(model)
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.spacy = spacy.load('en_core_web_sm')

        self.loss_fct = nn.CrossEntropyLoss(reduction='none')
        self.softmax = nn.Softmax(dim=-1)

        self.smart_type = 'smart-n'
        self.smart_n_metric = 'f1'

        self.disable_progress_bar_in_inference = False

        self.nlg_eval_mode = None # bin, bin_sp, nli, nli_sp
        self.verbose = verbose
    
    def inference_example_batch(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SummaC Style aggregation
        """
        self.disable_progress_bar_in_inference = True
        assert len(premise) == len(hypo), "Premise must has the same length with Hypothesis!"

        out_score = []
        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc="Evaluating", total=len(premise), disable=(not self.verbose)):
            out_score.append(self.inference_per_example(one_pre, one_hypo))
        
        return None, torch.tensor(out_score), None

    def inference_per_example(self, premise:str, hypo: str):
        """
        inference a example,
        premise: string
        hypo: string
        using self.inference to batch the process
        """
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield ' '.join(lst[i:i + n])
        
        premise_sents = sent_tokenize(premise)
        premise_sents = premise_sents or ['']

        n_chunk = len(premise.strip().split()) // 350 + 1
        n_chunk = max(len(premise_sents) // n_chunk, 1)
        premise_sents = [each for each in chunks(premise_sents, n_chunk)]

        hypo_sents = sent_tokenize(hypo)

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])
        
        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == 'nli_sp':
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:,0] ### use NLI head OR ALIGN head
            elif self.nlg_eval_mode == 'bin_sp':
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[1] ### use NLI head OR ALIGN head
            elif self.nlg_eval_mode == 'reg_sp':
                output_score = self.inference(premise_sent_mat, hypo_sents_mat)[0] ### use NLI head OR ALIGN head
            
            output_score = output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item() ### sum or mean depends on the task/aspect
            return output_score

        
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:,0] ### use NLI head OR ALIGN head
        output_score = output_score.view(len(premise_sents), len(hypo_sents)).max(dim=0).values.mean().item() ### sum or mean depends on the task/aspect

        return output_score


    def inference(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score_reg = []
        output_score_bin = []
        output_score_tri = []

        for mini_batch in tqdm(batch, desc="Evaluating", disable=not self.verbose or self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch)
                model_output_reg = model_output.reg_label_logits.cpu()
                model_output_bin = model_output.seq_relationship_logits # Temperature Scaling / 2.5
                model_output_tri = model_output.tri_label_logits
                
                model_output_bin = self.softmax(model_output_bin).cpu()
                model_output_tri = self.softmax(model_output_tri).cpu()
            output_score_reg.append(model_output_reg[:,0])
            output_score_bin.append(model_output_bin[:,1])
            output_score_tri.append(model_output_tri[:,:])
        
        output_score_reg = torch.cat(output_score_reg)
        output_score_bin = torch.cat(output_score_bin)
        output_score_tri = torch.cat(output_score_tri)
        
        if self.nlg_eval_mode is not None:
            if self.nlg_eval_mode == 'nli':
                output_score_nli = output_score_tri[:,0]
                return None, output_score_nli, None
            elif self.nlg_eval_mode == 'bin':
                return None, output_score_bin, None
            elif self.nlg_eval_mode == 'reg':
                return None, output_score_reg, None
            else:
                ValueError("unrecognized nlg eval mode")

        
        return output_score_reg, output_score_bin, output_score_tri
    
    def inference_reg(self, premise, hypo):
        """
        inference a list of premise and hypo

        Standard aggregation
        """
        self.model.is_reg_finetune = True
        if isinstance(premise, str) and isinstance(hypo, str):
            premise = [premise]
            hypo = [hypo]
        
        batch = self.batch_tokenize(premise, hypo)
        output_score = []

        for mini_batch in tqdm(batch, desc="Evaluating", disable=self.disable_progress_bar_in_inference):
            mini_batch = mini_batch.to(self.device)
            with torch.no_grad():
                model_output = self.model(mini_batch).seq_relationship_logits.cpu().view(-1)
            output_score.append(model_output)
        output_score = torch.cat(output_score)
        return output_score
    
    def batch_tokenize(self, premise, hypo):
        """
        input premise and hypos are lists
        """
        assert isinstance(premise, list) and isinstance(hypo, list)
        assert len(premise) == len(hypo), "premise and hypo should be in the same length."

        batch = []
        for mini_batch_pre, mini_batch_hypo in zip(self.chunks(premise, self.batch_size), self.chunks(hypo, self.batch_size)):
            try:
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation='only_first', padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            except:
                warning('text_b too long...')
                mini_batch = self.tokenizer(mini_batch_pre, mini_batch_hypo, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')
            batch.append(mini_batch)

        return batch
    def smart_doc(self, premise: list, hypo: list):
        """
        inference a example,
        premise: list
        hypo: list
        using self.inference to batch the process

        SMART Style aggregation
        """
        self.disable_progress_bar_in_inference = True
        assert len(premise) == len(hypo), "Premise must has the same length with Hypothesis!"
        assert self.smart_type in ['smart-n', 'smart-l']

        out_score = []
        for one_pre, one_hypo in tqdm(zip(premise, hypo), desc="Evaluating SMART", total=len(premise)):
            out_score.append(self.smart_l(one_pre, one_hypo)[1] if self.smart_type == 'smart-l' else self.smart_n(one_pre, one_hypo)[1])
        
        return None, torch.tensor(out_score), None

    def smart_l(self, premise, hypo):
        premise_sents = [each.text for each in self.spacy(premise).sents]
        hypo_sents = [each.text for each in self.spacy(hypo).sents]

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])
        
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:,0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))

        ### smart-l
        lcs = [[0] * (len(hypo_sents)+1)] * (len(premise_sents)+1)
        for i in range(len(premise_sents)+1):
            for j in range(len(hypo_sents)+1):
                if i != 0 and j != 0:
                    m = output_score[i-1, j-1]
                    lcs[i][j] = max([lcs[i-1][j-1]+m,
                                        lcs[i-1][j]+m,
                                        lcs[i][j-1]])

        return None, lcs[-1][-1] / len(premise_sents), None
    
    def smart_n(self, premise, hypo):
        ### smart-n
        n_gram = 1

        premise_sents = [each.text for each in self.spacy(premise).sents]
        hypo_sents = [each.text for each in self.spacy(hypo).sents]

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])
        
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:,0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))
        
        prec = sum([max([sum([output_score[i+n, j+n]/n_gram for n in range(0, n_gram)]) for i in range(len(premise_sents)-n_gram+1)]) for j in range(len(hypo_sents)-n_gram+1)])
        prec = prec / (len(hypo_sents) - n_gram + 1) if (len(hypo_sents) - n_gram + 1) > 0 else 0.


        premise_sents = [each.text for each in self.spacy(hypo).sents]# simple change
        hypo_sents = [each.text for each in self.spacy(premise).sents]#

        premise_sent_mat = []
        hypo_sents_mat = []
        for i in range(len(premise_sents)):
            for j in range(len(hypo_sents)):
                premise_sent_mat.append(premise_sents[i])
                hypo_sents_mat.append(hypo_sents[j])
        
        output_score = self.inference(premise_sent_mat, hypo_sents_mat)[2][:,0]
        output_score = output_score.view(len(premise_sents), len(hypo_sents))

        recall = sum([max([sum([output_score[i+n, j+n]/n_gram for n in range(0, n_gram)]) for i in range(len(premise_sents)-n_gram+1)]) for j in range(len(hypo_sents)-n_gram+1)])
        recall = prec / (len(hypo_sents) - n_gram + 1) if (len(hypo_sents) - n_gram + 1) > 0 else 0.

        f1 = 2 * prec * recall / (prec + recall)

        if self.smart_n_metric == 'f1':
            return None, f1, None
        elif self.smart_n_metric == 'precision':
            return None, prec, None
        elif self.smart_n_metric == 'recall':
            return None, recall, None
        else:
            ValueError("SMART return type error")

    def chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]
    
    def nlg_eval(self, premise, hypo):
        assert self.nlg_eval_mode is not None, "Select NLG Eval mode!"
        if (self.nlg_eval_mode == 'bin') or (self.nlg_eval_mode == 'nli') or (self.nlg_eval_mode == 'reg'):
            return self.inference(premise, hypo)
        
        elif (self.nlg_eval_mode == 'bin_sp') or (self.nlg_eval_mode == 'nli_sp') or (self.nlg_eval_mode == 'reg_sp'):
            return self.inference_example_batch(premise, hypo)
        
        else:
            ValueError("Unrecognized NLG Eval mode!")
