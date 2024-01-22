import json
import logging
import random
from typing import Optional, Sized
import numpy as np

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoTokenizer,
)
from torch.utils.data import Dataset, Sampler
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class DSTDataSet(Dataset):
    def __init__(self, dataset, model_name='bert-base-uncased', need_mlm=True, tokenizer_max_length=512) -> None:
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer_max_length = tokenizer_max_length
        self.config = AutoConfig.from_pretrained(model_name)
        self.dataset_type_dict = dict()

        self.dataset = dataset

        self.need_mlm = need_mlm

        self.dataset_type_dict_init()
    
    def dataset_type_dict_init(self):
        for i, example in enumerate(self.dataset):
            try:
                self.dataset_type_dict[example['task']].append(i)
            except:
                self.dataset_type_dict[example['task']] = [i]
    def random_word(self, tokens):
        """
        Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
        :param tokens: list of str, tokenized sentence.
        :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
        :return: (list of str, list of int), masked tokens and related labels for LM prediction
        """
        if not self.need_mlm: # disable masked language modeling
            return tokens, [-100] * len(tokens)

        output_label = []

        for i, token in enumerate(tokens):
            if token == self.tokenizer.pad_token_id:
                output_label.append(-100) # PAD tokens ignore
                continue
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.8:
                    tokens[i] = self.tokenizer.mask_token_id

                # 10% randomly change token to random token
                elif prob < 0.9:
                    tokens[i] = random.choice(list(range(self.tokenizer.vocab_size)))

                # -> rest 10% randomly keep current token

                # append current token to output (we will predict these later)
                output_label.append(token)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-100)

        return tokens, output_label
    
    def process_nli(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        tri_label = self.dataset[index]['orig_label'] if self.dataset[index]['orig_label'] != -1 else 1

        rand_self_align = random.random()
        if rand_self_align > 0.95: ### random self alignment
            text_b = self.dataset[index]['text_a']
            tri_label = 0
        elif self.dataset[index]['orig_label'] == 2 and random.random() > 0.95:
            text_a = self.dataset[index]['text_b'][0]
            text_b = self.dataset[index]['text_a']


        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(tri_label), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )

    def process_paraphrase(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        label = self.dataset[index]['orig_label']

        rand_self_align = random.random()
        if rand_self_align > 0.95: ### random self alignment
            text_b = self.dataset[index]['text_a']
            label = 1
        elif random.random() > 0.95:
            text_a = self.dataset[index]['text_b'][0]
            text_b = self.dataset[index]['text_a']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_qa(self, index):
        text_a = self.dataset[index]['text_a']
        if len(self.dataset[index]['text_c']) > 0:
            text_b = self.dataset[index]['text_b'][0] + ' ' + self.dataset[index]['text_c'][0]
        else:
            text_b = self.dataset[index]['text_b'][0]
        label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_coreference(self, index):
        text_a = self.dataset[index]['text_a']
        if len(self.dataset[index]['text_c']) > 0:
            text_b = self.dataset[index]['text_b'][0] + ' ' + self.dataset[index]['text_c'][0]
        else:
            text_b = self.dataset[index]['text_b'][0]
        label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_bin_nli(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )

    def process_fact_checking(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        tri_label = self.dataset[index]['orig_label'] if self.dataset[index]['orig_label'] != -1 else 1

        rand_self_align = random.random()
        if rand_self_align > 0.95: ### random self alignment
            text_b = self.dataset[index]['text_a']
            tri_label = 0
        elif self.dataset[index]['orig_label'] == 2 and random.random() > 0.95:
            text_a = self.dataset[index]['text_b'][0]
            text_b = self.dataset[index]['text_a']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(tri_label), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_summarization(self, index):
        text_a = self.dataset[index]['text_a']
        if random.random() > 0.5: # this will be a positive pair
            random_pos_sample_id = random.randint(0, len(self.dataset[index]['text_b'])-1)
            text_b = self.dataset[index]['text_b'][random_pos_sample_id]
            label = 1
        else: # this will be a negative pair
            label = 0
            if len(self.dataset[index]['text_c']) > 0:
                random_neg_sample_id = random.randint(0, len(self.dataset[index]['text_c'])-1)
                text_b = self.dataset[index]['text_c'][random_neg_sample_id]
            else:
                random_choose_from_entire_dataset_text_b = random.choice(self.dataset_type_dict['summarization'])
                text_b = self.dataset[random_choose_from_entire_dataset_text_b]['text_b'][0]

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_multiple_choice_qa(self, index):
        text_a = self.dataset[index]['text_a']
        if random.random() > 0.5: # this will be a positive pair
            text_b = self.dataset[index]['text_b'][0]
            label = 1
        else: # this will be a negative pair
            label = 0
            if len(self.dataset[index]['text_c']) > 0:
                random_neg_sample_id = random.randint(0, len(self.dataset[index]['text_c'])-1)
                text_b = self.dataset[index]['text_c'][random_neg_sample_id]
            else:
                random_choose_from_entire_dataset_text_b = random.choice(self.dataset_type_dict['multiple_choice_qa'])
                text_b = self.dataset[random_choose_from_entire_dataset_text_b]['text_b'][0]

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_extractive_qa(self, index):
        text_a = self.dataset[index]['text_a']
        if random.random() > 0.5: # this will be a positive pair
            random_pos_sample_id = random.randint(0, len(self.dataset[index]['text_b'])-1)
            text_b = self.dataset[index]['text_b'][random_pos_sample_id]
            label = 1
        else: # this will be a negative pair
            label = 0
            if len(self.dataset[index]['text_c']) > 0:
                random_neg_sample_id = random.randint(0, len(self.dataset[index]['text_c'])-1)
                text_b = self.dataset[index]['text_c'][random_neg_sample_id]
            else:
                random_choose_from_entire_dataset_text_b = random.choice(self.dataset_type_dict['extractive_qa'])
                text_b = self.dataset[random_choose_from_entire_dataset_text_b]['text_b'][0]

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )

    def process_ir(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][random.randint(0, len(self.dataset[index]['text_b'])-1)]
        label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(label), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(-100.0) # reg label, float
        )
    
    def process_wmt(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        reg_label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(reg_label) # reg label, float
        )
    
    def process_sts(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        reg_label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(reg_label) # reg label, float
        )

    def process_ctc(self, index):
        text_a = self.dataset[index]['text_a']
        text_b = self.dataset[index]['text_b'][0]
        reg_label = self.dataset[index]['orig_label']

        try:
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation='only_first')
        except:
            logging.warning('text_b too long...')
            tokenized_pair = self.tokenizer(text_a, text_b, padding='max_length', max_length=self.tokenizer_max_length, truncation=True)
        input_ids, mlm_labels = self.random_word(tokenized_pair['input_ids'])
        
        return (
            torch.tensor(input_ids), 
            torch.tensor(tokenized_pair['attention_mask']), 
            torch.tensor(tokenized_pair['token_type_ids']) if 'token_type_ids' in tokenized_pair.keys() else None, 
            torch.tensor(-100), # align label, 2 class
            torch.tensor(mlm_labels), # mlm label
            torch.tensor(-100), # tri label, 3 class
            torch.tensor(reg_label) # reg label, float
        )

    def __getitem__(self, index):
        if self.dataset[index]['task'] == 'nli':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_nli(index)

        if self.dataset[index]['task'] == 'bin_nli':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_bin_nli(index)
        
        if self.dataset[index]['task'] == 'paraphrase':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_paraphrase(index)

        if self.dataset[index]['task'] == 'fact_checking':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_fact_checking(index)
        
        if self.dataset[index]['task'] == 'summarization':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_summarization(index)
        
        if self.dataset[index]['task'] == 'multiple_choice_qa':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_multiple_choice_qa(index)
        
        if self.dataset[index]['task'] == 'extractive_qa':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_extractive_qa(index)
        
        if self.dataset[index]['task'] == 'qa':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_qa(index)

        if self.dataset[index]['task'] == 'coreference':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_coreference(index)

        if self.dataset[index]['task'] == 'ir':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_ir(index)
        
        if self.dataset[index]['task'] == 'sts':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_sts(index)
        
        if self.dataset[index]['task'] == 'ctc':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_ctc(index)

        if self.dataset[index]['task'] == 'wmt':
            input_ids, attention_mask, token_type_ids, align_label, mlm_labels, tri_label, reg_label = self.process_wmt(index)

        if token_type_ids is not None:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'token_type_ids': token_type_ids,
                'align_label': align_label,
                'mlm_label': mlm_labels,
                'tri_label': tri_label,
                'reg_label': reg_label
            }
        else:
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'align_label': align_label,
                'mlm_label': mlm_labels,
                'tri_label': tri_label,
                'reg_label': reg_label
            }
        

    def __len__(self):
        return len(self.dataset)

class PropSampler(Sampler[int]):
    def __init__(self, data_source: Optional[Sized]) -> None:
        super().__init__(data_source)
        self.K = 500000
        print("Initializing Prop Sampler")

        self.data_positions = dict()
        for i, example in tqdm(enumerate(data_source), desc="Initializing Sampler"):
            if example['dataset_name'] in self.data_positions.keys():
                self.data_positions[example['dataset_name']].append(i)
            else:
                self.data_positions[example['dataset_name']] = [i]
        self.all_dataset_names = list(self.data_positions.keys())
        self.dataset_lengths = {each:len(self.data_positions[each]) for each in self.data_positions}

        self.dataset_props = {each: min(self.dataset_lengths[each], self.K) for each in self.dataset_lengths}
        self.dataset_props_sum = sum([self.dataset_props[each] for each in self.dataset_props])
        


        print("Finish Prop Sampler initialization.")
        
    def __iter__(self):
        iter_list = []
        for each in self.dataset_props:
            iter_list.extend(np.random.choice(self.data_positions[each], size=self.dataset_props[each], replace=False).tolist())
        
        random.shuffle(iter_list)

        yield from iter_list
    
    def __len__(self):
        return self.dataset_props_sum

class DSTDataLoader(LightningDataModule):
    def __init__(self,dataset_config, val_dataset_config=None, sample_mode='seq', model_name='bert-base-uncased', is_finetune=False, need_mlm=True, tokenizer_max_length=512, train_batch_size=32, eval_batch_size=4, num_workers=16, train_eval_split=0.95, **kwargs):
        super().__init__(**kwargs)
        assert sample_mode in ['seq', 'proportion']
        self.sample_mode = sample_mode
        self.dataset_config = dataset_config
        self.val_dataset_config = val_dataset_config
        self.num_workers = num_workers
        self.train_eval_split = train_eval_split
        self.tokenizer_max_length = tokenizer_max_length
        self.model_name = model_name

        self.need_mlm = need_mlm
        self.is_finetune = is_finetune

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)

        self.train_bach_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is not None:
            print("Already Initilized LightningDataModule!")
            return
        
        self.init_training_set()

        self.dataset = dict()
        if not self.is_finetune:
            self.dataset['train'] = DSTDataSet(dataset=self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))], model_name=self.model_name, need_mlm=self.need_mlm)
            self.dataset['test'] = DSTDataSet(dataset=self.raw_dataset[int(self.train_eval_split*len(self.raw_dataset)):], model_name=self.model_name, need_mlm=self.need_mlm)
        else:
            self.dataset['train'] = DSTDataSet(dataset=self.raw_dataset[:], model_name=self.model_name, need_mlm=self.need_mlm)
            self.dataset['test'] = DSTDataSet(dataset=self.val_raw_dataset[:], model_name=self.model_name, need_mlm=self.need_mlm)
            
    
    def init_training_set(self):
        self.raw_dataset = []
        if self.sample_mode == 'seq':
            for each_dataset in self.dataset_config:
                dataset_length = sum([1 for line in open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8')])
                dataset_length_limit = self.dataset_config[each_dataset]['size'] if isinstance(self.dataset_config[each_dataset]['size'], int) else int(self.dataset_config[each_dataset]['size'] * dataset_length)
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            if i >= dataset_length_limit:
                                break
                            self.raw_dataset.append(json.loads(example)) ## + dataset_name
                    except:
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()
            
            random.shuffle(self.raw_dataset)
        
        elif self.sample_mode == 'proportion':
            for each_dataset in tqdm(self.dataset_config, desc="Loading data from disk..."):
                with open(self.dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    try:
                        for i, example in enumerate(f):
                            jsonobj = json.loads(example)
                            jsonobj['dataset_name'] = each_dataset
                            self.raw_dataset.append(jsonobj) ## + dataset_name
                    except:
                        print(f"failed to load data from {each_dataset}.json, exiting...")
                        exit()
            
            random.shuffle(self.raw_dataset)
        
        if self.is_finetune:
            self.val_raw_dataset = []
            for each_dataset in self.val_dataset_config:
                dataset_length = sum([1 for line in open(self.val_dataset_config[each_dataset]['data_path'], 'r', encoding='utf8')])
                dataset_length_limit = self.val_dataset_config[each_dataset]['size'] if isinstance(self.val_dataset_config[each_dataset]['size'], int) else int(self.val_dataset_config[each_dataset]['size'] * dataset_length)
                with open(self.val_dataset_config[each_dataset]['data_path'], 'r', encoding='utf8') as f:
                    for i, example in enumerate(f):
                        if i >= dataset_length_limit:
                            break
                        self.val_raw_dataset.append(json.loads(example))
            
            random.shuffle(self.val_raw_dataset)

    def prepare_data(self) -> None:
        AutoTokenizer.from_pretrained(self.model_name)
    
    def train_dataloader(self):
        if self.sample_mode == 'seq':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, shuffle=True, num_workers=self.num_workers)
        elif self.sample_mode == 'proportion':
            return DataLoader(self.dataset['train'], batch_size=self.train_bach_size, sampler=PropSampler(self.raw_dataset[:int(self.train_eval_split*len(self.raw_dataset))]), num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers)