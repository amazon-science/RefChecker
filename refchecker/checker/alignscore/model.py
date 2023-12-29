import math
from typing import Optional, Tuple
from transformers import AdamW, get_linear_schedule_with_warmup, AutoConfig
from transformers import BertForPreTraining, BertModel, RobertaModel, AlbertModel, AlbertForMaskedLM, RobertaForMaskedLM
import torch
import torch.nn as nn
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from dataclasses import dataclass



class BERTAlignModel(pl.LightningModule):
    def __init__(self, model='bert-base-uncased', using_pretrained=True, *args, **kwargs) -> None:
        super().__init__()
        # Already defined in lightning: self.device
        self.save_hyperparameters()
        self.model = model

        if 'muppet' in model:
            assert using_pretrained == True, "Only support pretrained muppet!"
            self.base_model = RobertaModel.from_pretrained(model)
            self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head
            
        elif 'roberta' in model:
            if using_pretrained:
                self.base_model = RobertaModel.from_pretrained(model)
                self.mlm_head = RobertaForMaskedLM.from_pretrained(model).lm_head
            else:
                self.base_model = RobertaModel(AutoConfig.from_pretrained(model))
                self.mlm_head = RobertaForMaskedLM(AutoConfig.from_pretrained(model)).lm_head
            
        elif 'albert' in model:
            if using_pretrained:
                self.base_model = AlbertModel.from_pretrained(model)
                self.mlm_head = AlbertForMaskedLM.from_pretrained(model).predictions
            else:
                self.base_model = AlbertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = AlbertForMaskedLM(AutoConfig.from_pretrained(model)).predictions
            
        elif 'bert' in model:
            if using_pretrained:
                self.base_model = BertModel.from_pretrained(model)
                self.mlm_head = BertForPreTraining.from_pretrained(model).cls.predictions
            else:
                self.base_model = BertModel(AutoConfig.from_pretrained(model))
                self.mlm_head = BertForPreTraining(AutoConfig.from_pretrained(model)).cls.predictions

        elif 'electra' in model:
            self.generator = BertModel(AutoConfig.from_pretrained('prajjwal1/bert-small'))
            self.generator_mlm = BertForPreTraining(AutoConfig.from_pretrained('prajjwal1/bert-small')).cls.predictions

            self.base_model = BertModel(AutoConfig.from_pretrained('bert-base-uncased'))
            self.discriminator_predictor = ElectraDiscriminatorPredictions(self.base_model.config)
            
    
        self.bin_layer = nn.Linear(self.base_model.config.hidden_size, 2)
        self.tri_layer = nn.Linear(self.base_model.config.hidden_size, 3)
        self.reg_layer = nn.Linear(self.base_model.config.hidden_size, 1)

        self.dropout = nn.Dropout(p=0.1)
        
        self.need_mlm = True
        self.is_finetune = False
        self.mlm_loss_factor = 0.5

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, batch):
        if 'electra' in self.model:
            return self.electra_forward(batch)
        base_model_output = self.base_model(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None
            )
        
        prediction_scores = self.mlm_head(base_model_output.last_hidden_state) ## sequence_output for mlm
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output)) ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        total_loss = None
        if 'mlm_label' in batch.keys(): ### 'mlm_label' and 'align_label' when training
            ce_loss_fct = nn.CrossEntropyLoss(reduction='sum')
            masked_lm_loss = ce_loss_fct(prediction_scores.view(-1, self.base_model.config.vocab_size), batch['mlm_label'].view(-1)) #/ self.con vocabulary
            next_sentence_loss = ce_loss_fct(seq_relationship_score.view(-1, 2), batch['align_label'].view(-1)) / math.log(2)
            tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch['tri_label'].view(-1)) / math.log(3)
            reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch['reg_label'].view(-1), reduction='sum')

            masked_lm_loss_num = torch.sum(batch['mlm_label'].view(-1) != -100)
            next_sentence_loss_num = torch.sum(batch['align_label'].view(-1) != -100)
            tri_label_loss_num = torch.sum(batch['tri_label'].view(-1) != -100)
            reg_label_loss_num = torch.sum(batch['reg_label'].view(-1) != -100.0)

        return ModelOutput(
            loss=total_loss,
            all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss]  if 'mlm_label' in batch.keys() else None,
            loss_nums=[masked_lm_loss_num, next_sentence_loss_num, tri_label_loss_num, reg_label_loss_num] if 'mlm_label' in batch.keys() else None,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

    def electra_forward(self, batch):
        if 'mlm_label' in batch.keys():
            ce_loss_fct = nn.CrossEntropyLoss()
            generator_output = self.generator_mlm(self.generator(
                input_ids = batch['input_ids'],
                attention_mask = batch['attention_mask'],
                token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None
            ).last_hidden_state)
            masked_lm_loss = ce_loss_fct(generator_output.view(-1, self.generator.config.vocab_size), batch['mlm_label'].view(-1))

            hallucinated_tokens = batch['input_ids'].clone()

            hallucinated_tokens[batch['mlm_label']!=-100] = torch.argmax(generator_output, dim=-1)[batch['mlm_label']!=-100]
            replaced_token_label = (batch['input_ids'] == hallucinated_tokens).long()#.type(torch.LongTensor) #[batch['mlm_label'] == -100] = -100
            replaced_token_label[batch['mlm_label']!=-100] = (batch['mlm_label'] == hallucinated_tokens)[batch['mlm_label']!=-100].long()
            replaced_token_label[batch['input_ids'] == 0] = -100 ### ignore paddings

        base_model_output = self.base_model(
            input_ids = hallucinated_tokens if 'mlm_label' in batch.keys() else batch['input_ids'],
            attention_mask = batch['attention_mask'],
            token_type_ids = batch['token_type_ids'] if 'token_type_ids' in batch.keys() else None
        )
        hallu_detect_score = self.discriminator_predictor(base_model_output.last_hidden_state)
        seq_relationship_score = self.bin_layer(self.dropout(base_model_output.pooler_output)) ## pooled output for classification
        tri_label_score = self.tri_layer(self.dropout(base_model_output.pooler_output))
        reg_label_score = self.reg_layer(base_model_output.pooler_output)

        total_loss = None

        if 'mlm_label' in batch.keys(): ### 'mlm_label' and 'align_label' when training
            total_loss = []
            ce_loss_fct = nn.CrossEntropyLoss()
            hallu_detect_loss = ce_loss_fct(hallu_detect_score.view(-1,2),replaced_token_label.view(-1))
            next_sentence_loss = ce_loss_fct(seq_relationship_score.view(-1, 2), batch['align_label'].view(-1))
            tri_label_loss = ce_loss_fct(tri_label_score.view(-1, 3), batch['tri_label'].view(-1))
            reg_label_loss = self.mse_loss(reg_label_score.view(-1), batch['reg_label'].view(-1))

            total_loss.append(10.0 * hallu_detect_loss if not torch.isnan(hallu_detect_loss).item() else 0.)
            total_loss.append(0.2 * masked_lm_loss if (not torch.isnan(masked_lm_loss).item() and self.need_mlm) else 0.)
            total_loss.append(next_sentence_loss if not torch.isnan(next_sentence_loss).item() else 0.)
            total_loss.append(tri_label_loss if not torch.isnan(tri_label_loss).item() else 0.)
            total_loss.append(reg_label_loss if not torch.isnan(reg_label_loss).item() else 0.)

            total_loss = sum(total_loss)

        return ModelOutput(
            loss=total_loss,
            all_loss=[masked_lm_loss, next_sentence_loss, tri_label_loss, reg_label_loss, hallu_detect_loss]  if 'mlm_label' in batch.keys() else None,
            prediction_logits=hallu_detect_score,
            seq_relationship_logits=seq_relationship_score,
            tri_label_logits=tri_label_score,
            reg_label_logits=reg_label_score,
            hidden_states=base_model_output.hidden_states,
            attentions=base_model_output.attentions
        )

    def training_step(self, train_batch, batch_idx):
        output = self(train_batch)

        return {'losses': output.all_loss, 'loss_nums': output.loss_nums}

    def training_step_end(self, step_output):
        losses = step_output['losses']
        loss_nums = step_output['loss_nums']
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

        loss_mlm_num = torch.sum(loss_nums[0])
        loss_bin_num = torch.sum(loss_nums[1])
        loss_tri_num = torch.sum(loss_nums[2])
        loss_reg_num = torch.sum(loss_nums[3])

        loss_mlm = torch.sum(losses[0]) / loss_mlm_num if loss_mlm_num > 0 else 0.
        loss_bin = torch.sum(losses[1]) / loss_bin_num if loss_bin_num > 0 else 0.
        loss_tri = torch.sum(losses[2]) / loss_tri_num if loss_tri_num > 0 else 0.
        loss_reg = torch.sum(losses[3]) / loss_reg_num if loss_reg_num > 0 else 0.

        total_loss = self.mlm_loss_factor * loss_mlm + loss_bin + loss_tri + loss_reg

        self.log('train_loss', total_loss)# , sync_dist=True
        self.log('mlm_loss', loss_mlm)
        self.log('bin_label_loss', loss_bin)
        self.log('tri_label_loss', loss_tri)
        self.log('reg_label_loss', loss_reg)

        return total_loss
    
    def validation_step(self, val_batch, batch_idx):
        if not self.is_finetune:
            with torch.no_grad():
                output = self(val_batch)

            return {'losses': output.all_loss, 'loss_nums': output.loss_nums}

        with torch.no_grad():
            output = self(val_batch)['seq_relationship_logits']
            output = self.softmax(output)[:, 1].tolist()
            pred = [int(align_prob>0.5) for align_prob in output]

            labels = val_batch['align_label'].tolist()

        return {"pred": pred, 'labels': labels}#, "preds":preds, "labels":x['labels']}

    def validation_step_end(self, step_output):
        losses = step_output['losses']
        loss_nums = step_output['loss_nums']
        assert len(loss_nums) == len(losses), 'loss_num should be the same length as losses'

        loss_mlm_num = torch.sum(loss_nums[0])
        loss_bin_num = torch.sum(loss_nums[1])
        loss_tri_num = torch.sum(loss_nums[2])
        loss_reg_num = torch.sum(loss_nums[3])

        loss_mlm = torch.sum(losses[0]) / loss_mlm_num if loss_mlm_num > 0 else 0.
        loss_bin = torch.sum(losses[1]) / loss_bin_num if loss_bin_num > 0 else 0.
        loss_tri = torch.sum(losses[2]) / loss_tri_num if loss_tri_num > 0 else 0.
        loss_reg = torch.sum(losses[3]) / loss_reg_num if loss_reg_num > 0 else 0.

        total_loss = self.mlm_loss_factor * loss_mlm + loss_bin + loss_tri + loss_reg

        self.log('train_loss', total_loss)# , sync_dist=True
        self.log('mlm_loss', loss_mlm)
        self.log('bin_label_loss', loss_bin)
        self.log('tri_label_loss', loss_tri)
        self.log('reg_label_loss', loss_reg)

        return total_loss

    def validation_epoch_end(self, outputs):
        if not self.is_finetune:
            total_loss = torch.stack(outputs).mean()
            self.log("val_loss", total_loss, prog_bar=True, sync_dist=True)
        
        else:
            all_predictions = []
            all_labels = []
            for each_output in outputs:
                all_predictions.extend(each_output['pred'])
                all_labels.extend(each_output['labels'])
            
            self.log("f1", f1_score(all_labels, all_predictions), prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.hparams.warmup_steps_portion * self.trainer.estimated_stepping_batches),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def mse_loss(self, input, target, ignored_index=-100.0, reduction='mean'):
        mask = (target == ignored_index)
        out = (input[~mask]-target[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "sum":
            return out.sum()

class ElectraDiscriminatorPredictions(nn.Module):
    """Prediction module for the discriminator, made up of two dense layers."""

    def __init__(self, config):
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.gelu = nn.GELU()

    def forward(self, discriminator_hidden_states):
        hidden_states = self.dense(discriminator_hidden_states)
        hidden_states = self.gelu(hidden_states)
        logits = self.dense_prediction(hidden_states).squeeze(-1)

        return logits

@dataclass
class ModelOutput():
    loss: Optional[torch.FloatTensor] = None
    all_loss: Optional[list] = None
    loss_nums: Optional[list] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    tri_label_logits: torch.FloatTensor = None
    reg_label_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None