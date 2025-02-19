import os 
import json 
import torch
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import (T5Config, T5ForConditionalGeneration, T5Tokenizer,
                          T5ForSequenceClassification, RobertaModel,ElectraModel,
                          RobertaForMaskedLM, PreTrainedModel, RobertaModel,ElectraForMaskedLM,
                            RobertaConfig
                          )
from transformers.models.roberta.modeling_roberta import RobertaLMHead,RobertaClassificationHead,RobertaForTokenClassification
from transformers.models.electra.modeling_electra import ElectraGeneratorPredictions

from transformers.modeling_outputs import (Seq2SeqLMOutput, MaskedLMOutput,SequenceClassifierOutput,
                        Seq2SeqSequenceClassifierOutput, SequenceClassifierOutput)

from typing import Optional, Union, List, Dict, Tuple
from loguru import logger 


class MaskGenerator(PreTrainedModel):
    def __init__(self, config) -> None:
        super(MaskGenerator, self).__init__(config)
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.post_init()
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        mlm_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        labels: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        generator_hidden_states = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        generator_sequence_output = generator_hidden_states[0]
        prediction_scores = self.lm_head(generator_sequence_output)

        loss = None 
        if mlm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), mlm_labels.view(-1))
        
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=generator_hidden_states.hidden_states,
            attentions=generator_hidden_states.attentions,
        )


class Discrimiator(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, 2)
        self.bias = nn.Parameter(torch.zeros(2))
        self.decoder.bias = self.bias
        self.act = nn.GELU()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.act(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x 

class CodeBERTForClassification(PreTrainedModel):
    def __init__(self, config: RobertaConfig, generator=None, mode=None):
        super(CodeBERTForClassification, self).__init__(config)
        self.mode = None
        if mode:
            self.set_model_mode(mode)
        self.config = config
        # classification head
        config.output_hidden_states = True
        config.output_attentions = True
        
        self.encoder = RobertaModel(config)
        self.mlm_head = RobertaLMHead(config)
        
        self.discrimiator = Discrimiator(config)
        
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0.,1.)
        self.generator = generator

        
    def set_model_mode(self, mode):
        assert mode in ['mlm', 'rtd']
        self.mode = mode
        logger.info(f'codebert mode switched to {mode}')

    def sample(self, logits):
        gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
        return (logits + gumbel).argmax(dim=-1)

    def save_encoder(self, path): 
        torch.save(self.encoder.state_dict(), path)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        mlm_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        mlm_labels: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_only_rtd_loss = False,
        reduction='mean',
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        
        if self.mode == 'mlm':
            outputs = self.encoder(
                input_ids,
                attention_mask=attention_mask,
            )
            sequence_output = outputs[0]
            
            prediction_scores = self.mlm_head(sequence_output)

            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
            
            return MaskedLMOutput(
                loss=loss,
                logits=prediction_scores,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            
            
        elif self.mode == 'rtd':
            mlm_gen_output = self.generator(input_ids, attention_mask, mlm_labels=mlm_labels)
            # generator generate tokens
            
            mlm_gen_logits = mlm_gen_output.logits # (B, L, vocab size)
            mlm_gen_logits = mlm_gen_logits[mlm_mask, :]
            
            generator_loss = mlm_gen_output.loss
                        
            with torch.no_grad():
                pred_toks = self.sample(mlm_gen_logits) # ( #mlm_positions, )
                # sample random tokens
                
                generated = input_ids.clone()
                generated[mlm_mask] = pred_toks
                # replace original token with random token
                
                # produce labels for discriminator, if not equal origianl token, set is_replace to 1
                is_replaced = mlm_mask.clone()
                is_replaced[mlm_mask] = (pred_toks != mlm_labels[mlm_mask]) # (B,L)
                            
            outputs = self.encoder(
                generated,
                attention_mask=attention_mask,
            )
            sequence_output = outputs[0]
            
            prediction_scores = self.discrimiator(sequence_output)
            # discriminator determines whether the token has been replaced
            
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)

            is_replaced = is_replaced.to(torch.int64)
            # ignore padding tokens
            is_replaced[~attention_mask.bool()] = -100
            # ignore special tokens
            is_replaced[generated == 0] = -100
            is_replaced[generated == 2] = -100
                        
            loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), is_replaced.view(-1), )

            
            if not return_only_rtd_loss:
                return SequenceClassifierOutput(
                    loss=loss + generator_loss,
                    logits=prediction_scores,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=prediction_scores,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )  
            