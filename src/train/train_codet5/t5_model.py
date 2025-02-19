import torch
from transformers import T5Config, T5ForConditionalGeneration
                          
from transformers.models.t5.modeling_t5 import T5ClassificationHead
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput

from typing import Optional, Union, Tuple
from loguru import logger 


class CodeT5ForClassificationAndGeneration(T5ForConditionalGeneration):

    def __init__(self, config: T5Config, mode=None):
        super(CodeT5ForClassificationAndGeneration, self).__init__(config)
        self.mode = None
        if mode:
            self.set_model_mode(mode)
        self.config = config
        # classification head
        
        self.sequence_label_classification_head = T5ClassificationHead(
            self.config
        )

    def set_model_mode(self, mode):
        assert mode in ['generation', 'identifier_cls']
        self.mode = mode
        logger.info(f'BART mode switched to {mode}')

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        
        assert self.mode, 'It is required to specific a mode for BART before the model is passed through'

        if self.mode == 'generation':
            return self.forward_gen(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)

        elif self.mode == 'identifier_cls':
            return self.forward_cls(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=decoder_input_ids,
                                    decoder_attention_mask=decoder_attention_mask,
                                    head_mask=head_mask,
                                    decoder_head_mask=decoder_head_mask,
                                    cross_attn_head_mask=cross_attn_head_mask,
                                    encoder_outputs=encoder_outputs,
                                    past_key_values=past_key_values,
                                    inputs_embeds=inputs_embeds,
                                    decoder_inputs_embeds=decoder_inputs_embeds,
                                    labels=labels,
                                    use_cache=use_cache,
                                    output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states,
                                    return_dict=return_dict)

    def forward_gen(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # breakpoint()
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            # loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def forward_cls(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None
    ):
        encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = encoder_outputs[0]
        
        lm_logits = self.sequence_label_classification_head(hidden_states)

        return SequenceClassifierOutput(
            logits=lm_logits,
            hidden_states=hidden_states
        )
