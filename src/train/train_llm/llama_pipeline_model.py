
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm, LlamaConfig, LlamaForCausalLM
import deepspeed
from deepspeed.pipe import PipelineModule, LayerSpec


class EmbeddingPipe(torch.nn.Embedding):
    def forward(self, args):
        input_ids, position_ids, attention_mask = args
        inputs_embeds = super().forward(input_ids)
        return (inputs_embeds, position_ids, attention_mask)


class ParallelTransformerLayerPipe(LlamaDecoderLayer):
    def __init__(self, config: LlamaConfig, layer_index: int, activation_checkpointing: bool = False):
        super().__init__(config, layer_index)
        
        self.activation_checkpointing = activation_checkpointing

    def forward(self, args):
        if self.activation_checkpointing:
            return self._ckpt_forward(args)

        hidden_states, position_ids, attention_mask = args
        
        outputs = LlamaDecoderLayer.forward(self,
                                            hidden_states,
                                            attention_mask,
                                            position_ids,
                                            )
        return (outputs[0], position_ids, attention_mask)

    def _ckpt_forward(self, args):
        hidden_states, position_ids, attention_mask  = args

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return LlamaDecoderLayer.forward(module, *inputs)

            return custom_forward

        # deepspeed checkpoint auto use outputs[0] if len(outputs) == 1
        outputs = deepspeed.checkpointing.checkpoint(
            create_custom_forward(self),
            hidden_states,
            attention_mask,
            position_ids,
            None,
        )
        # layer_outputs = torch.utils.checkpoint.checkpoint(
        #     create_custom_forward(self),
        #     hidden_states,
        #     attention_mask,
        #     position_ids,
        #     None,
        # )

        return outputs, position_ids, attention_mask, 


class LayerNormPipe(LlamaRMSNorm):
    def forward(self, args):
        hidden_states, *_ = args
        last_hidden_states = super().forward(hidden_states)
        return (last_hidden_states,)


class LMLayerPipe(torch.nn.Linear):
    def forward(self, args):
        hidden_states, = args
        logits = super().forward(hidden_states)
        return (logits,)


def loss_fn(outputs, labels):
    # print(outputs)
    logits, = outputs
    
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    loss_fct = torch.nn.CrossEntropyLoss()
    # print(shift_labels.size())
    # print(shift_logits.size())
    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

    return loss


def get_model(model_config: LlamaConfig, args, partition_method="type:ParallelTransformerLayerPipe", **kwargs):
    activation_checkpointing = True 
    layers=[
            LayerSpec(EmbeddingPipe, model_config.vocab_size, model_config.hidden_size),
            *[LayerSpec(ParallelTransformerLayerPipe, model_config, idx, activation_checkpointing)
                for idx in range(model_config.num_hidden_layers)],
            LayerSpec(LayerNormPipe, model_config.hidden_size, model_config.rms_norm_eps),
            LayerSpec(LMLayerPipe, model_config.hidden_size, model_config.vocab_size, bias=False),
        ]
            
    return PipelineModule(layers,
                        loss_fn=loss_fn,
                        num_stages=args.pipe_parallel_size,
                        base_seed=args.seed,
                        partition_method=partition_method,
                        activation_checkpoint_interval=args.gradient_checkpoint_interval,
                        **kwargs)