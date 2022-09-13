import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from transformers.models.wavlm.modeling_wavlm import (
    WavLMAttention, 
    WavLMFeedForward,
    WavLMEncoderLayer, 
    WavLMEncoderLayerStableLayerNorm, 
    WavLMEncoder, 
    WavLMEncoderStableLayerNorm,
    WavLMFeatureEncoder,
    WavLMPositionalConvEmbedding,
    WavLMPreTrainedModel,
    WavLMConfig,
    WavLMModel,
    WavLMForCTC,
    WavLMForSequenceClassification,
    WavLMForAudioFrameClassification,
    WavLMForXVector,
    AMSoftmaxLoss,
    TDNNLayer,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.activations import ACT2FN
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import ModelOutput
from torch.nn import CrossEntropyLoss
from typing import List, Tuple, Union, Optional

_HIDDEN_STATES_START_POSITION = 2

'''
    ACT2FN = {
        "gelu": GELUActivation(),
        "gelu_10": ClippedGELUActivation(-10, 10),
        "gelu_fast": FastGELUActivation(),
        "gelu_new": NewGELUActivation(),
        "gelu_python": GELUActivation(use_gelu_python=True),
        "linear": LinearActivation(),
        "mish": MishActivation(),
        "quick_gelu": QuickGELUActivation(),
        "relu": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "silu": SiLUActivation(),
        "swish": SiLUActivation(),
        "tanh": nn.Tanh(),
        }
'''

class AdaWavLMConfig(WavLMConfig):
    def __init__(self, 
                 *args, 
                 adapter_embedding_size={},
                 use_adapter_attn=True,
                 use_adapter_ff=True,
                 eadapter_act='gelu',
                 ladapter_act='gelu',
                 use_adapter_to_output=False,
                 adapter_to_output_layer_size={},
                 use_adapter_to_output_weighted_sum=False,
                 adapter_dropproba=None,
                 adapter_init='normal',
                 adapter_init_std=1e-3,
                 ladapter_init_std=1e-3,
                 adapter_init_mean=0,
                 adapter_init_value=1e-5,
                 use_statistics_pooling=False,
                 use_amsoftmax=False,
                 use_residual=True,
                 use_upsampling=True,
                 use_adapter_norm=True,
                 use_adapter_fc=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.adapter_embedding_size = adapter_embedding_size
        self.use_adapter_attn = use_adapter_attn
        self.use_adapter_ff = use_adapter_ff
        self.use_adapter_to_output = use_adapter_to_output
        self.adapter_to_output_layer_size = adapter_to_output_layer_size
        self.use_adapter_to_output_weighted_sum = use_adapter_to_output_weighted_sum
        self.use_upsampling = use_upsampling
        self.use_residual = use_residual
        self.eadapter_act = eadapter_act
        self.ladapter_act = ladapter_act
        self.adapter_init = adapter_init
        self.adapter_init_std = adapter_init_std 
        self.ladapter_init_std = ladapter_init_std 
        self.adapter_init_mean = adapter_init_mean
        self.adapter_init_value = adapter_init_value
        self.use_statistics_pooling = use_statistics_pooling
        self.use_amsoftmax = use_amsoftmax
        self.adapter_dropproba = adapter_dropproba
        self.use_adapter_norm = use_adapter_norm
        self.use_adapter_fc = use_adapter_fc
        
class AdapterLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
        self.linear_down = nn.Linear(config.hidden_size, config.adapter_embedding_size[layer])
        self.act = ACT2FN[config.eadapter_act] if config.eadapter_act else None
        self.linear_up = nn.Linear(config.adapter_embedding_size[layer], config.hidden_size)
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
                
    def forward(self, hidden_states):
        res = hidden_states
        hidden_states = self.act(self.linear_down(hidden_states)) \
                            if self.act else self.linear_down(hidden_states)
        hidden_states = self.linear_up(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        hidden_states = hidden_states + res
        return hidden_states
    
class AdapterToOutputLayer(nn.Module):
    def __init__(self, config, layer):
        super().__init__()
        self.config = config
  
        if config.use_adapter_fc:
            self.linear_down = nn.Linear(config.hidden_size, config.adapter_to_output_layer_size[layer])
        self.act = ACT2FN[config.ladapter_act] if config.ladapter_act else None 
    
        if config.use_upsampling:
            self.linear_up = nn.Linear(config.adapter_to_output_layer_size[layer], config.hidden_size)
        
        if config.adapter_dropproba:
            self.dropout = nn.Dropout(config.adapter_dropproba)
        
        if config.use_adapter_norm:
            self.layernorm = nn.LayerNorm(config.output_hidden_size, eps=config.layer_norm_eps)
                
    def forward(self, hidden_states):
        res = hidden_states
        if  self.config.use_adapter_fc:
            hidden_states = self.act(self.linear_down(hidden_states)) if self.act else self.linear_down(hidden_states)
        else:
            if self.act:
                hidden_states = self.act(hidden_states)
        
        if self.config.use_upsampling:
            hidden_states = self.linear_up(hidden_states)
            if self.config.use_adapter_postact and self.config.adapter_act:
                hidden_states = self.act(hidden_states)
        
        if self.config.adapter_dropproba:
            hidden_states = self.dropout(hidden_states)
        
        if self.config.use_adapter_norm:
            hidden_states = self.layernorm(hidden_states)
            
        if self.config.use_residual and self.config.use_upsampling:
            hidden_states = hidden_states + res

        return hidden_states

class AdaWavLMEncoderLayer(nn.Module):
    def __init__(self, config: AdaWavLMConfig, layer, has_relative_position_bias: bool = True):
        super().__init__()
        self.config = config
        self.attention = WavLMAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            num_buckets=config.num_buckets,
            max_distance=config.max_bucket_distance,
            has_relative_position_bias=has_relative_position_bias,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        if config.use_adapter_attn:
            self.adapter_layer_attn = AdapterLayer(config, layer) 
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.feed_forward = WavLMFeedForward(config)
        if config.use_adapter_ff:
            self.adapter_layer_ff = AdapterLayer(config, layer)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, position_bias=None, output_attentions=False, index=0):
        attn_residual = hidden_states
        hidden_states, attn_weights, position_bias = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            output_attentions=output_attentions,
            index=index,
        )
        hidden_states = self.dropout(hidden_states)
        if self.config.use_adapter_attn:
            hidden_states = self.adapter_layer_attn(hidden_states)
        hidden_states = attn_residual + hidden_states

        hidden_states = self.layer_norm(hidden_states)
        
        res = hidden_states
        hidden_states = self.feed_forward(hidden_states)
        if self.config.use_adapter_ff:
            hidden_states = res + self.adapter_layer_ff(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states, position_bias, )

        if output_attentions:
            outputs += (attn_weights,)
        
        return outputs

class AdaWavLMEncoder(WavLMEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [AdaWavLMEncoderLayer(config, layer=str(i), has_relative_position_bias=(i == 0)) if str(i) in list(config.adapter_embedding_size.keys())\
             else WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers) ])
        
class AdaLayerToOutWavLMEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pos_conv_embed = WavLMPositionalConvEmbedding(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout)
        
        self.layers = nn.ModuleList(
            [AdaWavLMEncoderLayer(config, layer=str(i), has_relative_position_bias=(i == 0)) if str(i) in list(config.adapter_embedding_size.keys())\
            else WavLMEncoderLayer(config, has_relative_position_bias=(i == 0)) for i in range(config.num_hidden_layers) ]
            )
        
        self.adapter_to_output = nn.ModuleDict(
            {layer:AdapterToOutputLayer(config, layer) for layer in list(config.adapter_to_output_layer_size.keys())}
        )
        
        self.num_adapter_to_output= len(config.adapter_to_output_layer_size.keys())
        self.num_adapter_layer = len(config.adapter_embedding_size.keys())
        if config.use_adapter_to_output_weighted_sum:
            if self.num_adapter_to_output:
                num_adapter_to_output_layers = self.num_adapter_to_output
            else:
                num_adapter_to_output_layers = self.num_adapter_layer
            
            self.adapter_to_output_layer_weights = nn.Parameter(torch.ones(num_adapter_to_output_layers) / num_adapter_to_output_layers)
            config.layerdrop=0.0
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        residual_adapter = ()
        
        if attention_mask is not None:
            # make sure padded tokens output 0
            hidden_states[~attention_mask] = 0.0

        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        position_bias = None

        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            # don't use LayerDrop in AdaToOutputWavLM
            dropout_probability = np.random.uniform(0, 1)

            skip_the_layer = self.training and i > 0 and (dropout_probability < self.config.layerdrop)
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # under deepspeed zero3 all gpus must run in sync
                if self.gradient_checkpointing and self.training:
                    # create gradient checkpointing function
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask,
                        position_bias,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_bias=position_bias,
                        output_attentions=output_attentions,
                        index=i,
                    )

                hidden_states, position_bias = layer_outputs[:2]
                
                layer_ada_keys = list(self.config.adapter_to_output_layer_size.keys()) 
                
                # adapt output of FeedForad module
                if str(i) in layer_ada_keys:        
                    residual_adapter += (self.adapter_to_output[str(i)](hidden_states),)
                    
            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        hidden_states = torch.stack(residual_adapter, dim=1)
        
        if self.config.use_adapter_to_output_weighted_sum:
            norm_weights = F.softmax(self.adapter_to_output_layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = hidden_states.mean(dim=1)
            
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class AdaWavLMPretraiedModel(WavLMPreTrainedModel):
    config_class = AdaWavLMConfig
    base_model_prefix = "wavlm"
    main_input_name = "input_values"
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    supports_gradient_checkpointing = True
    
    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (AdaLayerToOutWavLMEncoder, AdaWavLMEncoder)): # initialize adapter layer with identity projectioin        
            for name, param in module.named_parameters():
                if 'adapter' in name:
                    if 'linear' in name:
                        if 'weight' in name:
                            if self.config.adapter_init == 'normal': 
                                # initialize adapter as near-identity function
                                param.data.normal_(mean=self.config.adapter_init_mean, std=self.config.adapter_init_std)
                            elif self.config.adapter_init == 'uniform':
                                param.data.uniform_(a=-self.config.adapter_init_value, b=self.config.adapter_init_value)
                            elif self.config.adapter_init == 'constant':
                                nn.init.constant_(param, self.config.adapter_init_value)
                            elif self.config.adapter_init == 'eye':
                                nn.init.eye_(param)
                            elif self.config.adapter_init == 'zero':
                                param.data.zero_()
                            elif self.config.adapter_init == 'he':
                                nn.init.kaiming_uniform_(param, a=math.sqrt(5)) 
                            else:
                                raise ValueError('error') 
                        elif 'bias' in name:
                            param.data.zero_()
                    else:
                        continue
                        
    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (AdaWavLMEncoder, WavLMEncoderStableLayerNorm, WavLMFeatureEncoder)):
            module.gradient_checkpointing = value

class AdaWavLMModel(WavLMModel, AdaWavLMPretraiedModel):
    def __init__(self, config: AdaWavLMConfig):
        super().__init__(config)
        if config.do_stable_layer_norm:
            self.encoder = WavLMEncoderStableLayerNorm(config)
        elif config.use_adapter_to_output:
            self.encoder = AdaLayerToOutWavLMEncoder(config)
        else:
            self.encoder = AdaWavLMEncoder(config)
        AdaWavLMPretraiedModel._init_weights(self, self.encoder)
        
            
class AdaWavLMForCTC(WavLMForCTC, AdaWavLMPretraiedModel):
    def __init__(self, config: AdaWavLMConfig):
        if not config.use_upsampling:
            config.output_hidden_size = list(config.adapter_to_output_layer_size.values())[0]
        super().__init__(config)
        self.wavlm = AdaWavLMModel(config)
        self.lm_head = nn.Linear(config.output_hidden_size, config.vocab_size)
        self.post_init()
        
class AdaWavLMForSequenceClassification(WavLMForSequenceClassification, AdaWavLMPretraiedModel):
    def __init__(self, config: AdaWavLMConfig):
        if not config.use_upsampling:
            config.output_hidden_size = list(config.adapter_to_output_layer_size.values())[0]
        super().__init__(config)
        self.wavlm = AdaWavLMModel(config)
        self.projector = nn.Linear(config.output_hidden_size, config.classifier_proj_size)
        self.post_init()
        
class AdaWavLMForXVector(WavLMForXVector, AdaWavLMPretraiedModel):
    def __init__(self, config: AdaWavLMConfig):
        super().__init__(config)
        self.wavlm = AdaWavLMModel(config)

class AdaWavLMForAudioFrameClassification(WavLMForAudioFrameClassification, AdaWavLMPretraiedModel):
    def __init__(self, config: AdaWavLMConfig):
        super().__init__(config)
        self.wavlm = AdaWavLMModel(config)
            
            
class SpeakerVerificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    embeddings: torch.FloatTensor = None

class AdaWavLMForSpeakerVerification(AdaWavLMForSequenceClassification):
    def __init__(self, config):
        if not config.use_upsampling:
            config.output_hidden_size = list(config.adapter_to_output_layer_size.values())[0]
        super().__init__(config)
        self.config = config
        if config.use_statistics_pooling:
            self.classifier = nn.Linear(config.classifier_proj_size*2, config.num_labels)
        if config.use_amsoftmax:
            self.objective = AMSoftmaxLoss(config.num_labels, config.num_labels)
        
        self.projector = nn.Linear(config.output_hidden_size, config.classifier_proj_size)
        
        self.post_init()
        if self.config.use_statistics_pooling:
            self.objective = AMSoftmaxLoss(config.classifier_proj_size*2, config.num_labels)
        else:
            self.objective = AMSoftmaxLoss(config.classifier_proj_size, config.num_labels)
        
        
    def _get_feature_vector_attention_mask(self, 
                                           feature_vector_length: int,
                                           attention_mask: torch.LongTensor,
                                           add_adapter=None):
        # Effectively attention_mask.sum(-1), but not inplace to be able to run
        # on inference mode.
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        batch_size = attention_mask.shape[0]

        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # these two operations makes sure that all values before the output lengths idxs are attended to
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        
        return attention_mask, output_lengths
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SpeakerVerificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]
        
        hidden_states = self.projector(hidden_states)
        
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask, output_lengths = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            if self.config.use_statistics_pooling:
                mean_features = []; std_features = []
                for i, length in enumerate(output_lengths):
                    mean_features.append(hidden_states[i, :length].mean(dim=0))
                    std_features.append(hidden_states[i, :length].std(dim=0))
                mean_features = torch.stack(mean_features)
                std_features = torch.stack(std_features)
                pooled_output = torch.cat([mean_features, std_features], dim=-1)
                
            else:
                hidden_states[~padding_mask] = 0.0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.config.use_amsoftmax:
                loss = self.objective(logits, labels) 
            else:        
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SpeakerVerificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=pooled_output
        )

# for fine-tuning
class WavLMForSpeakerVerification(WavLMForSequenceClassification):
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SpeakerVerificationOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
        
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SpeakerVerificationOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            embeddings=pooled_output
        )
