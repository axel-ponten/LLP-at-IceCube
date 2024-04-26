""" Transformer encoder used in LLP gap reconstruction.

    The script is rewritten version of Thorsten Gluesenkamp's
    custom transformer implementation.

    This model is used to encode the input DOM summary statistics
    that can be passed to a conditional normalizing flow using e.g. jammyflows.

    Since IceCube events have varying number of hit DOMs,
    the input list of DOM summary statistics has varying sequence lengths.
    To handle this, the model uses xformers mha implementation which allows
    variable sequence length in the same batch.

    The script contains four classes:
    - LLPTransformerModel
        - top-level class that embeds the input
        and passes it to the transformer encoder
        and aggregates the output
    - CustomTransformerEncoder
        - ModuleList of CustomTransformerEncoderLayer
    - CustomTransformerEncoderLayer
        - Multihead attention encoder layer using xformers
    - qkv_projector
        - Multi-layer perceptron for input projection

    Author: Axel Ponten modified code from Thorsten Gluesenkamp
    Year: 2024
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.module import Module
from torch.nn.modules.container import ModuleList
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from argparse import Namespace
import copy
from typing import Optional, Callable

try:
    import xformers
except:
    print("cannot import xformers.. install package via pip!")
else:
    globals()["xformers"] = xformers
    from xformers.ops import fmha

# custom modules
from skip_mlp import SkipMLP
import config_parser

class LLPTransformerModel(nn.Module):
    """ Transformer encoder for IceCube events.
        Input is a vector of DOM summary statistics.
        Output is an encoding that can be passed to a conditional normalizing flow, such as jammyflows.
    """
    def __init__(self, **kwargs):
        super(LLPTransformerModel, self).__init__()
        
        cfg_parser=config_parser.config_parser()

        cfg_parser.add_default_kwarg("settings", "input_dim", 5, int)
        cfg_parser.add_default_kwarg("settings", "output_dim", 50, int)

        cfg_parser.add_default_kwarg("settings", "io_mlp_hidden_dims", "128", str)
        cfg_parser.add_default_kwarg("settings", "io_add_skip_connection", 0, int)

        ##TODO: this one can be removed probably .. should always be "single_self" .. i.e. default single MLP w/ skips
        cfg_parser.add_default_kwarg("settings", "attn_io_projection_type", "single_self", str, choices=["single_self", "single_qkv", "joint_qkv"])

        cfg_parser.add_default_kwarg("settings", "attn_do_perlayer_out_projection", 1, int, choices=[0,1])

        cfg_parser.add_default_kwarg("settings", "skip_input_projection", 0, int) # skipping the input projection step
        
        cfg_parser.add_default_kwarg("settings", "attn_computational_dim", -1, int)
       
        cfg_parser.add_default_kwarg("settings", "attn_num_layers", 2, int)
        cfg_parser.add_default_kwarg("settings", "attn_num_heads_per_layer", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_1", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_layer_norm_2", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_extra_layer_norm", 0, int)
        cfg_parser.add_default_kwarg("settings", "attn_layer_norm_first", 1, int)
        cfg_parser.add_default_kwarg("settings", "attn_use_residual_addition", 1, int)
        cfg_parser.add_default_kwarg("settings", "dtype", "float32", str, choices=["float64", "float32", "float16", "bfloat16"])
        cfg_parser.add_default_kwarg("settings", "attn_projection_type", "joint_qkv", str, choices=["single_self", "single_qkv", "joint_qkv"])
        
        cfg_parser.add_default_kwarg("settings", "attn_perform_final_mapping", 1, int, choices=[0,1])


        cfg_parser.add_default_kwarg("settings", "attn_package", "xformers", str, choices=["custom_pytorch", "official_pytorch_w_weights", "xformers", "geometric_scatter", "nested", "flash_attn"])


        cfg_parser.add_default_kwarg("settings", "attn_dropout", 0.1, float)
        cfg_parser.add_default_kwarg("settings", "attn_internal_mlp_dim", 512, int)

        cfg_parser.add_default_kwarg("settings", "attn_use_weighted_mean", 0, int)
        cfg_parser.add_default_kwarg("settings", "attn_aggregation_mode", "mean", str, choices=["mean", "mean_n_diagonal"])

        cfg_parser.add_default_kwarg("settings", "attn_operator", "none", str)


        settings_args, settings_kwargs=cfg_parser.parse_cfg(kwargs, "settings", check_passed_params_are_configured=True)

        #print("Configured attantion settings....")
        
        for k in settings_kwargs:
            #print(k, settings_kwargs[k])
            if(k!="dtype"):
                setattr(self, k, settings_kwargs[k])
        
        #print("------------------------")

        if(self.attn_operator is not None):
            if(self.attn_operator=="None" or self.attn_operator=="none"):
                self.attn_operator=None

        ## size B X NUM ITEMS X INPUT DIM
        #self.h0=nn.Parameter(torch.randn((1, 1, self.input_dim)))

        if(self.attn_computational_dim==-1):
            self.attn_computational_dim=self.output_dim

            if(self.attn_use_weighted_mean):
                self.attn_computational_dim+=1

        if(self.attn_aggregation_mode!="mean"):
            assert(self.attn_use_weighted_mean==False)
            assert(self.attn_computational_dim!=-1)
            assert(self.attn_perform_final_mapping)


        self.attn_dtype=settings_kwargs["dtype"]

        if(self.attn_dtype=="float64"):
            self.attn_dtype=torch.float64
        elif(self.attn_dtype=="float32"):
            self.attn_dtype=torch.float32
        elif(self.attn_dtype=="float16"):
            self.attn_dtype=torch.float16
        elif(self.attn_dtype=="bfloat16"):
            self.attn_dtype=torch.bfloat16

        assert(self.attn_computational_dim%self.attn_num_heads_per_layer == 0), ("Embedding / Computational dim must be divisible by number of attention heads!", self.attn_computational_dim, self.attn_num_heads_per_layer)

        if(settings_kwargs["skip_input_projection"]):
            ## we skip the input projeciton .. dimensions must match
            assert(self.attn_computational_dim==self.input_dim) 

            self.input_projector=lambda x: x
        else:

            ## TODO: dont need a qkv projector here. normal skip MLP fine
            self.input_projector=qkv_projector(input_dim=self.input_dim, 
                                               output_dim=self.attn_computational_dim, 
                                               mlp_hidden_dims=settings_kwargs["io_mlp_hidden_dims"],
                                               projection_type=settings_kwargs["attn_io_projection_type"],
                                               dtype=self.attn_dtype,
                                               add_skip_connection=settings_kwargs["io_add_skip_connection"]
                                               )
       
        
        self.extra_layer_norm=None
        if(self.attn_use_extra_layer_norm):
            self.extra_layer_norm=nn.LayerNorm(self.attention_input_dim, dtype=self.attn_dtype)
        
        # we use relative positional encodings in some layers.. have to define how
        base_encoder_args=[self.attn_computational_dim, self.attn_num_heads_per_layer]
        base_encoder_kwargs=dict()
        base_encoder_kwargs["dim_feedforward"]=self.attn_internal_mlp_dim
        base_encoder_kwargs["dropout"]=self.attn_dropout
        base_encoder_kwargs["norm_first"]=self.attn_layer_norm_first
        base_encoder_kwargs["use_layer_norm_1"]=self.attn_use_layer_norm_1
        base_encoder_kwargs["use_layer_norm_2"]=self.attn_use_layer_norm_2
        base_encoder_kwargs["use_residual_addition"]=self.attn_use_residual_addition
        base_encoder_kwargs["attn_package"]=settings_kwargs["attn_package"]
        base_encoder_kwargs["projection_hidden_dims"]=""
        base_encoder_kwargs["projection_type"]=settings_kwargs["attn_projection_type"]
        base_encoder_kwargs["dtype"]=self.attn_dtype
        base_encoder_kwargs["xformers_operator"]=self.attn_operator
        base_encoder_kwargs["do_perlayer_out_projection"]=self.attn_do_perlayer_out_projection

        self.transformer_encoder = CustomTransformerEncoder(base_encoder_args,
                                                            base_encoder_kwargs, 
                                                            self.attn_num_layers,
                                                            self.input_dim, 
                                                            extra_layer_norm=self.extra_layer_norm)
        

        if(settings_kwargs["attn_perform_final_mapping"]==1):
            if(settings_kwargs["attn_computational_dim"]!=-1):
                aggregation_dim=self.attn_computational_dim

                if(self.attn_aggregation_mode=="mean_n_diagonal"):
                    aggregation_dim*=2
                elif(self.attn_use_weighted_mean):
                    aggregation_dim-=1

                self.attention_to_output_mlp=SkipMLP(aggregation_dim, settings_kwargs["io_mlp_hidden_dims"], self.output_dim, dtype=self.attn_dtype, add_skip_connection=settings_kwargs["io_add_skip_connection"])

            else:

                if(self.attn_use_weighted_mean):
                    aggregation_dim=self.attn_computational_dim-1
                    ## a linear mapping to the output space...
                    self.attention_to_output_mlp=torch.nn.Linear(aggregation_dim, self.output_dim, dtype=self.attn_dtype)
                else:
                    self.attention_to_output_mlp= lambda x: x
        else:
            ## not output mapping mlp
            self.attention_to_output_mlp=None

    def _create_xformers_datarep_and_mask(self, datavecs, datalens):
        """ If variable length sequences are used,
        datalens must be 1D tensor of [S_0, S_1, ..., S_B],
        and datavecs should be a list of length B with items tensor([1 x S_i x D])
        """

        if(type(datavecs)==torch.Tensor):
            ## TODO: change this to directly use tensor
            maxlen=max(datalens)

            ## all datalens must be the same
            assert((maxlen==datalens).sum()==len(datalens))

            data=[vec.unsqueeze(0) for vec in datavecs]
        
        else:
            assert(type(datavecs)==list), "Datavecs must be a list of tensors"
            data=datavecs

       
        attn_bias, x = fmha.BlockDiagonalMask.from_tensor_list(data)

        return x, attn_bias

    def _create_datarep_and_mask(self, datavecs, datalens, nhead, attn_package=None, relative_encoding_indices=[]):

        """
        Returns:
        input_vecs: Depending on attn package, can be list of vecs or a padded tensor.
        """
            
        assert(type(datalens)==torch.Tensor), "Require datalens as tensor"

        data, mask = self._create_xformers_datarep_and_mask(datavecs, datalens)

        return data, mask, mask
    
    def _final_summation(self, result_matrix, mask, datalens, attn_package=None, add_weights=False, perform_final_aggregation=True, aggregation_mode="mean"):

        used_attn_package=self.attn_package
        if(attn_package is not None):
            used_attn_package=attn_package

        out=mask.split(result_matrix)

        if(perform_final_aggregation==False):
            # return list of
            return out

        if(add_weights):

            #weights=[torch.nn.functional.softmax(i[:,:,-1:], dim=1) for i in out]
            
            attn_result=torch.cat([ (torch.nn.functional.softmax(i[:,:,-1:], dim=1)*i[:,:,:-1] ).sum(dim=1) for i in out])

        else:

            if(aggregation_mode=="mean"):
                
                attn_result=torch.cat([i.mean(dim=1) for i in out])
            else:
                means=[i.mean(dim=1) for i in out]
                
                variances=[ ((means[ind][:,None,:]-out[ind])**2).mean(dim=1) for ind in range(len(out))]
                variances=torch.cat(variances)
                means=torch.cat(means)

                attn_result=torch.cat([means, variances], dim=-1)

        return attn_result

    def forward(self, datavecs, datalens, perform_final_aggregation=True, perform_final_mapping=True):
        #package-dependent processing

        # data and datalen preparation dependent on package
        padded_tensor_first, padding_mask_first, padding_mask_last=self._create_datarep_and_mask(datavecs, datalens, self.attn_num_heads_per_layer)
        
        # @TODO: comment this out
        # print("types forward ...---------_> ")
        # print(type(padded_tensor_first))
        # print(type(padding_mask_first))
        # print(type(padding_mask_last))
        # print("padding mask first")
        # print(padding_mask_first)
        # print("attn package", self.attn_package)
        # print("---------------------------")

        # embedding: input projection D -> C (computational dim)
        computational_input=self.input_projector(padded_tensor_first)
        
        # transformer layers
        result=self.transformer_encoder(computational_input, src_key_padding_mask=padding_mask_first)

        # aggregation
        
        result=self._final_summation(result, padding_mask_last,datalens, attn_package=self.attn_package, add_weights=self.attn_use_weighted_mean, perform_final_aggregation=perform_final_aggregation, aggregation_mode=self.attn_aggregation_mode)

        if(perform_final_mapping==False):
            return result

        assert(self.attention_to_output_mlp is not None), "Choose to perform final mapping, but attention_to_output_mlp is None... have to define Encoder with that flag on!"
        ret_val = self.attention_to_output_mlp(result)
            
        return ret_val

class CustomTransformerEncoder(Module):
    r"""TransformerEncoder is a stack of N encoder layers. Users can build the
    BERT(https://arxiv.org/abs/1810.04805) model with corresponding parameters.

    @TODO: fix documentation, it's outdated
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
        enable_nested_tensor: if True, input will automatically convert to nested tensor
            (and convert back on output). This will improve the overall performance of
            TransformerEncoder when padding rate is high. Default: ``True`` (enabled).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, 
                 encoder_layer_args, 
                 encoder_layer_kwargs,
                 num_layers, 
                 absolute_input_dim,
                 extra_layer_norm=None):
                 
                
        super().__init__()
            

        list_of_layers=[]
        self.indices_for_original_input_feeding=[]

        self.input_formats=[]

        
        for ind in range(num_layers):
            these_args=copy.deepcopy(encoder_layer_args)
            these_kwargs=copy.deepcopy(encoder_layer_kwargs)
            
            if("pytorch" in these_kwargs["attn_package"]):
                self.input_formats.append("p")
            else:
                self.input_formats.append("x")

            list_of_layers.append(CustomTransformerEncoderLayer(*these_args, **these_kwargs))

       
        self.layers=ModuleList(list_of_layers)

        self.num_layers = num_layers
        self.norm = extra_layer_norm
        self.absolute_input_dim=absolute_input_dim

    def forward(
            self,
            src: Tensor,
            mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: Optional[bool] = None) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask as ``mask``.
                Default: ``None``; try to detect a causal mask.
                Warning:
                ``is_causal`` provides a hint that ``mask`` is the
                causal mask. Providing incorrect hints can result in
                incorrect execution, including forward and backward
                compatibility.

        Shape:
            see the docs in Transformer class.
        """
        
        #print("src key paddingmask in overall encoder ")
        #print(src_key_padding_mask)
        global_relative_input=None

        if(len(self.indices_for_original_input_feeding)>0):
            rel_position_tensor=None
            ## make relative position base input

            element_list=[src.shape[1] for i in range(src.shape[0])]
            if(src_key_padding_mask is not None):
                element_list=[int(sum(mitem==0).cpu().detach()) for mitem in src_key_padding_mask]

            flattened_distances, used_elements=_get_tensor_of_diffs_flattened(src, element_list=element_list)
            
            print(flattened_distances.shape)
            print(used_elements)
            sys.exit(-1)
            

        output = src
        used_mask=src_key_padding_mask

        #print("input formats ", self.input_formats)
        for layer_ind, mod in enumerate(self.layers):
            #print("-------> Transformer layer ", layer_ind)

            if(layer_ind>0):
                if(self.input_formats[layer_ind]=="p"):
                    if(self.input_formats[layer_ind-1]=="x"):
                        # change x->p
                        output, used_mask=_transform_xformer_to_pytorch_rep(output, used_mask)

                elif(self.input_formats[layer_ind]=="x"):
                    if(self.input_formats[layer_ind-1]=="p"):
                        # change p->x
                        output, used_mask=_transform_pytorch_to_xformer_rep(output, used_mask)

            output = mod(output, src_mask=None, src_key_padding_mask=used_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class CustomTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)

    Fast path:
        forward() will use a special optimized implementation if all of the following
        conditions are met:

        - Either autograd is disabled (using ``torch.inference_mode`` or ``torch.no_grad``) or no tensor
          argument ``requires_grad``
        - training is disabled (using ``.eval()``)
        - batch_first is ``True`` and the input is batched (i.e., ``src.dim() == 3``)
        - norm_first is ``False`` (this restriction may be loosened in the future)
        - activation is one of: ``"relu"``, ``"gelu"``, ``torch.functional.relu``, or ``torch.functional.gelu``
        - at most one of ``src_mask`` and ``src_key_padding_mask`` is passed
        - if src is a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_, neither ``src_mask``
          nor ``src_key_padding_mask`` is passed
        - the two ``LayerNorm`` instances have a consistent ``eps`` value (this will naturally be the case
          unless the caller has manually modified one without modifying the other)

        If the optimized implementation is in use, a
        `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ can be
        passed for ``src`` to represent padding more efficiently than using a padding
        mask. In this case, a `NestedTensor <https://pytorch.org/docs/stable/nested.html>`_ will be
        returned, and an additional speedup proportional to the fraction of the input that
        is padding can be expected.
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, 
                 embed_dim: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.0,
                 activation: Callable[[Tensor], Tensor] = F.relu,
                 layer_norm_eps: float = 1e-5, 
                 norm_first: bool = False,
                 device=None, 
                 dtype=None,
                 use_layer_norm_1=True, 
                 use_layer_norm_2=True, 
                 use_residual_addition=True,
                 attn_package="xformers", # pytorch / xformers / flash
                 projection_hidden_dims="",
                 projection_type="joint_qkv",
                 xformers_operator=None,
                 do_perlayer_out_projection=1) -> None:

        factory_kwargs = {'device': device, 'dtype': dtype}

        super(CustomTransformerEncoderLayer, self).__init__()

        assert(projection_type=="joint_qkv")
        
        self.in_projector=qkv_projector(input_dim=embed_dim, 
                                        mlp_hidden_dims=projection_hidden_dims, 
                                        output_dim=embed_dim,
                                        projection_type=projection_type)

        self.nhead=nhead
        self.embed_dim=embed_dim
        self.dim_per_head=self.embed_dim//self.nhead
        assert(self.embed_dim % self.nhead ==0 ), "Embedding dim must be divisble by nhead!"

        ## required to work with pytorch 2.xx .. just a hack
        self_attn=dict()
        self_attn["batch_first"]=True
        self.self_attn=Namespace(**self_attn)

        ## TODO: make this optional?
        self.out_projector=None
        self.do_perlayer_out_projection=do_perlayer_out_projection
        if(self.do_perlayer_out_projection):
            self.out_projector=qkv_projector(input_dim=embed_dim, 
                                        mlp_hidden_dims=projection_hidden_dims, 
                                        output_dim=embed_dim,
                                        projection_type="single_self")
        else:
            assert(use_residual_addition!=1), "Residual addition of 1 (default transformer) requires out projection.. otherwise it is too restricting"

        # Implementation of Feedforward model
        self.linear1 = Linear(embed_dim, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, embed_dim, **factory_kwargs)

        self.norm_first = norm_first


        if(use_layer_norm_1):
            self.norm1 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm1 = lambda x: x

        if(use_layer_norm_2):
            self.norm2 = LayerNorm(embed_dim, eps=layer_norm_eps, **factory_kwargs)
        else:
            self.norm2 = lambda x: x

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # no string support for activation function
        assert(isinstance(activation, str) is False)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

        self.use_residual_addition=use_residual_addition

        self.projection_type=projection_type
        self.attn_package=attn_package

        self.dropout_p=dropout
        
    def __setstate__(self, state):
        super(CustomTransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, is_causal=False,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        ## replace default encoder layer with custom blocks
        ## right now we want no causal encoding
        assert(is_causal==False)
        assert(src_mask is None)

        x = src

        ## only use residual addition on SA block?
        if(self.use_residual_addition==2):
            if self.norm_first:
                
                sa_result = self._sa_block(self.norm1(x), None, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(sa_result))
              
            else:
                sa_result = self.norm1(x + self._sa_block(x, None, src_key_padding_mask))
                
                x = self.norm2(x + self._ff_block(sa_result))

        ## DEFAULT TRANSFORMER -- both SA and FF blocks use residual addition
        elif(self.use_residual_addition==1):
            if self.norm_first:
               
                x = x + self._sa_block(self.norm1(x), None, src_key_padding_mask)
                x = x + self._ff_block(self.norm2(x))
              
            else:
                x = self.norm1(x + self._sa_block(x, None, src_key_padding_mask))
                
                x = self.norm2(x + self._ff_block(x))
        elif(self.use_residual_addition==0):
            if self.norm_first:
                x = self._sa_block(self.norm1(x), None, src_key_padding_mask)
                x = self._ff_block(self.norm2(x))
            else:
                x = self.norm1(self._sa_block(x, None, src_key_padding_mask))
                x = self.norm2(self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor]) -> Tensor:

        ## in projection

        ## Q, k, V shape: B X num_length X Num_heads*dim_per_head
   
        batch_size=x.size(0)
        total_embed_dim=self.nhead*self.dim_per_head

        if(self.attn_package=="xformers"):

            q = self.in_projector(x)

            if(self.projection_type=="single_self"):
                q=joint
                k=joint
                v=joint
            else:
                q,k,v = q # q is tuple of q,k,v

            # split up last (attn dim) into nhead sectors with (attn_dim/nhead) subdimensionality
            q=q.reshape(q.shape[0], q.shape[1], self.nhead, -1)
            k=k.reshape(k.shape[0], k.shape[1], self.nhead, -1)
            v=v.reshape(v.shape[0], v.shape[1], self.nhead, -1)

            #### test
            #res=mh_attention_xformer_helpers.memory_efficient_with_lse(q,k,v,attn_bias=key_padding_mask, op=self.xformers_operator)
            ####

            ## probs*v
            out=fmha.memory_efficient_attention(q, k, v, attn_bias=key_padding_mask)

            out=out.reshape(q.shape[0], q.shape[1], -1)

            ## out projection not really needed without dropout
            if(self.do_perlayer_out_projection==1):
                out=self.out_projector(out)

            return self.dropout1(out)
            ## switch to nhead

        else:
            raise Exception("Unknown package ", self.attn_package)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class qkv_projector(nn.Module):
    def __init__(self, input_dim=50, output_dim=50, mlp_hidden_dims="",
                 projection_type="single_self", add_skip_connection=0, dtype=torch.float32):
        """
        Parameters:

        input_dim (int): input dimension
        output_dim (int): output dimension
        mlp_hidden_dims (str): Hidden dim structure, i.e. "128-256" or "" for just linear mapping
        projection_type (str): Project similar into same space for q,k,v ("single_self")
                               Project differently for q,k,v with one mapping for each ("single_qkv")
                               Project with a joint mapping for q,k,v ("joint_qkv") into 3*embedding_dim space
        """

        super(qkv_projector, self).__init__()

        self.projection_type = projection_type

        if self.projection_type == "single_self":
            self.projector = SkipMLP(input_dim, mlp_hidden_dims, output_dim,
                                     add_skip_connection=add_skip_connection, dtype=dtype)

        elif self.projection_type == "single_qkv":
            self.q_projector = SkipMLP(input_dim, mlp_hidden_dims, output_dim,
                                       add_skip_connection=add_skip_connection, dtype=dtype)
            self.k_projector = SkipMLP(input_dim, mlp_hidden_dims, output_dim,
                                       add_skip_connection=add_skip_connection, dtype=dtype)
            self.v_projector = SkipMLP(input_dim, mlp_hidden_dims, output_dim,
                                       add_skip_connection=add_skip_connection, dtype=dtype)

        elif self.projection_type == "joint_qkv":
            self.joint_projector = SkipMLP(input_dim, mlp_hidden_dims, 3 * output_dim,
                                           add_skip_connection=add_skip_connection, dtype=dtype)
        else:
            raise Exception("Hmm this should not happen, unknown projection type ... ", self.projection_type)

    def forward(self, x, split_qkv=True):

        if(self.projection_type=="single_self"):

            q=self.projector(x)

            return q

        elif(self.projection_type=="single_qkv"):

            q=self.q_projector(x)

            k=self.k_projector(x)

            v=self.v_projector(x)

            return q,k,v

        elif(self.projection_type=="joint_qkv"):

            joint=self.joint_projector(x)

            # split qkv for most applications (default)
            if(split_qkv):

                embed_dim=x.size(-1)

                q=joint[..., :embed_dim]
                k=joint[..., embed_dim:2*embed_dim]
                v=joint[..., 2*embed_dim:3*embed_dim]
                
                return q,k,v
            else:
                # some implementations (flash_attn) require merged qkv for fastest mode
                return joint
