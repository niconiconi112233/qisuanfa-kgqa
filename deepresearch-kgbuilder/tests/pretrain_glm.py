import argparse
import os
import torch
from torch import Tensor
from functools import partial
from typing import Any, Iterable, Union

import torch.distributed
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.training import get_args
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
import megatron.legacy.model
from megatron.core.models.gpt import GPTModel
from megatron.core.parallel_state import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_src_rank,
    get_data_parallel_group,
    get_data_parallel_world_size
)
from megatron.training import pretrain
from megatron.training.global_vars import get_sample_store, set_dataset_state, get_dataset_id_mapping
from megatron.core.transformer.spec_utils import import_module
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.training.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from glm import build_train_valid_test_datasets, build_packed_seq_params


class DummyClass:
    ...


def model_provider(pre_process=True, post_process=True):
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()

    print_rank_0('building GLM model ...')
    config = core_transformer_config_from_args(get_args())

    if args.post_self_attn_layernorm or args.post_mlp_layernorm:
        assert not args.add_bias_linear, "Adding bias to the linear layers is not supported with post-norm"

    assert args.use_mcore_models, "GLM only support megatron-core"
    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm,
            post_self_attn_layernorm=args.post_self_attn_layernorm,
            post_mlp_layernorm=args.post_mlp_layernorm
        )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
    )

    return model

def is_dataset_built_on_rank():
    if mpu.get_virtual_pipeline_model_parallel_world_size() is None:
        # If no VP, then all device should own a dataloader
        return True
    
    # If VP, then only the first VP stage should own a dataloader
    return mpu.get_virtual_pipeline_model_parallel_rank() == 0

SOURCE_LOSS_REDUCE_BUFFER = None

def _get_source_loss_reduce_buffer():
    global SOURCE_LOSS_REDUCE_BUFFER
    if SOURCE_LOSS_REDUCE_BUFFER is None:
        SOURCE_LOSS_REDUCE_BUFFER = torch.zeros(
            (2, len(get_dataset_id_mapping())),
            dtype=torch.float32,
            device=torch.cuda.current_device(),
        )  # [count, cumulative loss]
    else:
        SOURCE_LOSS_REDUCE_BUFFER.zero_()
    return SOURCE_LOSS_REDUCE_BUFFER

def get_batch(data_iterator: Iterable[tuple[int, dict[str, Any]]]):
    """Generate a batch"""
    args = get_args()
    timers = get_timers()

    assert args.eval_interval == 0, 'Evaluation is not supprted'
    assert args.context_parallel_size == 1, 'CP is currently not supported for glm dataloader'

    timers('batch-generator-data-iterator', log_level=2).start()
    data_identifier, data = next(data_iterator)
    timers('batch-generator-data-iterator').stop()

    sample_store = get_sample_store()

    if is_dataset_built_on_rank():
        assert data is not None
        assert data_identifier not in sample_store
        # Update dataset state
        data, dataset_state = data
        set_dataset_state(dataset_state)

        # To device
        timers('batch-generator-to-device', log_level=2).start()
        if args.pack_sequence:
            # for packed seqs we always have b=1
            # keep original batch shape to restore loss per source
            data['tokens'] = data['tokens'].view(1, -1)
            data['targets'] = data['targets'].view(1, -1)

            if args.micro_batch_size == 1:
                divisions_and_seqlen = data['divisions'].view(-1).to(torch.int32)
                divisions, max_seqlen = divisions_and_seqlen[:-1], divisions_and_seqlen[-1].item()
                divisions = divisions.cuda(non_blocking=True)
                packed_seq_params = PackedSeqParams(
                    qkv_format='thd',
                    cu_seqlens_q=divisions,
                    cu_seqlens_kv=divisions,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_kv=max_seqlen,
                )
            else:
                packed_seq_params = build_packed_seq_params(data['divisions'])
        else:
            packed_seq_params = None

        tokens = data['tokens'].long().cuda(non_blocking=True)
        labels = data['targets'].long().cuda(non_blocking=True)
        loss_masks = data['loss_masks'].long().cuda(non_blocking=True)
        sources = data['source'].long().cuda(non_blocking=True)  # assuming single source per sample
        timers('batch-generator-to-device').stop()

        batch = {
            'tokens': tokens, 
            'labels': labels, 
            'loss_mask': loss_masks, 
            'packed_seq_params': packed_seq_params,
            'sources': sources
        }

        vp_world_size = mpu.get_virtual_pipeline_model_parallel_world_size()
        life_counter = vp_world_size - 1 if vp_world_size else 0
    else:
        assert data is None
        assert data_identifier in sample_store
        life_counter, batch = sample_store.pop(data_identifier)
        life_counter -= 1

    if life_counter != 0:
        sample_store[data_identifier] = life_counter, batch

    # sanity check for deletion
    num_micro_batch = args.global_batch_size // mpu.get_data_parallel_world_size() // args.micro_batch_size
    assert len(sample_store) <= num_micro_batch, f"Sample store size {len(sample_store)} is larger than num_micro_batch {num_micro_batch}"

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()

def loss_func(loss_mask: Tensor, sources: Tensor, output_tensor: Tensor):
    """Loss function.

    Args:
        loss_mask (Tensor): Used to mask out some portions of the loss
        sources (Tensor): Used to mask out some portions of the loss
        output_tensor (Tensor): The tensor with the losses
    """    
    args = get_args()

    losses = output_tensor.float()
    original_shape = loss_mask.shape
    is_valid_pack = (loss_mask.sum() > 0).item()
    original_loss_mask = loss_mask
    loss_mask = loss_mask.view(-1).float()
    flatten_losses = losses.view(-1)
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(flatten_losses * loss_mask).view(1), loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        if is_valid_pack:
            loss = torch.sum(flatten_losses * loss_mask) / loss_mask.sum()
        else:
            print(f"Rank {torch.distributed.get_rank()}: Dummy loss calculation for empty pack")
            loss = losses.sum() * 0

    # Check individual rank losses are not NaN prior to DP all-reduce.
    if args.check_for_nan_in_loss_and_grad:
        global_rank = torch.distributed.get_rank()
        assert not loss.isnan(), (
            f'Rank {global_rank}: found NaN in local forward loss calculation. '
            f'Device: {torch.cuda.current_device()}, node: {os.uname()[1]}'
        )

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    # Reduce loss and source for each sample
    with torch.no_grad():
        source_loss_reduce_buffer = _get_source_loss_reduce_buffer()

        assert sources.shape == original_shape, (
            f"sources shape {sources.shape} != loss_mask original shape {original_shape}"
        )
        flattened_sources = sources.view(-1)
        assert args.context_parallel_size == 1, (
            "Context parallelism is not considered for multi-source per sample loss reduction"
        )
        assert ((loss_mask == 1) == (flattened_sources != 0)).all(), (
            f"Loss mask and sources mismatch, a source is not dummy if and only if the loss is dummy: "
            f"{loss_mask.tolist()} vs {flattened_sources.tolist()}"
        )
        source_loss_reduce_buffer[0].scatter_add_(0, flattened_sources, torch.ones_like(flatten_losses))
        source_loss_reduce_buffer[1].scatter_add_(0, flattened_sources, flatten_losses)

        torch.distributed.all_reduce(source_loss_reduce_buffer, group=get_data_parallel_group())
        loss_per_source = {'lm loss': averaged_loss[0], 'source_loss': source_loss_reduce_buffer.cpu()}
        del source_loss_reduce_buffer

    return loss * args.context_parallel_size, loss_per_source


def forward_step(data_iterator, model: GPTModel):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    tokens, labels, loss_mask, packed_seq_params, sources = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, None, None,
                          labels=labels, packed_seq_params=packed_seq_params)

    return output_tensor, partial(loss_func, loss_mask, sources)


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    if not is_dataset_built_on_rank():
        return None, None, None

    print_rank_0("> building train, validation, and test datasets for GLM ...")

    assert args.num_workers == 1, "GLM's dataloader requires num_worker = 1"
    assert args.pack_sequence or not args.fix_gmask_for_packing, "Impossible to fix gmask without packing"
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        tokenizer=get_tokenizer(),
        text_data_path=args.train_data_path,
        train_valid_test_num_samples=train_val_test_num_samples,
        multitask_data_paths=args.multitask_data_path,
        multitask_ratios=args.multitask_ratio,
        text_ratio=args.text_ratio,
        seq_length=args.seq_length,
        seed=args.seed,
        data_distribution_warmup=args.data_distribution_warmup,
        data_distribution_warmup_samples=args.data_distribution_warmup_samples,
        fix_gmask_for_packing=args.fix_gmask_for_packing,
        gpt=args.gpt,
        no_prefix_tokens=args.no_prefix_tokens,
        no_eos_token=args.no_eos_token,
        build_packed_seq_params=args.micro_batch_size == 1,
        same_source_in_packing_multitask_data=args.same_source_in_packing_multitask_data,
        no_packing_multitask_data=args.no_packing_multitask_data
    )

    print_rank_0("> finished creating GLM datasets ...")

    return train_ds, valid_ds, test_ds


def extra_args_provider(parser):
    group = parser.add_argument_group(title='GLM')

    # Model
    group.add_argument('--gpt', action='store_true')
    group.add_argument('--no-prefix-tokens', action='store_true')
    group.add_argument('--no-eos-token', action='store_true')
    group.add_argument('--no-interleaved-qkv', dest='interleaved_qkv', action='store_false')
    group.add_argument('--post-self-attn-layernorm', action='store_true')
    group.add_argument('--post-mlp-layernorm', action='store_true')
    group.add_argument('--rotary-base', type=int, default=10000)
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5)

    # MoE
    group.add_argument('--moe-num-shared-experts', type=int, default=0)
    group.add_argument('--moe-frequency', type=int, default=1)
    group.add_argument('--moe-num-first-dense-layers', type=int, default=0)

    # Data
    group.add_argument('--multitask-data-path', type=str, default=[], action='append', nargs='+')
    group.add_argument("--multitask-ratio", type=float, default=[], action='append', help="Ratio of qa training data")
    group.add_argument("--text-ratio", type=float, default=0, help="Ratio of text training data")
    group.add_argument('--data-distribution-warmup', action='store_true')
    group.add_argument('--data-distribution-warmup-samples', type=int, default=0)
    group.add_argument('--pack-sequence', action=argparse.BooleanOptionalAction, default=True)
    group.add_argument('--fix-gmask-for-packing', action='store_true')
    group.add_argument('--return-sample-identifier', default=True, type=bool) # TODO@lambda: pretttier
    group.add_argument('--same-source-in-packing-multitask-data',
                       action='store_true',
                       help='If True, the same source will be packed together in multitask data, '
                       'which is helpful for alltools training.')
    group.add_argument('--no-packing-multitask-data',
                       action='store_true',
                       help='Do not pack multitask data.')

    # Checkpoint
    group.add_argument('--load-optim-from-release', action='store_true')
    group.add_argument('--save-async-fast-checkpoint', default=False, action='store_true')
    group.add_argument('--force-warmup-steps', default=0, type=int)
    group.add_argument('--min-force-warmup-lr', default=1e-8, type=float)
    group.add_argument('--keep-optim-interval', default=None, type=int,
                       help='The interval to keep optimizer states, others will be deleted.')
    group.add_argument('--keep-optim-recent', default=None, type=int,
                       help='The number of recent optimizer states states to keep. '
                            'Take precedence over --keep-optim-interval.')


    # Reduce layers
    group.add_argument('--reduce-layers-for-lm-head', default=0, type=int)

    group.add_argument('--gqa-repeat-num', type=int, default=1)

    # Disable Evaluation
    group.set_defaults(eval_interval=0)
    group.set_defaults(eval_iters=0)

    return parser


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             extra_args_provider=extra_args_provider)
