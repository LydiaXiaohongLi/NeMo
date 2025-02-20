# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utilities for generating text."""

import pickle
from collections.abc import Iterable
from functools import partial
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from nemo.collections.common.tokenizers.tabular_tokenizer import TabularTokenizer
from nemo.collections.nlp.modules.common.megatron.utils import get_ltor_masks_and_position_ids
from nemo.collections.nlp.modules.common.text_generation_strategy import model_inference_strategy_dispatcher
from nemo.collections.nlp.modules.common.transformer.text_generation import LengthParam, OutputType, SamplingParam
from nemo.utils import AppState

try:
    from apex.transformer.pipeline_parallel.utils import _reconfigure_microbatch_calculator

    HAVE_APEX = True

except (ImportError, ModuleNotFoundError):

    HAVE_APEX = False

try:
    from megatron.core import parallel_state, tensor_parallel

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):

    HAVE_MEGATRON_CORE = False

__all__ = [
    "get_default_sampling_params",
    "get_default_length_params",
    "megatron_gpt_generate",
    "get_computeprob_response",
    "generate",
    "sample_token_greedy",
    "sample_token_topk",
]


def get_default_sampling_params():
    # default do greedy sampling
    sampling_params: SamplingParam = {
        "use_greedy": True,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "add_BOS": True,
        "all_probs": False,
        "compute_logprob": False,
    }

    return sampling_params


def get_default_length_params():
    # default do greedy sampling
    length_params: LengthParam = {"min_length": 0, "max_length": 30}

    return length_params


def megatron_gpt_generate(model, inputs, tokenizer, length_params, sampling_params, **strategy_args):
    # reproduce the old compute_prob method
    # a very special case
    if sampling_params['compute_logprob']:
        # need to overwrite some configuration, make it immutable
        sampling_params = sampling_params.copy()
        length_params = length_params.copy()
        length_params['max_length'] = 1
        sampling_params['all_probs'] = True
        sampling_params["add_BOS"] = False
        sampling_params['greedy'] = True
        response = generate(
            model,
            inputs=inputs,
            tokens_to_generate=length_params['max_length'],
            all_probs=sampling_params['all_probs'],
            compute_logprob=sampling_params['compute_logprob'],
            temperature=sampling_params['temperature'],
            add_BOS=sampling_params['add_BOS'],
            top_k=sampling_params['top_k'],
            top_p=sampling_params['top_p'],
            greedy=sampling_params['use_greedy'],
            repetition_penalty=sampling_params['repetition_penalty'],
            min_tokens_to_generate=length_params['min_length'],
            **strategy_args,
        )
        compute_prob_response = get_computeprob_response(tokenizer, response, inputs)
        return compute_prob_response

    if isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], (str, torch.Tensor)):
            output = generate(
                model,
                inputs=inputs,
                tokens_to_generate=length_params['max_length'],
                all_probs=sampling_params['all_probs'],
                compute_logprob=sampling_params['compute_logprob'],
                temperature=sampling_params['temperature'],
                add_BOS=sampling_params['add_BOS'],
                top_k=sampling_params['top_k'],
                top_p=sampling_params['top_p'],
                greedy=sampling_params['use_greedy'],
                repetition_penalty=sampling_params['repetition_penalty'],
                min_tokens_to_generate=length_params['min_length'],
                **strategy_args,
            )
            return output
        elif isinstance(inputs[0], dict):
            raise NotImplementedError("json object not implemented")
        else:
            raise NotImplementedError("unknown type is not implemented")
    else:
        raise NotImplementedError("unknown type is not implemented")


def get_computeprob_response(tokenizer, response, inputs):
    if parallel_state.is_pipeline_first_stage() or parallel_state.is_pipeline_last_stage():
        # we only have a response on the first and last pipeline stages
        compute_prob_response = {}
        new_token_ids = []
        new_tokens = []
        new_texts = []
        log_probs = []
        full_logprobs = []
        offsets = []
        for batch_id in range(len(response['tokens'])):
            if isinstance(inputs, (list, tuple)):
                if isinstance(inputs[0], str):
                    new_token_id = tokenizer.text_to_ids(inputs[batch_id])
                    new_text = inputs[batch_id]
                    token_len = len(new_token_id)
                elif isinstance(inputs[0], torch.Tensor):
                    token_len = int(inputs[1][batch_id].item())
                    new_token_id = inputs[0][batch_id][:token_len].tolist()
                    new_text = tokenizer.ids_to_text(new_token_id)
            new_token_ids.append(new_token_id)
            new_tokens.append(response['tokens'][batch_id][:token_len])
            new_texts.append(new_text)
            log_probs.append(response['logprob'][batch_id][:token_len])
            full_logprobs.append(response['full_logprob'][batch_id][:token_len])
            offsets.append(response['offsets'][batch_id][:-1])
        compute_prob_response['sentences'] = new_texts
        compute_prob_response['tokens'] = new_tokens
        compute_prob_response['token_ids'] = new_token_ids
        compute_prob_response['logprob'] = log_probs
        compute_prob_response['full_logprob'] = full_logprobs
        compute_prob_response['offsets'] = offsets
        return compute_prob_response
    else:
        # intermediate stages
        return None


def get_batch(model, tokenizer, context_tokens):
    """Generate batch from context tokens."""
    # Move to GPU.
    tokens = context_tokens.contiguous().cuda()
    # Get the attention mask and postition ids.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eos_id,
        model.cfg.get('reset_position_ids', False),
        model.cfg.get('reset_attention_mask', False),
        model.cfg.get('eod_mask_loss', False),
    )

    return tokens, attention_mask, position_ids


def tab_logits(logits, min_id, max_id, filter_value=-float('Inf')):
    logits[:, :min_id] = filter_value
    logits[:, max_id:] = filter_value
    return logits


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf'), started=None):
    """
       This function has been mostly taken from huggingface conversational
         ai code at
         https://medium.com/huggingface/how-to-build-a-state-of-the-art-
              conversational-ai-with-transfer-learning-2d818ac26313 

        @param logits: logits tensor
        @param top_k: keep only top k tokens with highest probability
        @param top_p: keep the top tokens with cumulative probability
        @filter_value: value to set filtered tokens to
        @started: a tensor of bools indicating whether the text generation starts for the batch
        returns the filtered logits
    """
    if top_k > 0:
        # Remove all tokens with a probability less than the
        # last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        if started is not None:
            for i in np.arange(indices_to_remove.size(0))[started.cpu().numpy()]:
                logits[i, indices_to_remove[i]] = filter_value
        else:
            logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Cconvert to 1D
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token
        # above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        if started is not None:
            for i in np.arange(sorted_indices.size(0))[started.cpu().numpy()]:
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = filter_value
        else:
            for i in range(sorted_indices.size(0)):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                logits[i, indices_to_remove] = filter_value

    return logits


def repetition_penalty(logits, repetition_penalty, used_tokens):
    """ Implement the repetition penalty, check paper 
    https://arxiv.org/pdf/1909.05858.pdf
    """
    if used_tokens is not None and repetition_penalty != 1.0:
        logits_update = torch.gather(logits, 1, used_tokens)
        logits = torch.scatter(logits, 1, used_tokens, logits_update / repetition_penalty)
    return logits


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to the first local rank
    in the model parallel group."""
    world_size = torch.distributed.get_world_size()
    all_ranks = np.arange(world_size)
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    # [pipeline dim, data parallel, tensor dim]
    all_ranks = all_ranks.reshape(pp_size, -1, tp_size)
    dp_rank = parallel_state.get_data_parallel_rank()
    return all_ranks[:, dp_rank, :].min()


def send_generate_info(
    context_tokens_tensor,
    context_length_tensor,
    tokens_to_generate,
    all_probs,
    compute_logprob,
    temperature,
    top_k,
    top_p,
    greedy,
    repetition_penalty,
    min_tokens_to_generate,
    end_strings,
):
    """
    Needs to be synced up with receive_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = get_model_parallel_src_rank()
    # Send the sizes of the tensors
    input_info = [
        context_tokens_tensor.size(0),  # batch_size
        context_tokens_tensor.size(1),  # seq_len
        tokens_to_generate,
        all_probs,
        compute_logprob,  # whether to compute log probabilities matrix
        temperature,
        top_k,
        top_p,
        greedy,
        repetition_penalty,
        min_tokens_to_generate,
    ]
    input_info_tensor = torch.cuda.FloatTensor(input_info)
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)

    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    # send end strings
    string_tensor = torch.as_tensor(
        np.frombuffer(pickle.dumps(end_strings), dtype=np.int8), device=torch.cuda.current_device()
    )
    size = torch.as_tensor([string_tensor.size(0)], device=torch.cuda.current_device(), dtype=torch.int64)
    torch.distributed.broadcast(size, src, model_parallel_group)
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)


def receive_generate_info():
    """
    Needs to be synced up with send_generate_info
    """
    model_parallel_group = parallel_state.get_model_parallel_group()
    src = get_model_parallel_src_rank()
    input_info_tensor = torch.empty(11, dtype=torch.float32, device=torch.cuda.current_device())
    torch.distributed.broadcast(input_info_tensor, src, model_parallel_group)
    batch_size = int(input_info_tensor[0].item())
    seq_len = int(input_info_tensor[1].item())
    tokens_to_generate = int(input_info_tensor[2].item())
    all_probs = bool(input_info_tensor[3].item())
    compute_logprob = bool(input_info_tensor[4].item())  # whether to compute log probabilities matrix
    temperature = float(input_info_tensor[5].item())
    top_k = int(input_info_tensor[6].item())
    top_p = float(input_info_tensor[7].item())
    greedy = bool(input_info_tensor[8].item())
    repetition_penalty = float(input_info_tensor[9].item())
    min_tokens_to_generate = int(input_info_tensor[10].item())

    context_length_tensor = torch.empty(batch_size, dtype=torch.int64, device=torch.cuda.current_device())
    context_tokens_tensor = torch.empty(batch_size, seq_len, dtype=torch.int64, device=torch.cuda.current_device())
    # Send variables to all ranks
    torch.distributed.broadcast(context_length_tensor, src, model_parallel_group)
    torch.distributed.broadcast(context_tokens_tensor, src, model_parallel_group)

    array_size = torch.empty(1, dtype=torch.int64, device=torch.cuda.current_device())
    torch.distributed.broadcast(array_size, src, model_parallel_group)

    string_tensor = torch.empty(array_size[0], dtype=torch.int8, device=torch.cuda.current_device())
    torch.distributed.broadcast(string_tensor, src, model_parallel_group)
    bytes = string_tensor.cpu().numpy().tobytes()
    end_strings = pickle.loads(bytes)

    return (
        context_length_tensor,
        context_tokens_tensor,
        tokens_to_generate,
        all_probs,
        compute_logprob,
        temperature,
        top_k,
        top_p,
        greedy,
        repetition_penalty,
        min_tokens_to_generate,
        end_strings,
    )


def synced_generate(
    model,
    inference_strategy,
    context_tokens_tensor,
    context_length_tensor,
    tokens_to_generate,
    all_probs,
    temperature,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_logprob=False,
    repetition_penalty=1.2,
    min_tokens_to_generate=0,
    end_strings=[],
):
    context_length = context_length_tensor.min().item()
    tokenizer = model.tokenizer
    if isinstance(tokenizer, TabularTokenizer):
        batch_token_iterator = tab_sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            temperature=temperature,
        )
    else:
        batch_token_iterator = sample_sequence_batch(
            model,
            inference_strategy,
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob=compute_logprob,
            temperature=temperature,
            end_strings=end_strings,
            extra={
                "top_p": top_p,
                "top_k": top_k,
                "greedy": greedy,
                "repetition_penalty": repetition_penalty,
                "min_tokens_to_generate": min_tokens_to_generate,
            },
        )

    for tokens, lengths, output_logits, full_logits in batch_token_iterator:
        context_length += 1

    if parallel_state.is_pipeline_last_stage():
        src = parallel_state.get_pipeline_model_parallel_last_rank()
        group = parallel_state.get_embedding_group()
        if compute_logprob:
            torch.distributed.broadcast(output_logits, src, group)
        if all_probs:
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()
            torch.distributed.broadcast(full_logits, src, group)

    else:
        if parallel_state.is_pipeline_first_stage():
            src = parallel_state.get_pipeline_model_parallel_last_rank()
            group = parallel_state.get_embedding_group()

            if compute_logprob:
                precision = model._trainer.precision
                if precision in [16, "16"]:
                    dtype = torch.float16
                elif precision == "bf16":
                    dtype = torch.bfloat16
                else:
                    dtype = torch.float32
                output_logits = torch.empty(
                    tokens.size(0), context_length - 1, dtype=dtype, device=torch.device("cuda")
                )
                torch.distributed.broadcast(output_logits, src, group)

            if all_probs:
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                full_logits = torch.empty(
                    tokens.size(0),
                    context_length - 1,
                    model.padded_vocab_size,
                    dtype=dtype,
                    device=torch.device("cuda"),
                )
                torch.distributed.broadcast(full_logits, src, group)
    if tokens is not None:
        return tokens[:, :context_length], output_logits, full_logits


def generate(
    model,
    inputs=None,
    tokens_to_generate=0,
    all_probs=False,
    temperature=1.0,
    add_BOS=False,
    top_k=0,
    top_p=0.0,
    greedy=False,
    compute_logprob=False,
    repetition_penalty=1.0,
    min_tokens_to_generate=0,
    end_strings=['<|endoftext|>'],
    **strategy_args,
) -> OutputType:
    """
    Args:
        model (NLPModel): text generative model
        inputs (Union[tuple, List[str]]): if it is a tuple, it is assumed to be (context_tokens_tensor, context_length_tensor). Otherwise it it a list of prompt text strings
        tokens_to_generate (int): The maximum length of the tokens to be generated.
        all_probs (bool): Return the log prob for all the tokens
        temperature (float): sampling temperature
        add_BOS (bool): add the bos token at the begining of the prompt
        top_k (int): The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p (float): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        greedy (bool):  Whether or not to use sampling ; use greedy decoding otherwise
        repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty
        min_tokens_to_generate (int): The minimum length of the tokens to be generated
        strategy_args, the extra arguments are treated as inference strategy arguments
        end_strings, a list of strings to stop generation when they are encountered in the output.
    Returns:
        OutputType: It generates the output in a dictionary type. It has the following keys:
            sentences: List[str], output sentences
            tokens: List[List[str]], output sentences borken into tokens
            logprob: List[Tensor], log prob of generated tokens
            full_logprob: List[Tensor], log prob of all the tokens in the vocab
            token_ids: List[Tensor], output sentence token ids
            offsets: List[List[int]]  # list of tokens start positions in text
    """
    if 'strategy' in strategy_args:
        inference_strategy = strategy_args['strategy']
    else:
        inference_strategy = model_inference_strategy_dispatcher(model, **strategy_args)
    tokenizer = model.tokenizer
    if torch.distributed.get_rank() == get_model_parallel_src_rank():
        if isinstance(inputs, tuple):
            context_tokens_tensor, context_length_tensor = inputs
        else:
            context_tokens_tensor, context_length_tensor = inference_strategy.tokenize_batch(
                inputs, tokens_to_generate, add_BOS
            )

        send_generate_info(
            context_tokens_tensor,
            context_length_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
        )
    else:
        (
            context_length_tensor,
            context_tokens_tensor,
            tokens_to_generate,
            all_probs,
            compute_logprob,
            temperature,
            top_k,
            top_p,
            greedy,
            repetition_penalty,
            min_tokens_to_generate,
            end_strings,
        ) = receive_generate_info()

    output = synced_generate(
        model,
        inference_strategy,
        context_tokens_tensor,
        context_length_tensor,
        tokens_to_generate,
        all_probs,
        temperature,
        compute_logprob=compute_logprob,
        top_k=top_k,
        top_p=top_p,
        greedy=greedy,
        repetition_penalty=repetition_penalty,
        min_tokens_to_generate=min_tokens_to_generate,
        end_strings=end_strings,
    )
    special_tokens = set()
    if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is not None:
        special_tokens.add(tokenizer.pad_token)
    if hasattr(tokenizer, 'eos_token') and tokenizer.eos_token is not None:
        special_tokens.add(tokenizer.eos_token)
    if hasattr(tokenizer, 'bos_token') and tokenizer.bos_token is not None:
        special_tokens.add(tokenizer.bos_token)
    if hasattr(tokenizer, 'cls_token') and tokenizer.cls_token is not None:
        special_tokens.add(tokenizer.cls_token)
    if hasattr(tokenizer, 'unk_token') and tokenizer.unk_token is not None:
        special_tokens.add(tokenizer.unk_token)
    if hasattr(tokenizer, 'sep_token') and tokenizer.sep_token is not None:
        special_tokens.add(tokenizer.sep_token)
    if hasattr(tokenizer, 'mask_token') and tokenizer.mask_token is not None:
        special_tokens.add(tokenizer.mask_token)
    if output is not None:
        decode_tokens, output_logits, full_logits = output
        resp_sentences = []
        resp_sentences_seg = []

        decode_tokens = decode_tokens.cpu().numpy().tolist()
        for decode_token in decode_tokens:
            sentence = tokenizer.ids_to_text(decode_token)
            resp_sentences.append(sentence)
            if not isinstance(tokenizer, TabularTokenizer):
                words = []
                for token in decode_token:
                    if not isinstance(token, Iterable):
                        token = [token]
                    word = tokenizer.ids_to_tokens(token)
                    if isinstance(word, Iterable):
                        word = word[0]
                    if hasattr(tokenizer.tokenizer, 'byte_decoder'):
                        word = bytearray([tokenizer.tokenizer.byte_decoder[c] for c in word]).decode(
                            'utf-8', errors='replace'
                        )
                    words.append(word)
                resp_sentences_seg.append(words)
            else:
                words = tokenizer.text_to_tokens(sentence)
                resp_sentences_seg.append(words)

        # offsets calculation
        all_offsets = []
        for item in resp_sentences_seg:
            offsets = [0]
            for index, token in enumerate(item):
                if index != len(item) - 1:
                    if token in special_tokens:
                        offsets.append(offsets[-1])
                    else:
                        offsets.append(len(token) + offsets[-1])
            all_offsets.append(offsets)

        output = {}
        output['sentences'] = resp_sentences
        output['tokens'] = resp_sentences_seg
        output['logprob'] = output_logits
        output['full_logprob'] = full_logits
        output['token_ids'] = decode_tokens
        output['offsets'] = all_offsets
        output = inference_strategy.post_generation_process(output)
        return output


def switch(val1, val2, boolean):
    boolean = boolean.type_as(val1)
    return (1 - boolean) * val1 + boolean * val2


def sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=False,
    compute_logprob=False,
    type_ids=None,
    temperature=None,
    end_strings=['<|endoftext|>'],
    extra={},
):
    # Importing here to avoid circular import errors

    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    assert (
        model.cfg.get('sequence_parallel', False) == False
    ), 'sequence_parallel should be False during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_granularity', None) is None
    ), 'activations_checkpoint_granularity should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'
    assert (
        model.cfg.get('activations_checkpoint_method', None) is None
    ), 'activations_checkpoint_method should be None during inference. Disable it in the model config if restoring from nemo or in hparams.yaml if restoring from PTL checkpoint'

    tokenizer = model.tokenizer
    # initialize the batch
    with torch.no_grad():
        context_length = context_lengths.min().item()
        inference_strategy.init_batch(context_tokens, context_length)
        # added eos_id to support the function generate_samples_eval that passes
        # eos_id as an argument and needs termination when that id id found.
        eod_id = tokenizer.eos_id
        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None
        all_generated_indices = None  # used to track all generated indices
        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        maxlen = inference_strategy.clip_max_len(maxlen)

        lengths = torch.ones([batch_size]).long().cuda() * maxlen
        while context_length < maxlen:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, maxlen, micro_batch_size, counter, context_length
            )
            output = inference_strategy.forward_step(batch, tensor_shape)

            if parallel_state.is_pipeline_last_stage():

                if compute_logprob:
                    output = output[0]['logits']
                    output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
                    assert output is not None
                    logits = output[:, -1].view(batch_size, -1).contiguous()

                else:
                    logits = output[0]['logits'][:, -1].contiguous()
                    logits = tensor_parallel.gather_from_tensor_model_parallel_region(logits)
                    assert logits is not None
                    logits = logits.view(batch_size, -1)

                # make sure it will generate at least min_length
                min_length = extra.get('min_tokens_to_generate', 0)
                if min_length > 0:
                    within_min_length = (context_length - context_lengths) < min_length
                    logits[within_min_length, eod_id] = -float('Inf')

                # make sure it won't sample outside the vocab_size range
                logits[:, tokenizer.vocab_size :] = -float('Inf')

                # started indicates whether the current token step passes the context_length, so we make sure not to overwrite the context tokens

                started = context_lengths <= context_length
                if extra.get('greedy', False):
                    prev = torch.argmax(logits, dim=-1).view(-1)
                else:
                    logits = logits.float()
                    logits /= temperature
                    # handle repetition penality
                    logits = repetition_penalty(logits, extra.get('repetition_penalty', 1.2), all_generated_indices)
                    logits = top_k_logits(
                        logits, top_k=extra.get('top_k', 0), top_p=extra.get('top_p', 0.9), started=started
                    )
                    probs = F.softmax(logits, dim=-1)
                    prev = torch.multinomial(probs, num_samples=1).view(-1)

                # Clamp the predicted out of vocabulary tokens
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)
                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # Replace sampled tokens w/ done token if EOD has already been sampled
                new_tokens = switch(new_tokens, eod_id, is_done)

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

                # Insert either new predicted or next prompt token
                tokens[:, context_length] = new_tokens

                if compute_logprob:
                    if output_logits is None:
                        output = F.log_softmax(output[:, :context_length, :], 2)

                        indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                        output_logits = torch.gather(output, 2, indices).squeeze(2)
                        all_generated_indices = indices[:, :, 0]
                        if all_probs:
                            full_logits = output
                    else:
                        output = F.log_softmax(output, 2)
                        indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                        new_output_logits = torch.gather(output, 2, indices).squeeze(2)

                        # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                        output_logits = torch.cat([output_logits, new_output_logits], 1)
                        all_generated_indices = torch.cat([all_generated_indices, indices[:, :, 0]], 1)
                        if all_probs:
                            full_logits = torch.cat([full_logits, output], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                #                done_token = (prev == eod_id).byte() & started.byte()
                done_token = inference_strategy.end_of_generation_condition(
                    tokens[:, : context_length + 1], prev, eod_id, end_strings
                )
                done_token = done_token.byte() & started.byte()

                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if compute_logprob:
                    if all_probs:
                        yield tokens, lengths, output_logits, full_logits
                    else:
                        yield tokens, lengths, output_logits, None
                else:
                    yield tokens, lengths, None, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def tab_sample_sequence_batch(
    model,
    inference_strategy,
    context_tokens,
    context_lengths,
    tokens_to_generate,
    all_probs=True,
    type_ids=None,
    temperature=None,
):
    app_state = AppState()
    micro_batch_size = context_tokens.shape[0]
    _reconfigure_microbatch_calculator(
        rank=app_state.global_rank,
        rampup_batch_size=None,
        global_batch_size=micro_batch_size,
        micro_batch_size=micro_batch_size,
        data_parallel_size=1,
    )
    tokenizer = model.tokenizer
    sizes = tokenizer.code_column.sizes
    tokens_per_row = sum(sizes) + 1
    columns = tokenizer.code_column.columns
    num_columns = len(columns)
    tokenid_range = []
    for i in range(num_columns):
        tokenid_range.extend(tokenizer.code_column.get_range(i))
    # initialize the batch
    with torch.no_grad():
        context_length = context_lengths.min().item()
        inference_strategy.init_batch(context_tokens, context_length)
        context = context_tokens[:, :context_length]
        # the context may start in the middle of the row,
        # calculate the offset according to the position of '\n' or '<|endoftext|>'
        positions = torch.where(context == tokenizer.eor)[1]
        if len(positions) == 0:
            positions = torch.where(context == tokenizer.eod)[1]
        if len(positions) != 0:
            max_position = positions.max().item()
            # TODO, need to make sure context of different batch have the same offset lengths")
            # otherwise, need to calculate offset per batch_id
            offset = (context_length - max_position - 1) % tokens_per_row
        else:
            offset = 0

        eod_id = tokenizer.eos_id

        counter = 0

        batch_size = context_tokens.size(0)
        is_done = torch.zeros([batch_size]).byte().cuda()
        tokens = context_tokens
        output_logits = None

        # Generate enough tokens for the longest sequence
        maxlen = tokens_to_generate + context_lengths.max().item()

        if maxlen > model.cfg.encoder_seq_length:
            maxlen = model.cfg.encoder_seq_length

        lengths = torch.ones([batch_size]).long().cuda() * maxlen

        while context_length < maxlen:
            batch, tensor_shape = inference_strategy.prepare_batch_at_step(
                tokens, maxlen, micro_batch_size, counter, context_length
            )
            output = inference_strategy.forward_step(batch, tensor_shape)

            if parallel_state.is_pipeline_last_stage():
                output = output[0]['logits'].float()
                output = tensor_parallel.gather_from_tensor_model_parallel_region(output)
                assert output is not None
                output = output.float()
                logits = output[:, -1].view(batch_size, -1).contiguous()
                token_in_row = (counter + offset) % tokens_per_row
                logits = logits.float()
                logits /= temperature
                if token_in_row == tokens_per_row - 1:
                    # line break
                    eor_id = tokenizer.eor
                    eod_id = tokenizer.eos_id
                    min_id = min(eor_id, eod_id)
                    max_id = max(eor_id, eod_id) + 1
                    logits = tab_logits(logits, min_id, max_id)
                else:
                    # limit the range
                    min_id, max_id = tokenid_range[token_in_row]
                    logits = tab_logits(logits, min_id, max_id)
                probs = F.softmax(logits, dim=-1)
                prev = torch.multinomial(probs, num_samples=1).view(-1)
                started = context_lengths <= context_length
                # Clamp the out of vocabulary tokens.
                prev = torch.clamp(prev, max=tokenizer.vocab_size - 1)

                new_tokens = switch(tokens[:, context_length].view(-1), prev, started)

                # post process the inference tokens based on the strategy
                inference_strategy.post_process(tokens, new_tokens, context_length)

                tokens[:, context_length] = new_tokens

                if output_logits is None:
                    output_context = F.log_softmax(output[:, :context_length, :], 2)
                    indices = torch.unsqueeze(tokens[:, 1 : context_length + 1], 2)
                    output_logits = torch.gather(output_context, 2, indices).squeeze(2)
                    if all_probs:
                        full_logits = output_context
                else:
                    output_context = F.log_softmax(output, 2)
                    indices = torch.unsqueeze(new_tokens, 1).unsqueeze(2)
                    new_output_logits = torch.gather(output_context, 2, indices).squeeze(2)

                    # TODO(rprenger) we're copying output_logits every time.  Should pre-allocate
                    output_logits = torch.cat([output_logits, new_output_logits], 1)
                    if all_probs:
                        full_logits = torch.cat([full_logits, output_context], 1)

                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_embedding_group()
                torch.distributed.broadcast(new_tokens, src, group)

                done_token = (prev == eod_id).byte() & started.byte()
                just_finished = (done_token & ~is_done).bool()
                lengths[just_finished.view(-1)] = context_length
                is_done = is_done | done_token

                done = torch.all(is_done)
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)
                if all_probs:
                    yield tokens, lengths, output_logits, full_logits
                else:
                    yield tokens, lengths, output_logits, None

            else:
                if parallel_state.is_pipeline_first_stage():
                    src = parallel_state.get_pipeline_model_parallel_last_rank()
                    group = parallel_state.get_embedding_group()
                    new_tokens = torch.empty_like(tokens[:, context_length])
                    torch.distributed.broadcast(new_tokens, src, group)
                    tokens[:, context_length] = new_tokens
                    yield tokens, None, None, None
                else:
                    yield None, None, None, None

                done = torch.cuda.ByteTensor([0])
                src = parallel_state.get_pipeline_model_parallel_last_rank()
                group = parallel_state.get_pipeline_model_parallel_group()
                torch.distributed.broadcast(done, src, group)

            context_length += 1
            counter += 1
            if done:
                break


def sample_token_greedy(logits):
    """
    Greedy sampling. Returns the token with the highest probability, and corresponding log_prob.

    Args:
        logits: [batch_size, vocab_size] - unnormalized log probabilities of the next token
    
    Returns:
        log_probs: [batch_size] - log probabilities of the sampled tokens
        token_ids: [batch_size] - sampled token ids
    """
    log_probs, token_ids = torch.max(torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)

    return log_probs, token_ids


def sample_token_topk(logits, top_k=0, top_p=0.0, temperature=1.0, filter_value=-float('Inf')):
    """
    Greedy sampling. Returns the token with the highest probability, and corresponding log_prob.

    Args:
        logits: [batch_size, vocab_size] - unnormalized log probabilities of the next token
        top_k: int - if > 0: only sample from top k tokens with highest probability
        top_p: float - if > 0.0: only sample from a subset of candidates, where the cumulative probability
        temperature: float - temperature for sampling
        filter_value: float - value to set filtered tokens to
    
    Returns:
        log_probs: [batch_size] - log probabilities of the sampled tokens
        token_ids: [batch_size] - sampled token ids
    """
    logits = logits.float()
    logits /= temperature
    logits = top_k_logits(logits, top_k=top_k, top_p=top_p, filter_value=filter_value)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    token_ids = torch.multinomial(log_probs.exp(), num_samples=1).view(-1)
    log_probs = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)

    return log_probs, token_ids


def sample_token_topk_beam_search(logits: torch.Tensor, beam_size: int = 1, dim: int = -1, log_softmax: bool = True):
    """
    Beam search selection of top K predictions per target (dim). Returns the beam_size tokens ids with the highest
    probability and the corresponding log_prob per target

    Args:
        logits: [batch_size, vocab_size] or [batch_size, vocab_size] - unnormalized log probabilities of the next token,
        beam_size: int > 1 - number of tokens to return with the highest probability per target
        dim: int - dim of log_softmax and topk selection
        log_softmax: bool - if to calculate log softmax  for log probabilities


    Returns:
        log_probs: [batch_size, beam_size] - log probabilities of the sampled tokens
        token_ids: [batch_size, beam_size] - sampled token ids
    """
    if log_softmax:
        log_probs = torch.nn.functional.log_softmax(logits, dim=dim)
    else:
        log_probs = logits
    # get top candidates for each item in batch
    log_probs, token_ids = torch.topk(log_probs, beam_size, dim=dim)

    return log_probs, token_ids


def compute_beam_search_len_penalty(lengths: torch.Tensor, alpha: int) -> torch.Tensor:
    """
    Length penalty used in the beam search
    Args:
        lengths: lengths of decoded sequences
        alpha: params of the penalty
    Returns:
         tensor with the penalty value
    """
    return ((5 + lengths) / 6).pow(alpha)


def get_sampling_token_fn(sampling_method: str, sampling_kwargs: dict) -> Tuple[Callable, dict]:
    """
    Specifies the sampling function that takes in a tensor of logits [batch_size, vocab_size] and returns a tuple
    (tensor of log_probs [batch_size], tensor of sampled from logits [batch_size]).
    If the beam search is enabled, the sampling function returns tensors [batch_size, beam_size]

    Args:
        sampling_method: the sampling method to use in the decode steps. Currently supported methods are
                          "beam-search"/"greedy"/"topkp"
        sampling_kwargs: dict with arguments to be passed to the sampling function.
                         For sampling method 'beam-search', the following kwargs are supported:
                         beam_size - int, number of the best sequences at each decode iteration to be left per target
                         beam_alpha - int, the parameter of length penalty applied to predicted sequences
                         keep_only_best_tokens - used in the beam search, boolean flag if to output only best sequence
                                                 of predicted tokens (True) or beam_size predictions per target
                         return_scores - used in the beam search, boolean flag if to return scores at the top of
                                         predictions and logits

    Returns:
        sample_token_fn: the sampling function
        default_sampling_kwargs: sampling_kwargs augmented with default sampling kwargs
    """
    all_default_sampling_kwargs = {
        'greedy-search': {},
        'topkp-sampling': {'top_k': 0, 'top_p': 0.0, 'temperature': 1.0},
        'beam-search': {'beam_size': 1, 'beam_alpha': 0.0, 'keep_only_best_tokens': False, 'return_scores': False},
    }

    # update default sampling kwargs with user provided kwargs
    default_sampling_kwargs = all_default_sampling_kwargs[sampling_method].copy()
    default_sampling_kwargs.update(sampling_kwargs)
    # sampling_kwargs = default_sampling_kwargs

    if sampling_method == 'greedy-search':
        sampling_token_fn = sample_token_greedy

    elif sampling_method == "topkp-sampling":
        top_k = default_sampling_kwargs['top_k']
        top_p = default_sampling_kwargs['top_p']
        temperature = default_sampling_kwargs['temperature']
        sampling_token_fn = partial(sample_token_topk, top_k=top_k, top_p=top_p, temperature=temperature)

    elif sampling_method == "beam-search":
        beam_size = default_sampling_kwargs['beam_size']
        sampling_token_fn = partial(sample_token_topk_beam_search, beam_size=beam_size)

    else:
        raise ValueError(
            f'Invalid sampling method {sampling_method}. '
            f'Supported sampling methods are {all_default_sampling_kwargs.keys()}'
        )

    return sampling_token_fn, default_sampling_kwargs
