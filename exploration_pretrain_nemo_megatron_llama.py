# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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

# Notes:
# 1. update dataset loading in scripts: NeMo/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py and NeMo/nemo/collections/nlp/data/language_modeling/megatron/gpt_dataset.py
# 2. TODO Could not import distributed_fused_adam optimizer from Apex?
# 3. TODO could not install transformer-engine https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/installation.html?
# 4. TODO, understand distributed_fused_adam? sequence_parallel? why virtual pipeline parallel (this makes self.model a list)? get_attention_mask_from_fusion? gradient_accumulation_fusion? transformer-engine (AutocastTransformerLayer)? headscale? normalize_attention_scores? nsys_profile?
# 5. TODO, resume from mid epoch check point is not supported by pytorch lightning trainer, https://pytorch-lightning.readthedocs.io/en/1.6.2/advanced/fault_tolerant_training.html, and this fault tolerant training also not working properly?
# 6. TODO, data loading problem! not able to load in all data using memoryview, (some cannot pickle error); and all data load in using numpy exceed cpu memory; only load in partial data for now, but this will affect learning rate decay; load data follow https://lightning.ai/forums/t/how-to-switch-dataloaders-between-epochs-in-lightning/137??
# 7. TODO, sequence parallel for RMS norm not supoorted, need to have similar support for FastRMSNorm (c++ needed?) https://github.com/NVIDIA/apex/blob/master/apex/contrib/layer_norm/layer_norm.py
# 8. TODO, xformers memory efficient attention (added, but not supporting relative position embedding)
# 9. TODO, Anything else common used in industry? anything in deepspeed but not in nemo-megatron?

import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.plugins.environments import TorchElasticEnvironment
from pytorch_lightning.trainer.connectors.checkpoint_connector import CheckpointConnector

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import (
    GradScaler,
    MegatronHalfPrecisionPlugin,
    NLPDDPStrategy,
    PipelineMixedPrecisionPlugin,
)
from nemo.core.config import hydra_runner
from nemo.utils import logging
from nemo.utils.exp_manager import exp_manager
import torch

mp.set_start_method("spawn", force=True)


@hydra_runner(config_path=".", config_name="exploration_pretrain_nemo_megatron_llama")
def main(cfg) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f'\n{OmegaConf.to_yaml(cfg)}')

    megatron_amp_o2 = cfg.model.get('megatron_amp_O2', False)
    with_distributed_adam = cfg.model.optim.get('name') == 'distributed_fused_adam'

    plugins = []
    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,  # we don't use DDP for async grad allreduce
        gradient_as_bucket_view=cfg.model.gradient_as_bucket_view,
        find_unused_parameters=False,
    )
    if cfg.trainer.precision in [16, 'bf16']:
        scaler = None
        if cfg.trainer.precision == 16:
            scaler = GradScaler(
                init_scale=cfg.model.get('native_amp_init_scale', 2 ** 32),
                growth_interval=cfg.model.get('native_amp_growth_interval', 1000),
                hysteresis=cfg.model.get('hysteresis', 2),
            )
        if megatron_amp_o2 and not with_distributed_adam:
            plugins.append(MegatronHalfPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))
        else:
            plugins.append(PipelineMixedPrecisionPlugin(precision=cfg.trainer.precision, device='cuda', scaler=scaler))

    if cfg.get('cluster_type', None) == 'BCP':
        plugins.append(TorchElasticEnvironment())

    trainer = Trainer(plugins=plugins, strategy=strategy, **cfg.trainer)

    exp_manager(trainer, cfg.exp_manager)

    # update resume from checkpoint found by exp_manager
    if cfg.model.resume_from_checkpoint is not None:
        resume_from_checkpoint = cfg.model.resume_from_checkpoint
    else:
        resume_from_checkpoint = trainer._checkpoint_connector.resume_from_checkpoint_fit_path

    logging.info(f'Resuming training from checkpoint: {resume_from_checkpoint}')

    trainer._checkpoint_connector = CheckpointConnector(trainer, resume_from_checkpoint=resume_from_checkpoint)

    # hydra interpolation does not work here as the interpolation key is lost when PTL saves hparams
    with open_dict(cfg):
        cfg.model.precision = cfg.trainer.precision

    model = MegatronGPTModel(cfg.model, trainer)

    trainer.fit(model)


if __name__ == '__main__':
    # tf32 option 1 (?)
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # tf32 option 2 (?)
    torch.set_float32_matmul_precision('high')
    main()
