import os

import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch import Callback
from torchmetrics.functional import(
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr)
import wandb
from transformers.debug_utils import DebugUnderflowOverflow

from src.losses.LossFn import LossFn

import src.utils as utils

class TSHearPLModule(pl.LightningModule):
    def __init__(self, 
                 model, model_params,
                 sr, freeze_bm = False,
                 init_ckpt = None,
                 optimizer=None, optimizer_params=None,
                 scheduler=None, scheduler_params = None,
                 loss_params=None):
        super(TSHearPLModule, self).__init__()

        self.model = utils.import_attr(model)(**model_params) 

        """print("D has the value: ", self.model.D)
        print("L has the value: ", self.model.L)
        print("I has the value: ", self.model.I)
        print("J has the value: ", self.model.J)
        print("B has the value: ", self.model.B)
        print("H has the value: ", self.model.H)"""
        
        if init_ckpt is not None:
            m_ckpt = torch.load(init_ckpt)
            self.model.load_state_dict(m_ckpt)
            # state_dict = torch.load(init_ckpt)['state_dict']
            # state_dict = {k[6:] : v for k, v in state_dict.items() if k.startswith('model.') }

        if freeze_bm:
            for param in self.model.parameters():
                param.requires_grad = False

        # debug_overflow = DebugUnderflowOverflow(self.joint_model)
        self.sr = sr

        # Values to log
        self.val_samples = []
        self.train_samples = []

        # Metric to monitor
        self.monitor = 'val/si_snr_i_sm'
        self.monitor_mode = 'max'

        # Initialize loss function
        loss_args = {}
        if loss_params is not None:
            loss_args = loss_params
        self.loss_fn = LossFn(**loss_args)

        # Initialize optimizer
        self.optimizer = utils.import_attr(optimizer)(self.parameters(), **optimizer_params)

        # Initialize scheduler
        if scheduler is not None:
            if scheduler == 'sequential':
                schedulers = []
                milestones = []
                for scheduler_param in scheduler_params:
                    sched = utils.import_attr(scheduler_param['name'])(self.optimizer, **scheduler_param['params'])
                    schedulers.append(sched)
                    milestones.append(scheduler_param['epochs'])

                # Cumulative sum for milestones
                for i in range(1, len(milestones)):
                    milestones[i] = milestones[i-1] + milestones[i]

                # Remove last milestone as it is implied by num epochs
                milestones.pop()

                self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, schedulers, milestones)
            else:
                self.scheduler = utils.import_attr(scheduler)(self.optimizer, **scheduler_params)
        else:
            self.scheduler = scheduler

    def forward(self, x):
        return self.model(x)['output']

    def _step(self, batch, step='train'):
        inputs, targets = batch
        batch_size = inputs['mixture'].shape[0]

        # Forward pass
        y_m = self.model(inputs)
        output_m = y_m['output']

        # Compute loss and reorder outputs
        loss_m = self._loss(pred=output_m, tgt=targets['target'])

        # Log metrics for large model
        snr_i_m = torch.mean(self._metric_i(snr, inputs['mixture'], output_m, targets['target']))
        si_snr_i_m = torch.mean(self._metric_i(si_snr, inputs['mixture'], output_m, targets['target']))

        # Log small model metrics
        on_step = step == 'train'
        self.log(
            f'{step}/loss_sm', loss_m, batch_size=batch_size, on_step=on_step,
            on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            f'{step}/snr_i_sm', snr_i_m.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=False,
            sync_dist=True)
        self.log(
            f'{step}/si_snr_i_sm', si_snr_i_m.mean(),
            batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
            sync_dist=True)

        # Log additional metrics for validation and test
        if step in ['val', 'test']:
            pass

        output_m = output_m / torch.abs(output_m).max() * torch.abs(targets['target']).max()

        sample = {
            'mixture': inputs['mixture'],
            'target': targets['target'],
            'output_m': output_m.detach(),
        }

        return loss_m, sample

    def get_torch_model(self):
        return self.model

    def _loss(self, pred, tgt, **kwargs):
        return self.loss_fn(pred, tgt, **kwargs)

    def _metric_i(self, metric, src, pred, tgt):
        _vals = []
        for s, t, p in zip(src, tgt, pred):
            _vals.append((metric(p, t) - metric(s, t)).mean())
        return torch.stack(_vals)

    def training_step(self, batch, batch_idx):
        loss_m, sample = self._step(batch, step='train')

        # Save some outputs for visualization
        if batch_idx % 200 == 0:
            self.train_samples.append(sample)

        return loss_m

    def validation_step(self, batch, batch_idx):
        _, sample = self._step(batch, step='val')

        # Save some outputs for visualization
        if batch_idx % 10 == 0:
            self.val_samples.append(sample)

        return sample['output_m']

    def test_step(self, batch, batch_idx):
        _, sample = self._step(batch, step='test') # hier ggf noch was ändern

        # Save some outputs for visualization
        if batch_idx % 10 == 0:
            self.val_samples.append(sample)

        return sample['output_m']

    def configure_optimizers(self):
        if self.scheduler is not None:
            # For reduce LR on plateau, we need to provide more information
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler_cfg = {
                    "scheduler": self.scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "monitor": self.monitor,
                    "strict": False
                }
            else:
                scheduler_cfg = self.scheduler
            return [self.optimizer], [scheduler_cfg]
        else:
            return self.optimizer

class TSHearLogger(Callback):
    def _log_audio(self, logger, key, samples, sr):
        columns = ['mixture', 'target', 'output_bm', 'output_sm']
        wandb_samples = []
        for i, sample in enumerate(samples):
            for k in columns:
                wandb_samples.append(wandb.Audio(
                    sample[k][0].permute(1, 0).cpu().numpy(),
                    sample_rate=sr, caption=f'{i}/{k}'))
        logger.experiment.log({key: wandb_samples})

    def on_epoch_start(self):
        print('\n')

    def on_train_epoch_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "train/audio_samples", pl_module.train_samples,
            sr=pl_module.sr)
        pl_module.train_samples.clear()

    def on_validation_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "val/audio_samples", pl_module.val_samples,
            sr=pl_module.sr)
        pl_module.val_samples.clear()

    def on_test_end(self, trainer, pl_module):
        self._log_audio(
            trainer.logger, "test/audio_samples", pl_module.val_samples,
            sr=pl_module.sr)
        pl_module.val_samples.clear()