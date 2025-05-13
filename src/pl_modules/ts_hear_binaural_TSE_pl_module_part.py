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
from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality as PESQ
from torchmetrics.functional.audio.stoi import short_time_objective_intelligibility as STOI
import torchaudio


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

        print("Wert von D aus SM-TSE-25.json model:", model_params['D'])
        print("Wert von L aus SM-TSE-25.json model:", model_params['L'])
        print("Wert von I aus SM-TSE-25.json model:", model_params['I'])
        print("Wert von J aus SM-TSE-25.json model:", model_params['J'])
        print("Wert von B aus SM-TSE-25.json model:", model_params['B'])
        print("Wert von H aus SM-TSE-25.json model:", model_params['H'])
        self.model = utils.import_attr(model)(**model_params) 

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

    def _step(self, batch, batch_idx, step='train'):
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

            # log metrics for model output and ground truth
            output_m_norm = output_m / torch.abs(output_m).max() * torch.abs(targets['target']).max()
            pesq_m = PESQ(preds=output_m_norm, target=targets['target'], fs=16000, mode="wb")
            self.log(
                f'{step}/pesq_m_output_gt', pesq_m.mean().to(self.device),
                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
                sync_dist=True)
            
            stoi_m = STOI(preds=output_m_norm, target=targets['target'], fs=16000)
            self.log(
                f'{step}/stoi_m_output_gt', stoi_m.mean().to(self.device),
                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
                sync_dist=True)
            
            # log metrics for model input (mixture) and ground truth
            inputs['mixture'] = inputs['mixture'] / torch.abs(inputs['mixture']).max() * torch.abs(targets['target']).max() # TODO: weiß nicht ob ich diese Zeile noch brauche
            pesq_m = PESQ(preds=inputs['mixture'], target=targets['target'], fs=16000, mode="wb")
            self.log(
                f'{step}/pesq_m_input_gt', pesq_m.mean().to(self.device),
                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
                sync_dist=True)
            
            stoi_m = STOI(preds=inputs['mixture'], target=targets['target'], fs=16000)
            self.log(
                f'{step}/stoi_m_input_gt', stoi_m.mean().to(self.device),
                batch_size=batch_size, on_step=False, on_epoch=True, prog_bar=True,
                sync_dist=True)

        if step =='test':
            # save the samples for the first 20 batches to listen to them
            if batch_idx < 20:
                for sample_idx in range(output_m.size(0)):
                    output_m_norm = output_m / torch.abs(output_m).max() * torch.abs(targets['target']).max()

                    # .detach() sorgt dafür, dass der Tensor vom Berechnungsgraphen getrennt wird, sodass keine Gradienten mehr berechnet werden
                    # .cpu() sorgt dafür, dass der Tensor auf die CPU verschoben wird (das erwartet torchaudio.save)
                    m_audio = output_m_norm[sample_idx].detach().cpu()

                    # Wenn das Audio nur einen Channel (Mono, kein Stereo) hat, füge einen Dummy-Kanal hinzu
                    if m_audio.ndim == 1:
                        m_audio = m_audio.unsqueeze(0)

                    current_dir = os.path.dirname(__file__)
                    base_output_dir = os.path.join(current_dir, "..", "data", "inference_samples")
                    job_id = os.environ.get("SLURM_JOB_ID")

                    # Erstelle einen Subordner, der nach der JobID benannt ist
                    output_dir = os.path.join(base_output_dir, f"job_{job_id}_big", "Output")
                    os.makedirs(output_dir, exist_ok=True)
                    m_filename = os.path.join(output_dir, f"output_sm_batch{batch_idx}_sample{sample_idx}.wav")
                    
                    torchaudio.save(m_filename, m_audio, 16000)

                # Trainingsdaten Mixture und Targets abspeichern zum anhören
                for bat_idx in range(batch_size):
                    mixtures_audio = inputs["mixture"][bat_idx].detach().cpu()
                    targets_audio = targets["target"][bat_idx].detach().cpu()


                    if mixtures_audio.ndim == 1:
                        mixtures_audio = mixtures_audio.unsqueeze(0)

                    if targets_audio.ndim == 1:
                        targets_audio = targets_audio.unsqueeze(0)   

                    current_dir = os.path.dirname(__file__)
                    base_output_dir = os.path.join(current_dir, "..", "data", "inference_samples")
                    job_id = os.environ.get("SLURM_JOB_ID")
                    output_dir_mix = os.path.join(base_output_dir, f"job_{job_id}_big", "Mixture")
                    output_dir_tar = os.path.join(base_output_dir, f"job_{job_id}_big", "Targets")


                    os.makedirs(output_dir_mix, exist_ok=True)
                    os.makedirs(output_dir_tar, exist_ok=True)

                    mixtures_data = os.path.join(output_dir_mix, f"mixture_batch{batch_idx}_sample_{bat_idx}.wav")
                    targets_data = os.path.join(output_dir_tar, f"targets_batch{batch_idx}_sample_{bat_idx}.wav")

                    torchaudio.save(mixtures_data, mixtures_audio, 16000)
                    torchaudio.save(targets_data, targets_audio, 16000)


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
        loss_m, sample = self._step(batch, batch_idx, step='train')

        # Save some outputs for visualization
        if batch_idx % 200 == 0:
            self.train_samples.append(sample)

        return loss_m

    def validation_step(self, batch, batch_idx):
        _, sample = self._step(batch, batch_idx, step='val')

        # Save some outputs for visualization
        if batch_idx % 10 == 0:
            self.val_samples.append(sample)

        return sample['output_m']

    def test_step(self, batch, batch_idx):
        _, sample = self._step(batch, batch_idx, step='test') # hier ggf noch was ändern

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