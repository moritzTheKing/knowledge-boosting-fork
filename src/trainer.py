import os, glob
import argparse

import torch
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import wandb
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning import Fabric

import time

import src.utils as utils


def main(args, hparams):    
    os.makedirs(args.run_dir, exist_ok=True)
    
    # Setting all the random seeds to the same value.
    # This is important in a distributed training setting.
    # Each rank will get its own set of initial weights.
    # If they don't match up, the gradients will not match either,
    # leading to training that may not converge.
    pl.seed_everything(42)

    # Initialize the model
    pl_module = utils.import_attr(hparams.pl_module)(**hparams.pl_module_args)

    # print stuff for easier logging
    utils.count_parameters(pl_module)

    # Init trainer
    #wandb_run_name = os.path.basename(args.run_dir.rstrip('/'))
    wandb_logger = TensorBoardLogger(save_dir=args.run_dir, name='lightning_logs') #WandbLogger(project='kd-inference', save_dir=args.run_dir, name=wandb_run_name)
    # Init callbacks
    callbacks = [utils.import_attr(hparams.pl_logger)()]
    if not args.test:
        callbacks += [
            ModelCheckpoint(
                dirpath=args.run_dir, monitor=None, save_last=True),
            ModelCheckpoint(
                dirpath=os.path.join(args.run_dir, 'best'),
                filename="{epoch}-{step}",
                monitor=pl_module.monitor, mode=pl_module.monitor_mode, save_top_k=5)
        ]

    if 'scheduler' in hparams.pl_module_args and hparams.pl_module_args['scheduler'] is not None:
        print("Logging learning rate")
        lr_logger = LearningRateMonitor(logging_interval='step')
        callbacks += [lr_logger]
        
    # Gradient clipping
    grad_clip = 0
    if hasattr(hparams, 'grad_clip'):
        print("USING GRADIENT CLIPPING", hparams.grad_clip)
        grad_clip = hparams.grad_clip

    print("so viele GPUs werden zum Training genutzt", torch.cuda.device_count())
    # Init trainer
    trainer = pl.Trainer(
        accelerator="gpu", devices=torch.cuda.device_count(), strategy='ddp', 
        max_epochs=hparams.epochs,
        logger=wandb_logger, limit_train_batches=args.frac, gradient_clip_val=grad_clip, # callbacks=callbacks
        limit_val_batches=args.frac, limit_test_batches=args.frac)

    # Maximum 4 worker per GPU
    num_workers = min(len(os.sched_getaffinity(0)), 4)

    # Choose checkpoint to use
    # If provided, use checkpoint given
    ckpt_path = None
    if args.ckpt is not None:
        ckpt_path = args.ckpt
    else:
        # If testing, choose best if available
        if args.test:
            best_ckpts = sorted(glob.glob(os.path.join(args.run_dir, '*.ckpt')))
            print("best ckpts", best_ckpts)
            if len(best_ckpts) > 0:
                ckpt_path = best_ckpts[-1]
        # If training, choose last if available
        else:
            if os.path.exists(os.path.join(args.run_dir, 'last.ckpt')):
                ckpt_path = os.path.join(args.run_dir, 'last.ckpt')
                print("last ckpts", ckpt_path)

    print("the ckpt we used", ckpt_path)

    # Test and return if --test flag is set
    if args.test:

        # Initialize the test dataset
        test_ds = utils.import_attr(hparams.test_dataset)(**hparams.test_data_args)
        batch_size = hparams.eval_batch_size
        test_dl = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        results = trainer.test(pl_module, test_dl, ckpt_path=ckpt_path, verbose=True)

        for loss in results:
            print(f"Results hat {len(loss)} Elemente: {loss}")
        
        return

    # Initialize the train dataset
    train_ds = utils.import_attr(hparams.train_dataset)(**hparams.train_data_args)
    train_batch_size = int(hparams.batch_size / 1) # Batch size per GPU
    train_dl = torch.utils.data.DataLoader(
        train_ds, batch_size=train_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )

    # Initialize the validation dataset
    val_ds = utils.import_attr(hparams.val_dataset)(**hparams.val_data_args)
    val_dl = torch.utils.data.DataLoader(
        val_ds, batch_size=hparams.eval_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Train
    trainer.fit(pl_module, train_dl, val_dl, ckpt_path=ckpt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run_dir', type=str, required=True)
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Resume training with an existing run_dir.')
    parser.add_argument(
        '--ckpt', type=str, default=None,
        help='Path to the checkpoint to resume training from. If not specified, the '
             'last.ckpt in run_dir will be used, if available.')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Fraction of the dataset to use for train, val and test.')
    args = parser.parse_args()

    if os.environ.get('LOCAL_RANK', None) is None:
        if not args.resume and not args.test:
            assert not os.path.exists(args.run_dir), \
                f"run_dir {args.run_dir} already exists. " \
                "Use --resume if you intend to resume an existing run."

    # Load hyperparameters
    hparams = utils.Params(args.config)

    print("Batchsize:", hparams.batch_size)

    # Start der Zeitmessung
    start_time = time.time()

    # Run
    main(args, hparams)

    # Ende der Zeitmessung
    end_time = time.time()
    runtime_seconds = end_time - start_time

    # Umrechnung in Stunden, Minuten und Sekunden
    hours = int(runtime_seconds // 3600)
    minutes = int((runtime_seconds % 3600) // 60)
    seconds = runtime_seconds % 60

    print(f"Job-Laufzeit: {hours}h {minutes}m {seconds:.2f}s")