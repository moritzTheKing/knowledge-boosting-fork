import os
import argparse
import torch
import torchaudio
from torch.utils.data import DataLoader
from datasets.BakedTSE_dataset import BakedTSE_dataset

class Inference():
    def __init__(self):
        pass

    def run(self, args):

            # create Dataset
            dataset = BakedTSE_dataset(mixtures_dir=args.mixtures_dir,)
            dataloader = DataLoader(dataset, shuffle=False)

            # create output directory
            os.makedirs(args.output_dir, exist_ok=True)

            for i, (inputs, targets) in 20:
                mixture = inputs['mixture'].squeeze(0)
                rolled  = inputs['_mixture'].squeeze(0)
                target  = targets['target'].squeeze(0)
                spk_id  = inputs['spk_id']
                sample_dir = inputs['sample_dir'][0]

                # filenames for output directory
                base_filename = f"sample_{i}"
                mixture_filename = os.path.join(args.output_dir, base_filename + "_mixture.wav")
                target_filename  = os.path.join(args.output_dir, base_filename + "_target.wav")
                rolled_filename  = os.path.join(args.output_dir, base_filename + "_rolled.wav")
                spk_id_filename  = os.path.join(args.output_dir, base_filename + "_spkid.pt")

                # save audiofiles
                torchaudio.save(mixture_filename, mixture)
                torchaudio.save(target_filename, target)
                torchaudio.save(rolled_filename, rolled)
                # save speaker id
                torch.save(spk_id, spk_id_filename)

                print(f"Sample {i} aus '{sample_dir}' gespeichert.")