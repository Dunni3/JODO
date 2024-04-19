from run_lib_mod import evaluate
import run_lib
from absl import app, flags
from ml_collections.config_flags import config_flags
import logging
import os
from rdkit import RDLogger
import argparse
import sys
from pathlib import Path
import pickle
from functools import partial

def parse_args():
    p = argparse.ArgumentParser(description="VPSDE evaluation script")
    p.add_argument('--model', type=str, required=True)
    p.add_argument('--output_file', type=Path, help='Path to the output file', required=True)
    p.add_argument('--n_mols', type=int, default=10, help='Number of molecules to sample')
    p.add_argument('--max_batch_size', type=int, default=100, help='Maximum batch size')
    p.add_argument('--sampling_steps', type=int, default=1000)

    args, unknown = p.parse_known_args()

    if args.model not in ['qm9', 'geom']:
        raise ValueError("Model not recognized: " + args.model)



    # modify command line arguments according to the model chosen
    if args.model == 'qm9':
        argv = sys.argv 
        argv_new = argv[:1] + [
                                '--config=configs/vpsde_qm9_uncond_jodo.py', 
                                '--mode=eval', '--workdir=exp_uncond/vpsde_qm9_jodo', 
                                "--config.eval.ckpts=30", 
                                f"--config.eval.num_samples={args.n_mols}",
                                f'--config.eval.batch_size={args.max_batch_size}', 
                                f'--config.sampling.steps={args.sampling_steps}',
                            ]
    else:
        argv = sys.argv
        argv_new = argv[:1] + [
                                '--config=configs/vpsde_geom_uncond_jodo.py', 
                                '--mode=eval', '--workdir=exp_uncond/vpsde_geom_jodo_base', 
                                "--config.model.n_layers=6",
                                "--config.model.nf=128",
                                "--config.eval.ckpts=30", 
                                f"--config.eval.num_samples={args.n_mols}",
                                f'--config.eval.batch_size={args.max_batch_size}', 
                                f'--config.sampling.steps={args.sampling_steps}',
                            ]
    sys.argv = argv_new

    return args


FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True
)
flags.DEFINE_string('workdir', None, 'Work directory.')
flags.DEFINE_enum('mode', None, ['train', 'eval'],
                  'Running mode: train or eval')
flags.DEFINE_string('eval_folder', 'eval', 'The folder name for storing evaluation results')
flags.DEFINE_boolean('deterministic', True, 'Set random seed for reproducibility.')
flags.mark_flags_as_required(['workdir', 'config', 'mode'])


def main(argv, args=None):
    # Set random seed
    if FLAGS.deterministic:
        run_lib.set_random_seed(FLAGS.config)

    # Ignore info output by RDKit
    RDLogger.DisableLog('rdApp.error')
    RDLogger.DisableLog('rdApp.warning')


    rdkit_mols, sampling_time = evaluate(FLAGS.config, FLAGS.workdir, FLAGS.eval_folder, args)

    with open(args.output_file, 'wb') as f:
        pickle.dump((rdkit_mols, sampling_time), f)



if __name__ == '__main__':
    args = parse_args()
    main_w_args = partial(main, args=args)
    app.run(main_w_args)
