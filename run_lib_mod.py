import os
import torch
import logging
import numpy as np
import random
import pickle
import time
import rdkit.Chem as Chem

from datasets import get_dataset, inf_iterator, get_dataloader
from models.ema import ExponentialMovingAverage
import losses
from utils import *
from evaluation import *
import visualize
from models import *
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn, get_cond_sampling_eval_fn, get_cond_multi_sampling_eval_fn
from cond_gen import *
from rdkit.Geometry import Point3D

bond_list = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE,
             Chem.rdchem.BondType.AROMATIC]
stability_bonds = {Chem.rdchem.BondType.SINGLE: 1, Chem.rdchem.BondType.DOUBLE: 2, Chem.rdchem.BondType.TRIPLE: 3,
                   Chem.rdchem.BondType.AROMATIC: 1.5}


def build_rd_mol(positions, atom_types, formal_charges, edge_types, dataset_info):
    """Convert the generated tensors to rdkit mols and check stability"""
    dataset_name = dataset_info['name']
    atom_decoder = dataset_info['atom_decoder']
    if 'atom_fc_num' in dataset_info:
        atom_fcs = dataset_info['atom_fc_num']
    else:
        atom_fcs = {}
    atom_num = atom_types.size(0)

    # convert to rdkit mol
    # atoms
    mol = Chem.RWMol()
    for atom in atom_types:
        a = Chem.Atom(atom_decoder[atom.item()])
        mol.AddAtom(a)

    # add formal charges to Mol
    if formal_charges.shape[-1] == 0:
        formal_charges = torch.zeros_like(atom_types)

    for atom_id, fc in enumerate(formal_charges):
        atom = mol.GetAtomWithIdx(atom_id)
        atom_str = atom.GetSymbol()
        if fc != 0:
            atom_fc = atom_str + str(fc.item())
            # print('formal charges: ' + atom_fc)
            if atom_fc in atom_fcs:
                # print('add formal charges, ' + atom_fc)
                atom.SetFormalCharge(fc.item())

    # add bonds
    edge_index = torch.nonzero(edge_types)
    for i in range(edge_index.size(0)):
        src, dst = edge_index[i]
        if src < dst:
            order = edge_types[src, dst]
            mol.AddBond(src.item(), dst.item(), bond_list[int(order)])

    try:
        mol = mol.GetMol()
    except Chem.KekulizeException:
        return None

    # add positions to Mol
    if positions is not None:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, Point3D(positions[i][0].item(), positions[i][1].item(), positions[i][2].item()))
        mol.AddConformer(conf)

    return mol

def evaluate(config, workdir, eval_folder="eval", args=None):
    """Runs the evaluation pipeline with VPSDE for graphs or point clouds"""
    # Create directory to eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    # Build dataset
    train_ds, _, test_ds, dataset_info = get_dataset(config, transform=False)

    # Initialize model
    nodes_dist = get_node_dist(dataset_info)
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 2 ** 20
    logging.info('model size: {:.1f}MB'.format(model_size))

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(config.sde.schedule, continuous_beta_0=config.sde.continuous_beta_0,
                                      continuous_beta_1=config.sde.continuous_beta_1)

    # Obtain data scaler and inverse scaler
    # scaler = get_data_scaler(config)
    inverse_scaler = get_data_inverse_scaler(config)

    # Checkpoint name
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    ckpts = config.eval.ckpts
    if ckpts != '':
        ckpts = ckpts.split(',')
        ckpts = [int(ckpt) for ckpt in ckpts]
    else:
        ckpts = [_ for _ in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1)]

    # Build sampling functions
    if config.eval.enable_sampling:
        sampling_fn = get_sampling_fn(config, noise_scheduler, nodes_dist, config.eval.batch_size,
                                      config.eval.num_samples, inverse_scaler)

    # Obtain train dataset and eval dataset
    # train_mols = [train_ds[i].rdmol for i in range(len(train_ds))]
    # test_mols = [test_ds[i].rdmol for i in range(len(test_ds))]

    # Build evaluation metrics
    # EDM_metric = get_edm_metric(dataset_info, train_mols)
    # EDM_metric_2D = get_2D_edm_metric(dataset_info, train_mols)
    # mose_metric = get_moses_metrics(test_mols, n_jobs=32, device=config.device)

    # Begin evaluation
    for ckpt in ckpts:
        ckpt_path = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt))
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError("Checkpoint path error: " + ckpt_path)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
        ema.copy_to(model.parameters())

        start = time.time()
        processed_mols = sampling_fn(model)
        rdkit_mols = []
        for m in processed_mols:
            pos, atom_types, edge_types, fc = m
            rdkit_mols.append(build_rd_mol(pos, atom_types, fc, edge_types, dataset_info))
        end = time.time()

        return rdkit_mols, end-start
