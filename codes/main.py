# -*- coding: utf-8 -*-

"""
Main script to run an experiment (train and evaluate) form command line.
Requires a yml config file with the details of the experiment and the data
preprocessed to graphs.
From albertoembru/StackGAN
"""

import argparse
import pprint
import yaml

from siamese_graph_trainer import run_several_experiments


def parse_args():
    parser = argparse.ArgumentParser(description='Graph-Based Siamese Network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='non-optional config file',
                        default='GBSN_test.yml', type=str, required=True)
    args = parser.parse_args()
    return args


def cfg_from_file(filename):
    """Load a config file"""

    with open(filename, 'r') as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg = cfg_from_file(args.cfg_file)
        
    print('Using config:')
    pprint.pprint(cfg)

    # Define our variables
    # ========== General options
    device = cfg['device']
    dataset_name = cfg['dataset_name']
    folder_sufix = cfg['folder_sufix']
    doc_dict_folder_prefix = cfg['doc_dict_folder_prefix']
    # doc_dict_folder = (cfg['doc_dict_folder_prefix'] + cfg['dataset_name'] +
    #                    cfg['folder_sufix'])
    ds_list_folder_prefix = cfg['ds_list_folder_prefix']
    # ds_list_folder = cfg['ds_list_folder_prefix'] + cfg['dataset_name']
    dest_folder_prefix = cfg['dest_folder_prefix']
    # dest_folder_base = cfg['dest_folder_prefix'] + cfg['dataset_name']
    epochs = cfg['epochs']
    checkpoint_freq = cfg['checkpoint_freq']
    lim = cfg['lim'] 
    bm_free_epochs = cfg['bm_free_epochs']
    lr = cfg['lr']
    batch_size = cfg['batch_size']
    num_workers = cfg['num_workers']
    # ========== Datasets options
    ds_op = cfg['ds_op']
    # ========== Model options
    exp_ops = cfg['exp_ops']
    repeat_experiment = cfg['repeat_experiment']
    # ========== Loss options
    main_loss = cfg['main_loss']

    # Call run_several_experiments
    run_several_experiments(dataset_name, folder_sufix, dest_folder_prefix,
                    ds_op, exp_ops, repeat_experiment,
                    doc_dict_folder_prefix, ds_list_folder_prefix, lim,
                    batch_size, num_workers,
                    epochs, device, main_loss,
                    bm_free_epochs,
                    checkpoint_freq, lr)

    # Train our model
#     if cfg.TRAIN.FLAG:

#     else:
