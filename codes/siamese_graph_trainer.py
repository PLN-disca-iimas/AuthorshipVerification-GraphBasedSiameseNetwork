# -*- coding: utf-8 -*-

"""Module to train and evaluate GBSN.

Oriented to perform experiments with different parameters, record logs,
generate plots and obtain metrics
"""

__author__ = '{Daniel Embarcadero-Ruiz}'

import os
import sys
import time
import glob
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from tabulate import tabulate
import warnings

import torch
from torch.utils.tensorboard import SummaryWriter
# import torch_geometric.nn as gnn
from torch_geometric.data import DataLoader
from typing import Tuple

from common_func_tesis import (time_string, save_obj, load_obj, print_time,
                               my_print, get_gpu_memory_device)
from datasets_PAN import fit_dict
from pan20_verif_evaluator import auc, c_at_1, f1, f_05_u_score, brier_score
from siamese_graph import (GraphSiameseDatasetDictJoin,
                           # GBFeatures, TextFeatures,
                           # SiameseNetwork,
                           # GlobalAttentionSelect,
                           sparse_encoded_to_torch,
                           _default_type, _class_dict,
                           define_model, load_checkpoint, save_checkpoint)

# For using INAOE server
# import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="7"

SEED = 0
torch.manual_seed(SEED)
torch.set_default_tensor_type(_default_type)

_tsfm_dict = {'graph-pos': sparse_encoded_to_torch,
              # 'graph-pos-embed': sparse_pos_embed_to_torch, # Compatibility
              'text_feat': None}


# ============================== Training scheme ==========

# ==================== Auxiliar functions ==========

def cast(tensor):
    """To cast tensor to float when we need.

    Wrote in these way to easily change to double in experiments"""

    return tensor.float()


def to_device(posible_tuple, device):
    """To send a tuple of tensors or a single tensor to some device"""

    try:
        return posible_tuple.to(device)
    except(AttributeError):
        return [p.to(device) for p in posible_tuple]


# ==================== Scores ==========

def pan21_scores(true, pred, silent_warnings=False):
    """Function to wrap PAN21 scores used"""

    # context manager to change warnings just in these function
    with warnings.catch_warnings():
        if silent_warnings:
            warnings.simplefilter('ignore')

        roc_auc = auc(true, pred)
        f1_score = f1(true, pred)
        c_at_1_score = c_at_1(true, pred)
        f_05 = f_05_u_score(true, pred)
        brier = brier_score(true, pred)
        average = sum([roc_auc, f1_score, c_at_1_score, f_05, brier]) / 5
        score = {'roc_auc': roc_auc,
                 'f1_score': f1_score,
                 'c_at_1': c_at_1_score,
                 'f_05': f_05,
                 'brier': brier,
                 'average': average}

    return score


# ==================== Losses ==========
# Several loss functions tested to improve performance

def soft_f1_loss(y_lgts, true):
    """Soft f1 loss. Based in the f1 score.
    Ideas taken from:
    https://towardsdatascience.com/
    the-unknown-benefits-of-using-a-soft-f1-loss-in-classification-systems
    -753902c0105d
    https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
    """

    pred = y_lgts.sigmoid()
    tp = torch.sum(torch.mul(true, pred))
    fp = torch.sum(torch.mul(1 - true, pred))
    fn = torch.sum(torch.mul(true, 1 - pred))
    soft_f1 = (2 * tp) / (2 * tp + fn + fp + 1e-16)
    return 1 - soft_f1


def double_soft_f1_loss(y_lgts, true):
    """Simetric form of soft f1 with respect of positive and negative problems
    """

    pred = y_lgts.sigmoid()
    tp = torch.sum(torch.mul(true, pred))
    fp_plus_fn = (torch.sum(torch.mul(1 - true, pred)) +
                  torch.sum((torch.mul(true, 1 - pred))))
    tn = torch.sum(torch.mul(1 - true, 1 - pred))
    soft_f1_class1 = (2*tp) / (2*tp + fp_plus_fn + 1e-16)
    soft_f1_class0 = (2*tn) / (2*tn + fp_plus_fn + 1e-16)
    double_soft_f1 = 0.5 * (soft_f1_class1 + soft_f1_class0)
    return 1 - double_soft_f1


def hinge_loss(pred, true_bis):
    res = torch.clamp(1 - (true_bis*pred), min=0)
    return res


def bound_f1_loss(y_lgts, true):
    """Bound for the f1 score using hinge loss.
    Ideas taken from ...
    """
    # TODO: Doc.

    pred = y_lgts.sigmoid()
    # true tensor in targets -1, 1
    true_bis = 2*true - 1

    tp_l = torch.sum(true*(1 - hinge_loss(pred, true_bis)))
    fp_u = torch.sum((1 - true)*(hinge_loss(pred, true_bis)))
    y_pos = torch.sum(true)
    bound_f1 = (2 * tp_l) / (y_pos + tp_l + fp_u + 1e-16)
    return 1 - bound_f1


class meanLoss():
    """Class to define a new loss as a weighted sum of two different losses"""

    def __init__(self, loss_1, loss_2, weight_1=0.5, weight_2=0.5):
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.weight_1 = weight_1
        self.weight_2 = weight_2

    def __call__(self, y_lgts, true):
        l1 = self.loss_1(y_lgts, true)
        l2 = self.loss_2(y_lgts, true)
        return self.weight_1 * l1 + self.weight_2 * l2

    def __str__(self):
        return ('meanLoss_' + str(self.loss_1) + '_' +
                str(self.loss_2) + '_' + str(self.weight_1) + '-' +
                str(self.weight_2))


def define_loss_options(main_loss, with_loss_fn_aux=True):
    """Define loss object and optionally a dict of auxiliar loss objectsrgs:

    Function used to define:
    loss_fn: Main loss function used to train
    loss_fn_aux_dict: Dict of auxiliar losses plotted to explore how them works
    """

    # ========== Loss options (Predefined)
    loss_ops = {'BCE': torch.nn.BCEWithLogitsLoss(),
                'soft_f1_loss': soft_f1_loss,
                'double_soft_f1_loss': double_soft_f1_loss,
                'bound_f1_loss': bound_f1_loss,
                'mean_BCE_sf1': meanLoss(torch.nn.BCEWithLogitsLoss(),
                                         soft_f1_loss),
                'mean_BCE_dsf1': meanLoss(torch.nn.BCEWithLogitsLoss(),
                                          double_soft_f1_loss),
                'mean_BCE_bf1': meanLoss(torch.nn.BCEWithLogitsLoss(),
                                         bound_f1_loss),
                'mean_25_BCE_dsf1': meanLoss(torch.nn.BCEWithLogitsLoss(),
                                             double_soft_f1_loss,
                                             0.25, 0.75),
                'mean_10_BCE_dsf1': meanLoss(torch.nn.BCEWithLogitsLoss(),
                                             double_soft_f1_loss,
                                             0.10, 0.90)}
    loss_fn = loss_ops[main_loss]
    if with_loss_fn_aux:
        loss_fn_aux_dict = loss_ops
        del loss_fn_aux_dict['mean_BCE_sf1']
        del loss_fn_aux_dict['mean_BCE_dsf1']
        del loss_fn_aux_dict['mean_BCE_bf1']
        del loss_fn_aux_dict['mean_25_BCE_dsf1']
        del loss_fn_aux_dict['mean_10_BCE_dsf1']
        if main_loss in loss_fn_aux_dict.keys():
            del loss_fn_aux_dict[main_loss]
    else:
        loss_fn_aux_dict = None

    return loss_fn, loss_fn_aux_dict


# ==================== Load data ==========
# The persisted files with the graph representations can be huge (about 40Gb
# for the large dataset full graphs.
# The GBSN ensemble need to load files from different graph representations


def load_dictionary_parts(dict_path):
    """To load several files within same folder in a single dictionary"""

    list_of_paths = glob.glob(dict_path + '*')
    if len(list_of_paths) == 0:
        print('No se encontró archivo para ', dict_path)
        return None

    print('Loading from:', list_of_paths)
    whole_dict = dict()
    for path in list_of_paths:
        part = load_obj(path, fast=True)
        whole_dict.update(part)

    return whole_dict


def define_doc_dict(doc_dict_folder, data_type):
    """To read from disk the dictionary of some type of data.

    First the graph data was persisted in a single file and after the graph
    data was persisted separate in adjacence data and node feature data.
    """

    if data_type in ['short', 'med', 'full']:
        graph_pos = os.path.join(doc_dict_folder,
                                 ('sparse_encoded_dict__' + data_type))
        doc_dict = load_dictionary_parts(graph_pos)
        if doc_dict is None:
            graphs = os.path.join(doc_dict_folder,
                                  'sparse_dict_' + data_type)
            pos = os.path.join(doc_dict_folder,
                               'pos_encoded_dict_' + data_type)
            graph_dict = load_dictionary_parts(graphs)
            pos_dict = load_dictionary_parts(pos)
            doc_dict = {key: (graph_dict[key], pos_dict[key])
                        for key in graph_dict.keys()}
            del graph_dict
            del pos_dict

        data_available = 'graph-pos'
    else:
        text_feat = os.path.join(doc_dict_folder, 'text_feat_dict_')
        doc_dict = load_dictionary_parts(text_feat)
        data_available = 'text_feat'

    return doc_dict, data_available


def define_ds_dl_join(doc_dict_folder, data_type_list, ds_list_folder, lim,
                      batch_size, num_workers, f=sys.stdout):
    """To define dataset and dataloader for train, val and test split.

    First: Read data from disk
        - Load the list of problems. ds_list_ is the list of the problems, each
        problem has two text_id.
        - Load the dictionary of data (can be dictionary of graphs or
        dictionary of stylistic data). Is important just create one dictionary
        even if we use more than one time some data. For example if the GBSN
        ensemble uses two full graph components we just want one full graph
        dictionary in memory.
    Then define dataset using the class defined in siamese_graph
    Then define dataloader objects with the pytorch geometric class
    """

    # ========== Load ds_lists
    ds_list_train = load_obj(os.path.join(ds_list_folder,
                                          'ds_list_train_n'), fast=True)
    ds_list_val = load_obj(os.path.join(ds_list_folder,
                                        'ds_list_val_n'), fast=True)
    ds_list_test = load_obj(os.path.join(ds_list_folder,
                                         'ds_list_test_n'), fast=True)
    # ========== Load doc_dicts
    start_time_l = time.time()
    doc_dicts_dict = \
        {data_type: define_doc_dict(doc_dict_folder, data_type) for
         data_type in set(data_type_list)}
    list_of_dicts_dict = dict()
    for data_type, (doc_dict, data_available) in doc_dicts_dict.items():
        list_of_dicts_dict[data_type] = \
            (fit_dict(doc_dict, [ds_list_train, ds_list_val, ds_list_test]),
             data_available)
        del doc_dict

    del doc_dicts_dict
    print_time(start_time_l, 'Load doc_dict', f)
    # ========== Define datasets
    list_of_dicts_list, data_available_list = \
        zip(*[list_of_dicts_dict[data_type] for data_type in
              data_type_list])
    ds_train = \
        GraphSiameseDatasetDictJoin(ds_list_train,
                                    [lod[0] for lod in
                                     list_of_dicts_list],
                                    [_tsfm_dict[da] for da in
                                     data_available_list],
                                    lim=lim)
    ds_val = \
        GraphSiameseDatasetDictJoin(ds_list_val,
                                    [lod[1] for lod in
                                     list_of_dicts_list],
                                    [_tsfm_dict[da] for da in
                                     data_available_list],
                                    lim=lim)
    ds_test = \
        GraphSiameseDatasetDictJoin(ds_list_test,
                                    [lod[2] for lod in
                                     list_of_dicts_list],
                                    [_tsfm_dict[da] for da in
                                     data_available_list],
                                    lim=lim)

    print(f"Number of training graphs: {len(ds_train)}", file=f)
    print(f'Number of val graphs: {len(ds_val)}', file=f)
    print(f'Number of test graphs: {len(ds_test)}', file=f)
    # ========== Define dataloaders
    dl_train = DataLoader(ds_train, batch_size=batch_size,
                          shuffle=True, follow_batch=['x_s', 'x_t'],
                          num_workers=num_workers)
    dl_val = DataLoader(ds_val, batch_size=batch_size,
                        shuffle=False, follow_batch=['x_s', 'x_t'],
                        num_workers=num_workers)
    dl_test = DataLoader(ds_test, batch_size=batch_size,
                         shuffle=False, follow_batch=['x_s', 'x_t'],
                         num_workers=num_workers)

    print_time(start_time_l, 'Load all', f)
    return dl_train, dl_val, dl_test


# ==================== Plot metrics ==========

def print_metrics_model(metrics, model_label, losses_val, scores_val,
                        losses_aux_val, f=sys.stdout, epoch_init=0):
    """To print scores of a model with good look"""

    if model_label + '_epoch' in metrics.keys():
        epoch = metrics[model_label + '_epoch']
        epoch_ind = epoch - epoch_init
        print("\n ===== " + model_label + f":\nEpoch: {epoch}", file=f)
        model_loss = metrics[model_label + '_loss']
        model_score = metrics[model_label + '_score']
        model_stats = \
            [['main loss', losses_val[epoch_ind], model_loss],
             ['average', scores_val[epoch_ind]['average'],
              model_score['average']],
             ['roc_auc', scores_val[epoch_ind]['roc_auc'],
              model_score['roc_auc']],
             ['f1_score', scores_val[epoch_ind]['f1_score'],
              model_score['f1_score']],
             ['c_at_1', scores_val[epoch_ind]['c_at_1'],
              model_score['c_at_1']],
             ['f_05', scores_val[epoch_ind]['f_05'], model_score['f_05']],
             ['brier', scores_val[epoch_ind]['brier'], model_score['brier']]]
        if losses_aux_val is not None:
            model_loss_aux = metrics[model_label + '_loss_aux']
            model_stats.extend(
               [[k, losses_aux_val[epoch_ind][k], model_loss_aux[k]]
                for k in losses_aux_val[0].keys()])

        print(tabulate(model_stats,
                       headers=['val', 'test']), file=f)
        return epoch
    else:
        print("\n ===== " + model_label + ' no found', file=f)
        return epoch_init


def plot_metrics(losses_train, losses_val, losses_aux_val,
                 scores_val, bm_epoch, es_epoch,
                 run_name, dest_folder, epoch_init=0, title=True):
    """To plot losses and scores along the epochs trained"""

    # parameters
    fontsize = 10

    epochs = len(losses_train)
    epochs_range = range(epoch_init, epochs + epoch_init)
    bm_epoch_ind = bm_epoch - epoch_init
    if es_epoch is not None:
        es_epoch_ind = es_epoch - epoch_init

    # ===== losses
    plt.plot(epochs_range, losses_train, '-b', lw=2, label='loss train')
    plt.plot(epochs_range, losses_val, '-g', lw=2, label='loss val')
    colors_dict = {'BCE': 'darkgreen',
                   'soft_f1_loss': 'mediumseagreen',
                   'double_soft_f1_loss': 'limegreen',
                   'bound_f1_loss': 'seagreen',
                   'mean_BCE_sf1': 'gold',
                   'mean_BCE_dsf1': 'goldenrod',
                   'mean_BCE_bf1': 'darkgoldenrod'}
    if losses_aux_val is not None:
        for k in losses_aux_val[0].keys():
            aux_list = [score[k] for score in losses_aux_val]
            if k in colors_dict.keys():
                c = colors_dict[k]
            else:
                c = 'crimson'

            plt.plot(epochs_range, aux_list, color=c, label=k)

    plt.legend(loc='lower left', fontsize=fontsize)
    # best model marks
    plt.vlines([bm_epoch], 0, losses_val[bm_epoch_ind],
               colors='k', lw=1)
    plt.hlines(losses_val[bm_epoch_ind], epochs_range[0], epochs_range[-1],
               colors='k', lw=1)
    # early stop marks
    if es_epoch is not None:
        plt.vlines([es_epoch], 0, losses_val[es_epoch_ind],
                   colors='k', lw=1, ls='dashed')
        plt.hlines(losses_val[es_epoch_ind], epochs_range[0], epochs_range[-1],
                   colors='k', lw=1, ls='dashed')

    plt.autoscale()
    plt.ylim([0, 0.8])
    plot_name = run_name + '_loss'
    if title:
        plt.title(plot_name)

    plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
    plt.show()
    plt.clf()
    # ===== metrics
    # unpack scores val
    average_val = [score['average'] for score in scores_val]
    roc_auc_val = [score['roc_auc'] for score in scores_val]
    f1_score_val = [score['f1_score'] for score in scores_val]
    c_at_1_val = [score['c_at_1'] for score in scores_val]
    f_05_val = [score['f_05'] for score in scores_val]
    brier_val = [score['brier'] for score in scores_val]

    plt.plot(epochs_range, average_val, color='darkred', ls='solid', lw=2.5,
             label='average')
    plt.plot(epochs_range, roc_auc_val, '--r', label='roc_auc')
    plt.plot(epochs_range, f1_score_val, '--c', label='f1')
    plt.plot(epochs_range, c_at_1_val, '--m', label='c@1')
    plt.plot(epochs_range, f_05_val, '--y', label='f_05')
    plt.plot(epochs_range, brier_val, color='fuchsia', ls='dashed',
             label='brier')
    plt.legend(loc='lower right', fontsize=fontsize)
    # best model marks
    plt.vlines([bm_epoch], 0, average_val[bm_epoch_ind],
               colors='k', lw=1)
    plt.hlines(average_val[bm_epoch_ind], epochs_range[0], epochs_range[-1],
               colors='k', lw=1)
    # early stop marks
    if es_epoch is not None:
        plt.vlines([es_epoch], 0, average_val[es_epoch_ind],
                   colors='k', lw=1, ls='dashed')
        plt.hlines(average_val[es_epoch_ind], epochs_range[0],
                   epochs_range[-1], colors='k', lw=1, ls='dashed')

    # # Plot with y axis from 0 to 1
    # plt.autoscale()
    # plt.ylim([0, 1])
    # plt.tight_layout()
    # plot_name = run_name + '_scores'
    # plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
    # if title:
    #     plt.title(plot_name)

    # # Plot with y axis from 0.5 to 1
    # plt.ylim([0.5, 1])
    # plt.tight_layout()
    # plot_name = run_name + '_scores-2'
    # plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
    # if title:
    #     plt.title(plot_name)

    # Plot with y axis from 0.75 to 1
    plt.ylim([0.75, 1])
    plt.tight_layout()
    plot_name = run_name + '_scores-3'
    plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
    if title:
        plt.title(plot_name)

    plt.clf()


def print_and_plot_metrics(metrics, dest_folder, f, run_name, epoch_init=0):
    """Main function to plot metrics of the best models and last model

    First: print average metrics for:
        - Best model with respect of loss in val split,
        - Best model with respect of average metrics in val split (NO USED
    IN FINAL MODEL, JUST TO EXPLORE IN DEVELOPMENT)
        - The model in the last epoch.
    Second: Plot how losses and metrics change across epochs."""

    # ========== Load metrics
    losses_train = metrics['losses_train']
    losses_val = metrics['losses_val']
    scores_val = metrics['scores_val']
    if ('losses_aux_val' in metrics.keys() and
            metrics['losses_aux_val'][0] is not None):
        losses_aux_val = metrics['losses_aux_val']
    else:
        losses_aux_val = None

    # ========== Print stats
    bm_epoch = \
        print_metrics_model(metrics, 'best_model', losses_val, scores_val,
                            losses_aux_val, f, epoch_init)
    print_metrics_model(metrics, 'best_model_sa',
                        losses_val, scores_val,
                        losses_aux_val, f, epoch_init)
    print_metrics_model(metrics, 'last_model',
                        losses_val, scores_val,
                        losses_aux_val, f, epoch_init)

    if 'early_stop_model_epoch' in metrics.keys():
        es_epoch = metrics['early_stop_model_epoch']
        print(f"\n ===== Early stop model:\nEpoch: {es_epoch}", file=f)
        if 'early_stop_model_loss' in metrics.keys():
            print('pendiente', file=f)
        else:
            print('Early stop model deactivated', file=f)
    else:
        print("\n===== Early stop model:\nEpoch: NA", file=f)
        es_epoch = len(losses_val) - 1 + epoch_init

    average_val = [score['average'] for score in scores_val]
    max_avg = max(average_val)
    print(f'\n===== Max average in val: {max_avg:.6} '
          f'epoch {average_val.index(max_avg)}',
          file=f)

    # ========== Plots
    plot_metrics(losses_train, losses_val, losses_aux_val,
                 scores_val, bm_epoch, es_epoch,
                 run_name, dest_folder, epoch_init)


# ==================== Optimize threshold ==========

class LinearAdjust():
    """To linearly transform outputs. Reference in the paper"""

    def __init__(self, th, margin):
        self.th_i = th - margin
        self.th_f = th + margin
        self.coef_1 = 0.5 / (th - margin)
        self.coef_2 = 0.5 / (1 - th - margin)

    def linear_func(self, x):
        if x < self.th_i:
            return self.coef_1 * x
        elif x > self.th_f:
            return min([self.coef_2 * x + 1 - self.coef_2, 1])
        else:
            return 0.5

    def apply(self, pred):
        return np.array([self.linear_func(xi) for xi in pred])


def define_scores_grid(true, pred, fine_grid=False):
    """Auxiliar in optimize_threshold. Define the grid of apply LinearAdjust
    to the predictions"""

    # Define threshold grid
    if fine_grid:
        th_ops = np.arange(0.05, 1, 0.01)
        margin_ops = np.arange(0, 0.5, 0.01)
    else:
        th_ops = np.arange(0.05, 1, 0.05)
        margin_ops = np.arange(0, 0.3, 0.05)

    res_dict = dict()
    for th in th_ops:
        margin_dict = dict()
        for margin in margin_ops:
            if th - margin > 0 and th + margin < 1:
                pred_adjust = LinearAdjust(th, margin).apply(pred)
                # Call scores suppressing warnings
                res = pan21_scores(true, pred_adjust, silent_warnings=True)
                margin_dict[f'{margin:.2f}'] = res
            else:
                res = {'roc_auc': 0, 'f1_score': 0, 'c_at_1': 0, 'f_05': 0,
                       'brier': 0, 'average': 0}
                margin_dict[f'{margin:.2f}'] = res

        res_dict[f'{th:.2f}'] = margin_dict

    df = pd.DataFrame(res_dict)
    df_dict = dict()
    keys = ['roc_auc', 'f1_score', 'c_at_1', 'f_05', 'brier', 'average']
    for k in keys:
        df_dict[k] = df.applymap(lambda x: x[k]).transpose()

    return df_dict


def define_best_threshold(true, pred, fine_grid=False):
    """Optimize threshold by a search grid over the scores"""

    # Define dataframe of scores
    df_dict = define_scores_grid(true, pred, fine_grid)
    # Define best th and margin
    df_average = df_dict['average']
    max_avg = df_average.max().max()
    best_ops = [(df_average.index[x], df_average.columns[y])
                for x, y in zip(*np.where(df_average.values == max_avg))]
    # Select best option
    if len(best_ops) > 1:
        min_margin = min([float(best_op[1]) for best_op in best_ops])
        best_op = [best_op for best_op in best_ops
                   if float(best_op[1]) == min_margin][0]
    else:
        best_op = best_ops[0]

    return best_op, df_dict


def plot_as_heatmap(df_dict, dest_folder, plot_label, fine_grid=False,
                    title=True):
    if fine_grid:
        args_dict = {'vmin': 0.5,
                     'vmax': 1}
    else:
        args_dict = {'annot': True,
                     'fmt': ".4f",
                     'vmin': 0.8,
                     'vmax': 0.95,
                     'annot_kws': {"size": 7}}

    args_part_dict = {'roc_auc': {'cmap': 'Blues'},
                      'brier': {'cmap': 'Blues'},
                      'f1_score': {'cmap': 'BuPu'},
                      'c_at_1': {'cmap': 'BuPu'},
                      'f_05': {'cmap': 'BuPu'},
                      'average': {'cmap': 'Greens'}}
    for k in df_dict.keys():
        sns.heatmap(df_dict[k], **args_part_dict[k], **args_dict)
        plot_name = plot_label + '_' + k
        if title:
            plt.title(plot_name)

        plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
        plt.clf()


def plot_predictions(pred, true, dest_folder, plot_name, title=True):
    cat = np.stack([pred, true]).transpose()
    df = pd.DataFrame(cat, columns=['prediction', 'true'])
    sns.histplot(data=df, x='prediction', hue='true', bins=50,
                 multiple='stack')
    if title:
        plt.title(plot_name)

    plt.savefig(os.path.join(str(dest_folder), plot_name) + '.png')
    plt.clf()


def optimize_threshold(metrics, model_label, dest_folder, run_name, f,
                       fine_grid=False):
    """Find best thershold in val, plot heatmaps and scores in test split

    Fist find the best threshold looking at the scores in val split and plot
    heatmap.
    Then find the best threshold looking at the scores in test split and plot
    heatmap. The prediction never use these information, only relevant during
    development.
    Then print best results and plot distribution before and after optimize.
    Finally persist results."""

    if model_label + '_true_val' not in metrics.keys():
        print(f'\n===== Optimize threshold: no found {model_label}:', file=f)
    else:
        true_val = metrics[model_label + '_true_val']
        pred_val = metrics[model_label + '_pred_val']
        true_test = metrics[model_label + '_true']
        pred_test = metrics[model_label + '_pred']
        bm_score = metrics[model_label + '_score']
        # ========== Adjust threshold in val set
        best_op_val, df_dict_val = \
            define_best_threshold(true_val, pred_val, fine_grid)
        plot_as_heatmap(df_dict_val, dest_folder,
                        run_name + '_' + model_label + '_th-val',
                        fine_grid)
        # ========== test threshold in test set
        best_op_test, df_dict_test = \
            define_best_threshold(true_test, pred_test, fine_grid)
        plot_as_heatmap(df_dict_test, dest_folder,
                        run_name + '_' + model_label + '_th-test',
                        fine_grid)
        # Define best th and best margin
        th_best = best_op_val[0]
        margin_best = best_op_val[1]

        print(f'\n===== Optimize threshold for {model_label}:', file=f)
        print('Test: Original', file=f)
        print(bm_score['average'], file=f)

        print('Val: Best th posible', file=f)
        print(best_op_val, file=f)
        print(df_dict_val['average'][margin_best][th_best], file=f)
        print('Test: Optimized', file=f)
        print(df_dict_test['average'][margin_best][th_best], file=f)
        print('Test: Best th posible', file=f)
        print(best_op_test, file=f)
        print(df_dict_test['average'][best_op_test[1]][best_op_test[0]],
              file=f)

        # Plot predictions
        plot_predictions(pred_test, true_test, dest_folder,
                         run_name + '_' + model_label + 'pred_test_raw')
        pred_adjust = \
            LinearAdjust(float(th_best), float(margin_best)).apply(pred_test)
        plot_predictions(pred_adjust, true_test, dest_folder,
                         run_name + '_' + model_label +
                         'pred_test_' + str(best_op_val))

        # Save thresholds
        thresholds = {'best_op_val': best_op_val,
                      'df_dict_val': df_dict_val,
                      'best_op_test': best_op_test,
                      'df_dict_test': df_dict_test}
        path = os.path.join(dest_folder,
                            run_name + '_thresholds-' + model_label)
        print(f'\nSaving thresholds for {model_label} in:', path)
        save_obj(thresholds, path)


# ==================== Eval and train ==========

def eval_fn(model, dl, device, loss_fn,
            loss_fn_aux_dict=None, return_pred=False):
    """Evaluate a model over a split"""

    iterator = dl

    loss_sum = 0.0
    instances_count = 0
    pred_list = []
    true_list = []
    with torch.no_grad():
        model.eval()
        if loss_fn_aux_dict is None:
            # Resultado de loss_fn_aux nulo
            loss_aux_avg_dict = None
            for pair, y in tqdm(iterator):
                pair = to_device(pair, device)
                out = model(pair).squeeze()
                del pair
                # convertir etiquetas objetivo
                y2 = cast(y).to(device)
                loss = loss_fn(out, y2)
                # ----- Statistics
                l_y = len(y)
                loss_sum += loss.item() * l_y
                instances_count += l_y
                pred_list.append(out.sigmoid().cpu().numpy())
                true_list.append(np.array(y))

        else:
            # Equivale a loss_sum
            loss_aux_sum_dict = {key: 0.0 for key in loss_fn_aux_dict.keys()}
            for pair, y in tqdm(iterator):
                pair = to_device(pair, device)
                out = model(pair).squeeze()
                del pair
                # convertir etiquetas objetivo
                y2 = cast(y).to(device)
                loss = loss_fn(out, y2)
                # ----- Statistics
                l_y = len(y)
                loss_sum += loss.item() * l_y
                instances_count += l_y
                pred_list.append(out.sigmoid().cpu().numpy())
                true_list.append(np.array(y))
                for (key, item) in loss_fn_aux_dict.items():
                    loss_aux = item(out, y2)
                    loss_aux_sum_dict[key] += loss_aux.item() * l_y

            loss_aux_avg_dict = {k: v / instances_count for
                                 k, v in loss_aux_sum_dict.items()}

        loss_avg = loss_sum / instances_count
        pred = np.concatenate(pred_list)
        true = np.concatenate(true_list)

    score = pan21_scores(true, pred)
    if return_pred:
        return loss_avg, score, loss_aux_avg_dict, pred, true
    else:
        return loss_avg, score, loss_aux_avg_dict


def train_fn(model, opt, loss_fn, epochs, device,
             model_class, model_args,
             loss_fn_aux_dict,
             dl_train, dl_val,
             dest_folder, bm_free_epochs,
             checkpoint_freq=50, epoch_init=0, loss_init=100,
             epochs_to_stop=20):
    """Train a GBSN model along epochs"""

    # ========== Tensorboard
    log_dir = os.path.join(dest_folder, 'runs')
    writer = SummaryWriter(log_dir)
    # tb = program.TensorBoard()
    # tb.configure(argv=[None, '--logdir', log_dir])
    # url = tb.launch()

    gpu_usage = get_gpu_memory_device()
    # historial de pérdida
    losses_train = []
    losses_val = []
    scores_val = []
    losses_aux_val = []

    # pseudo paro
    bad_epochs = 0
    best_loss = loss_init
    best_score_avg = 0.5
    best_model = {'epoch': epoch_init}
    early_stop_flag = True
    early_stop_model = None
    print('Start training')
    print(f'Initial epoch: {epoch_init}')
    for e_count in trange(epochs):
        epoch = epoch_init + e_count
        # --------------- Train epoch
        loss_sum = 0.0
        instances_count = 0
        model.train()
        for pair, y in tqdm(dl_train):
            # ----- Forward
            pair = to_device(pair, device)
            out = model(pair).squeeze()
            del pair
            # convertir etiquetas objetivo
            y = cast(y).to(device)
            loss = loss_fn(out, y)
            del out
            # ----- Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
            # ----- Statistics
            l_y = len(y)
            del y
            loss_sum += loss.item() * l_y
            instances_count += l_y

        gpu_usage = max([gpu_usage,
                         get_gpu_memory_device()])
        loss_avg_train = loss_sum / instances_count
        losses_train.append(loss_avg_train)
        # --------------- Eval epoch
        loss_avg_val, score_val, loss_aux_avg_val = \
            eval_fn(model, dl_val, device, loss_fn,
                    loss_fn_aux_dict=loss_fn_aux_dict)
        gpu_usage = max([gpu_usage,
                         get_gpu_memory_device()])
        losses_val.append(loss_avg_val)
        scores_val.append(score_val)
        losses_aux_val.append(loss_aux_avg_val)
        # --------------- Save to Tensorboard
        writer.add_scalar('loss/train', loss_avg_train, epoch)
        writer.add_scalar('loss/val', loss_avg_val, epoch)
        for key in score_val.keys():
            writer.add_scalar('score_val/' + key, score_val[key], epoch)

        # --------------- Loss: Pseudo paro temprano y best model.
        # A partir época bm_free_epoch
        if e_count >= bm_free_epochs:
            if loss_avg_val < best_loss:
                best_loss = loss_avg_val
                # Save to disk
                best_model = {'epoch': epoch,
                              'model': model,
                              'model_class': model_class,
                              'model_args': model_args,
                              'optimizer': opt,
                              'loss': loss}
                save_checkpoint(best_model, 'best_model', dest_folder,
                                epoch_label='')

                bad_epochs = 0
            else:
                bad_epochs += 1
                if early_stop_flag and bad_epochs == epochs_to_stop:
                    early_stop_flag = False
                    early_stop_model = {'epoch': best_model['epoch']}

        # --------------- Scores average best model
        if score_val['average'] > best_score_avg:
            best_score_avg = score_val['average']
            # Save to disk
            best_model_sa = {'epoch': epoch,
                             'model': model,
                             'model_class': model_class,
                             'model_args': model_args,
                             'optimizer': opt,
                             'loss': loss}
            save_checkpoint(best_model_sa, 'best_model_sa', dest_folder,
                            epoch_label='')

        # --------------- Save checkpoint
        if epoch % checkpoint_freq == 0 and epoch > 0:
            print('Saving checkpoint in epoch', epoch)
            checkpoint_model = {'epoch': epoch,
                                'model': model,
                                'model_class': model_class,
                                'model_args': model_args,
                                'optimizer': opt,
                                'loss': loss}
            save_checkpoint(checkpoint_model, 'checkpoint', dest_folder)

    # Save last checkpoint
    print('Saving checkpoint in LAST epoch', epoch)
    last_model = {'epoch': epoch,
                  'model': model,
                  'model_class': model_class,
                  'model_args': model_args,
                  'optimizer': opt,
                  'loss': loss}
    save_checkpoint(last_model, 'checkpoint_last', dest_folder)

    print("Finished Training")
    metrics = {'losses_train': np.asarray(losses_train),
               'losses_val': np.asarray(losses_val),
               'scores_val': scores_val,
               'losses_aux_val': losses_aux_val}
    return metrics, early_stop_model, gpu_usage


def add_model_metrics(metrics, model_dict, model_label,
                      dl_test, dl_val, dl_train,
                      device, loss_fn, loss_fn_aux_dict, f):
    """Update metrics with the scores in test and val splits."""

    start_time_test = time.time()
    print('Testing ' + model_label)
    print('Getting scores and predictions in test split')
    bm_loss, bm_score, bm_loss_aux, pred, true = \
        eval_fn(model_dict['model'], dl_test, device, loss_fn,
                loss_fn_aux_dict=loss_fn_aux_dict, return_pred=True)
    metrics[model_label + '_loss'] = bm_loss
    metrics[model_label + '_score'] = bm_score
    metrics[model_label + '_loss_aux'] = bm_loss_aux
    metrics[model_label + '_pred'] = pred
    metrics[model_label + '_true'] = true
    print_time(start_time_test, 'Test ' + model_label, f)

    print('Getting scores and predictions in val split')
    bm_loss_val, bm_score_val, bm_loss_aux_val, pred_val, true_val = \
        eval_fn(model_dict['model'], dl_val, device, loss_fn,
                loss_fn_aux_dict=loss_fn_aux_dict, return_pred=True)
    # metrics[model_label + '_loss'] = bm_loss_val
    # metrics[model_label + '_score'] = bm_score_val
    # metrics[model_label + '_loss_aux'] = bm_loss_aux_val
    metrics[model_label + '_pred_val'] = pred_val
    metrics[model_label + '_true_val'] = true_val
    return metrics


def train_scheme(model, opt, loss_fn, epochs, device, loss_fn_aux_dict,
                 dl_train, dl_val, dl_test,
                 dest_folder_exp, run_name, f, model_class, model_args,
                 batch_size, exp_op, bm_free_epochs, checkpoint_freq,
                 epoch_init=0, loss_init=100,
                 sparse_mode=False):
    """To train, evaluate, persist scores and plot metrics for a model."""

    start_time = time.time()
    # ========== Print setup to file
    print('\nSparse mode:', sparse_mode, file=f)
    print('device', device, file=f)
    print('tensor type', _default_type, file=f)
    print('Epochs:', epochs, file=f)
    print('Batch size:', batch_size, file=f)
    print('Starting with exp_op:', file=f)
    my_print(exp_op, file=f)
    print('loss_fn:', loss_fn, file=f)
    print('loss_fn_aux_dict:', file=f)
    my_print(loss_fn_aux_dict, file=f)
    print('\n===== Model:', file=f)
    print('\n', model, file=f)
    print('\n===== Optimizer:', file=f)
    print(opt, file=f)

    print('Sparse mode:', sparse_mode)
    print('device', device)
    print('tensor type', _default_type)
    print('Epochs:', epochs)
    print('Batch size:', batch_size)
    print('Starting with exp_op:')
    my_print(exp_op)
    print('loss_fn:', loss_fn)
    print('loss_fn_aux_dict:')
    my_print(loss_fn_aux_dict)
    print('\n===== Model:')
    print('\n', model)
    print('\n===== Optimizer:')
    print(opt)

    gpu_usage = get_gpu_memory_device()
    # ========== Train model
    start_time_train = time.time()
    metrics, early_stop_model, gpu_usage = \
        train_fn(model, opt, loss_fn, epochs, device,
                 model_class, model_args,
                 loss_fn_aux_dict,
                 dl_train, dl_val,
                 dest_folder_exp, bm_free_epochs, checkpoint_freq,
                 epoch_init, loss_init)
    gpu_usage = max([gpu_usage, get_gpu_memory_device()])
    print_time(start_time_train, 'Train model', f)

    # ========== Test and save models
    # ===== Last model
    last_model_dict = {'epoch': epochs - 1 + epoch_init,
                       'model': model}
    metrics['last_model_epoch'] = last_model_dict['epoch']
    metrics = add_model_metrics(metrics, last_model_dict, 'last_model',
                                dl_test, dl_val, dl_train,
                                device, loss_fn, loss_fn_aux_dict, f)
    gpu_usage = max([gpu_usage, get_gpu_memory_device()])

    # ===== Best model
    best_model_paths = glob.glob(os.path.join(dest_folder_exp,
                                              'best_model_.pth'))
    if len(best_model_paths) == 1:
        best_model_path = best_model_paths[0]
        best_model = copy.deepcopy(model)
        best_model_dict = \
            load_checkpoint(best_model_path, model=best_model, device=device)
        metrics['best_model_epoch'] = best_model_dict['epoch']
        metrics = add_model_metrics(metrics, best_model_dict, 'best_model',
                                    dl_test, dl_val, dl_train,
                                    device, loss_fn, loss_fn_aux_dict, f)
        gpu_usage = max([gpu_usage, get_gpu_memory_device()])
        del best_model_dict
        del best_model
    else:
        print('No found: best_model')

    # ===== Best model scores average
    best_model_sa_paths = glob.glob(os.path.join(dest_folder_exp,
                                                 'best_model_sa_.pth'))
    if len(best_model_sa_paths) == 1:
        best_model_sa_path = best_model_sa_paths[0]
        best_model_sa = copy.deepcopy(model)
        best_model_sa_dict = \
            load_checkpoint(best_model_sa_path, model=best_model_sa,
                            device=device)
        metrics['best_model_sa_epoch'] = best_model_sa_dict['epoch']
        metrics = add_model_metrics(metrics, best_model_sa_dict,
                                    'best_model_sa',
                                    dl_test, dl_val, dl_train,
                                    device, loss_fn, loss_fn_aux_dict, f)
        gpu_usage = max([gpu_usage, get_gpu_memory_device()])
        del best_model_sa_dict
        del best_model_sa
    else:
        print('No found: best_model_sa')

    # ===== Early stop model
    if early_stop_model is not None:
        metrics['early_stop_model_epoch'] = early_stop_model['epoch']
        print('Early stop model deactivated')

    # ========== Save metrics
    metrics_path = os.path.join(dest_folder_exp, run_name + '_metrics')
    print('\nSaving raw metrics in:', metrics_path, file=f)
    save_obj(metrics, metrics_path)
    # ========== Print metrics
    print_and_plot_metrics(metrics, dest_folder_exp, f, run_name, epoch_init)
    # ========== Optimize threshold
    optimize_threshold(metrics, 'best_model',  dest_folder_exp, run_name, f)
    optimize_threshold(metrics, 'best_model_sa',  dest_folder_exp, run_name, f)

    # ========== Print time and gpu_usage
    print('\n==========', file=f)
    print('Max gpu_usage:', gpu_usage, file=f)
    print('Max gpu_usage:', gpu_usage)
    print_time(start_time, 'Total time for ' + run_name, f)
    f.close()
    print('Finish experiment:', run_name)
    print_time(start_time, 'Total time for ' + run_name)


# ============================== Experiments scheme ==========

# TODO: Documentar parámetros
def run_several_experiments(dataset_name, folder_sufix, dest_folder_prefix,
                            ds_op, exp_ops, repeat_experiment,
                            doc_dict_folder_prefix, ds_list_folder_prefix, lim,
                            batch_size, num_workers,
                            epochs, device, main_loss,
                            bm_free_epochs,
                            checkpoint_freq, lr, run_name_base='GBSN'):
    """Function to run several expermients over GBSN

    """

    # Time stamp
    time_stamp = time_string(short=True)

    # Define variables
    doc_dict_folder = (doc_dict_folder_prefix + dataset_name +
                       folder_sufix)
    ds_list_folder = ds_list_folder_prefix + dataset_name
    dest_folder = (dest_folder_prefix + dataset_name + '_' + time_stamp)
    loss_fn, loss_fn_aux_dict = define_loss_options(main_loss)

    # Create folder
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # ========== Unpack parameters dataset
    exp_label = ds_op['exp_label']
    if exp_label != 'ensemble':
        data_type_list = [exp_label]
    else:
        data_type_list = ds_op['data_type_list']

    print('Starting with ds_op : ', exp_label)
    # ========== Setup log file dataset
    run_name = run_name_base + exp_label + '_' + time_stamp
    log_path = os.path.join(dest_folder, run_name + '_log.txt')
    print('log save in: ' + log_path)
    f = open(log_path, 'w+')
    # ========== Define dataset and dataloaders
    dl_train, dl_val, dl_test = \
        define_ds_dl_join(doc_dict_folder, data_type_list,
                          ds_list_folder, lim,
                          batch_size, num_workers, f)
    f.close()

    # ========== train and test different models
    for i, exp_op in enumerate(exp_ops):
        # ========== Unpack parameters model and change to classes
        # default model class is SiameseNetwork
        model_class = _class_dict['SiameseNetwork']
        model_args = exp_op['model_args']
        model_path_list = []
        for i, component in enumerate(model_args['raw_components_list']):
            # If component defines a class and arguments (Mainly GBSN single):
            if isinstance(component, dict):
                assert 'class' in component.keys()
                # Agrega a model_path_list
                model_path_list.append(str(component['class']))
                if component['class'] == 'GBFeatures':
                    # Define objects instead of strings in model_args
                    component['class'] = _class_dict[component['class']]
                    component['args']['conv_type'] = \
                        _class_dict[component['args']['conv_type']]
                    component['args']['pool_type'] = \
                        _class_dict[component['args']['pool_type']]

                elif component['class'] == 'TextFeatures':
                    # Define objects instead of strings in model_args
                    component['class'] = _class_dict[component['class']]

            # If component is a model_path. (For GBSN ensemble)
            elif isinstance(component, str):
                model_path = component
                model_path_list.append(model_path)
                # Transform to a checkpoint
                checkpoint = torch.load(model_path, map_location='cpu')
                model_args['raw_components_list'][i] = checkpoint

            # Solo conservamos checkpoint_list exp_op['model_args']
            # model_args = {k: v for (k, v) in model_args.items()
            #                         if k != 'checkpoint_list'}

        for j in range(repeat_experiment):
            print('Starting with repetition ', str(j))
            # ========== Change dest folder
            run_id = exp_label + '_' + str(i) + '-' + str(j)
            dest_folder_exp = os.path.join(dest_folder, run_id)
            # =============== Start experiment ==============
            if not os.path.exists(dest_folder_exp):
                os.makedirs(dest_folder_exp)

            # ========== Setup log file experiment
            run_name = run_name_base + time_stamp + '_' + run_id
            log_path = os.path.join(dest_folder_exp, run_name + '_log.txt')
            print('log save in: ' + log_path)
            f = open(log_path, 'w+')
            print('ds_op: ', ds_op, file=f)
            print('experiment repetition: ', j, ' of ', repeat_experiment,
                  file=f)
            # print('model_patn_list: ', model_path_list, file=f)
            print('data_type_list: ', data_type_list, file=f)

            # ========== Define model
            model = define_model(model_class, model_args)
            # optimizer
            model = model.to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            # ========== Train, test, plot and save metrics
            train_scheme(model, opt, loss_fn, epochs, device,
                         loss_fn_aux_dict, dl_train, dl_val, dl_test,
                         dest_folder_exp, run_name, f, model_class,
                         model_args, batch_size, exp_op, bm_free_epochs,
                         checkpoint_freq)


# ============================== New to apply the model ==========

class GBSN_linear_adjust:
    def __init__(self, model_path: str, th_adjust: Tuple[float, float],
                 data_type_list: list, device: torch.device):
        self.model_path = model_path
        self.th_adjust = th_adjust
        self.data_type_list = data_type_list
        self.device = device
        # Initialize attributes
        self.model = None       # GBSN model, no adjust
        self.prob_id = None
        self.pred_raw = None
        self.pred_adjust = None

    def load_model(self):
        model_dict = load_checkpoint(self.model_path, device=self.device)
        self.model = model_dict['model']

    def predict(self, dl: torch.utils.data.DataLoader):
        """Return predictions from self.model"""
        pred_list = []
        prob_id_list = []
        with torch.no_grad():
            self.model.eval()
            # TODO: Cambiar prob_id a la primer posición
            for pair, prob_id in tqdm(dl):
                pair = to_device(pair, self.device)
                out = self.model(pair)
                out = out.sigmoid().squeeze().cpu().numpy()

                pred_list.append(out)
                prob_id_list.extend(prob_id)

        self.pred_raw = np.concatenate(pred_list)
        self.prob_ids = np.asarray(prob_id_list)

        # threshold adjust
        th_best = self.th_adjust[0]
        margin_best = self.th_adjust[1]
        self.pred_adjust = \
            LinearAdjust(th_best, margin_best).apply(self.pred_raw)

        return self.prob_ids, self.pred_adjust

    def free_memory(self):
        # Free memory from attribute model:
        del self.model
