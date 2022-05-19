# -*- coding: utf-8 -*-

import os
from typing import Tuple, Type, Union

import numpy as np
import torch
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import convert, softmax
from torch_scatter import scatter_add

SEED = 0
torch.manual_seed(SEED)

_default_type = 'torch.FloatTensor'
# _default_type = 'torch.DoubleTensor'
torch.set_default_tensor_type(_default_type)

# TODO: Add support for sparse_mode
# ==================== Save and load models ==========


def define_model(model_class: Type[torch.nn.Module], model_args: dict,
                 sparse_mode: bool = False) -> torch.nn.Module:
    """Wrapper fuction to define model"""

    model = model_class(**model_args)
    return model


def save_checkpoint(model_dict: dict, model_label: str, folder_to_save: str,
                    epoch_label: str = 'auto') -> None:
    """Guarda un punto de control."""

    state_dict = {'epoch': model_dict['epoch'],
                  'model_state_dict': model_dict['model'].state_dict(),
                  'model_class': model_dict['model_class'],
                  'model_args': model_dict['model_args']}
    if 'optimizer' in model_dict.keys():
        state_dict['optimizer_state_dict'] = \
            model_dict['optimizer'].state_dict()
        state_dict['loss'] = model_dict['loss']

    if epoch_label == 'auto':
        save_name = model_label + f"_{model_dict['epoch']:04}.pth"
    else:
        save_name = model_label + '_' + epoch_label + '.pth'

    torch.save(state_dict, os.path.join(folder_to_save, save_name))


def load_checkpoint(checkpoint: Union[str, dict], model=None, opt_class=None,
                    device: str = 'cpu', pretrain: bool = True):
    """ Create model class accordingly to the parammeters passed. Two use cases:
        Pass a dictionary with the setting of the model:

        Pass a string with the path of a pretrained model
    """

    # TODO Remove this functionality in order to be consistent in the input
    # If pass a string, read the pretrained model in the path
    if isinstance(checkpoint, str):
        # cambiar map_location a cpu
        checkpoint = torch.load(checkpoint, map_location='cpu')

    model_dict = {'epoch': checkpoint['epoch']}
    # If we do not pass model class explicitly
    # TODO: Reemplazar por assert
    if model is None:
        # Try to get the class from the checkpoint
        if 'model_class' in checkpoint:
            model = define_model(checkpoint['model_class'],
                                 checkpoint['model_args'])
            model_dict['model_class'] = checkpoint['model_class']
            model_dict['model_args'] = checkpoint['model_args']
        else:
            print('Para este checkpoint se necesita especificar arquitectura '
                  'del modelo')
            exit()

    # If pretrain is set to true, load the state dict
    if pretrain:
        model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()
    model = model.to(device)
    model_dict['model'] = model
    if opt_class is not None:
        opt = opt_class(model.parameters(), lr=0.001)
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        loss = checkpoint['loss']
        model_dict['optimizer'] = opt
        model_dict['loss'] = loss

    return model_dict


# ============================== New define layers ==========

class GlobalAttentionSelect(gnn.GlobalAttention):

    def forward(self, x_att, x_feat, batch, size=None):
        """"""
        x_att = x_att.unsqueeze(-1) if x_att.dim() == 1 else x_att
        x_feat = x_feat.unsqueeze(-1) if x_feat.dim() == 1 else x_feat
        size = batch[-1].item() + 1 if size is None else size

        gate = self.gate_nn(x_att).view(-1, 1)
        x_feat = self.nn(x_feat) if self.nn is not None else x_feat
        assert gate.dim() == x_feat.dim() and gate.size(0) == x_feat.size(0)

        gate = softmax(gate, batch, num_nodes=size)
        out = scatter_add(gate * x_feat, batch, dim=0, dim_size=size)

        return out


# ============================== Graph dataset classes ==========

# -------------------- standard tensor --------------------

class PairData(Data):
    """Clase para aprovechar dataloader de torch_geometric en pares de gráficas
    """

    def __init__(self, edge_index_s, x_s, edge_attr_s,
                 edge_index_t, x_t, edge_attr_t,
                 text_feat_s=None, text_feat_t=None):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_attr_s = edge_attr_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_attr_t = edge_attr_t

    def __inc__(self, key, value):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value)


# TODO: Mover al script donde se ocupa
def sparse_encoded_to_torch(sparse_encoded):
    """Función para transformar de matriz dispersa al formato de
    torch_geometric"""

    edge_index, edge_weight = \
        convert.from_scipy_sparse_matrix(sparse_encoded[0])
    pos = torch.from_numpy(sparse_encoded[1].toarray())
    # Use float
    edge_weight = edge_weight.float()
    x = pos.float()
    return edge_index, edge_weight, x


def datapair_format(s, t):
    """To define the format from the data type obtained from dataset."""

    if isinstance(s, np.ndarray):
        pair = np.concatenate([s, t])
        pair = torch.from_numpy(pair).float()

    # En otro caso debera interpretarse como PairData
    else:
        pair = PairData(edge_index_s=s[0],
                        edge_attr_s=s[1],
                        x_s=s[2],
                        edge_index_t=t[0],
                        edge_attr_t=t[1],
                        x_t=t[2])
        if len(s) == 4:
            pair.text_feat_s = s[3]
            pair.text_feat_t = t[3]

    return pair


class GraphSiameseDatasetDict(Dataset):
    """Dataset class for graph data in single format

    To use with dictionary, all in memory"""

    def __init__(self, ds_list, doc_dict, tsfm, lim=None):
        self.tsfm = tsfm
        self.lim = lim
        self.problem_list, self.truth_list, self.doc_dict = \
            self.define_lists(ds_list, doc_dict)
        self.len = len(self.truth_list)
        assert len(self.problem_list) == self.len

    def define_lists(self, ds_list, doc_dict):
        problem_list = ds_list['problem_list']
        truth_list = ds_list['truth_list']
        if self.lim is not None:
            problem_list = problem_list[:self.lim]
            truth_list = truth_list[:self.lim]

        return problem_list, truth_list, doc_dict

    def __getitem__(self, i):
        id_s = self.problem_list[i]['text_ids'][0]
        id_t = self.problem_list[i]['text_ids'][1]
        s = self.doc_dict[id_s]
        t = self.doc_dict[id_t]
        if self.tsfm is not None:
            s, t = self.tsfm(s), self.tsfm(t)

        pair = datapair_format(s, t)
        return pair, self.truth_list[i]

    def __len__(self):
        return self.len


class GraphSiameseDatasetDictJoin(Dataset):
    """Dataset class for graph data and stylistic data"""

    def __init__(self, ds_list, doc_dict_list, tsfm_list, lim=None):
        self.tsfm_list = tsfm_list
        self.lim = lim
        self.dataset_list = \
            [GraphSiameseDatasetDict(ds_list, doc_dict, tsfm, lim) for
             (doc_dict, tsfm) in zip(doc_dict_list, tsfm_list)]
        self.verify_integrity()

    def verify_integrity(self):
        for dataset in self.dataset_list[1:]:
            assert dataset.len == self.dataset_list[0].len
        for dataset in self.dataset_list[1:]:
            for i, t in enumerate(self.dataset_list[0].truth_list):
                assert t == dataset.truth_list[i]

    def __getitem__(self, i):
        item_list = [dataset[i][0] for dataset in self.dataset_list]
        return item_list, self.dataset_list[0][i][1]

    def __len__(self):
        return self.dataset_list[0].len


# ============================== Models ==========

def define_dense_layer(in_ch, mid_ch, out_ch, layers_num, final_relu):
    """Auxiliar to define fully conected neural network"""

    layers_list = [torch.nn.Linear(in_ch, mid_ch),
                   torch.nn.ReLU()]
    for i in range(layers_num-2):
        layers_list.append(torch.nn.Linear(mid_ch, mid_ch))
        layers_list.append(torch.nn.ReLU())

    layers_list.append(torch.nn.Linear(mid_ch, out_ch))
    if final_relu:
        layers_list.append(torch.nn.ReLU())

    return torch.nn.Sequential(*layers_list)


class GCLayer(torch.nn.Module):
    """Class defining graph layer used.

    Formed by a convolutional layer, batch normalization and ReLU activation.
    Auxiliar to adjust the usage of the layers implemented."""

    def __init__(self, conv_type, in_ch, out_ch):
        super().__init__()
        self.conv = conv_type(in_ch, out_ch)
        self.norm = gnn.BatchNorm(out_ch)

    def forward(self, x, edge_index, edge_weight, x_0=None):
        if isinstance(self.conv, gnn.GCN2Conv):
            x = self.norm(self.conv(x, x_0, edge_index, edge_weight))
        else:
            x = self.norm(self.conv(x, edge_index, edge_weight))

        x = x.relu()
        return x


class GBFeatures(torch.nn.Module):
    """Class to define a single graph-based feature extraction component"""

    def __init__(self, num_node_features=38,
                 conv_layers_num=6,
                 conv_type=gnn.LEConv,
                 conv_args={'alpha': 0.1},
                 # conv_residual=False,
                 h_ch=64, out_ch=64,
                 pool_type=GlobalAttentionSelect,
                 pool_att_ch=32, pool_att_layers=2,
                 pool_ref='last',
                 # drop_p=None,
                 ):
        # initialize variables
        super().__init__()
        self.num_node_features = num_node_features
        self.conv_layers_num = conv_layers_num
        self.conv_type = conv_type
        self.conv_args = None if conv_type is None else conv_args
        self.h_ch = h_ch
        self.out_ch = out_ch
        self.pool_type = pool_type
        self.pool_att_ch = pool_att_ch
        self.pool_att_layers = pool_att_layers
        self.pool_ref = pool_ref

        # =============== Node feature extraction layers
        # ========== Conv layers
        conv_layers_list = [GCLayer(self.conv_type, self.num_node_features,
                                    self.h_ch)]
        for i in range(self.conv_layers_num - 2):
            conv_layers_list.append(GCLayer(self.conv_type,
                                            self.h_ch, self.h_ch))

        conv_layers_list.append(GCLayer(self.conv_type,
                                        self.h_ch, self.out_ch))
        self.conv_layers = torch.nn.ModuleList(conv_layers_list)

        # =============== Pooling layer
        if self.pool_type == GlobalAttentionSelect:
            if self.pool_ref in ['first-last']:
                self.ch_pool = num_node_features + out_ch
            else:
                self.ch_pool = out_ch

            gate_nn = define_dense_layer(self.ch_pool, pool_att_ch, 1,
                                         self.pool_att_layers, True)
            self.pool = self.pool_type(gate_nn)

        elif self.pool_type == gnn.global_mean_pool:
            self.pool = self.pool_type

    def apply_conv_layers(self, x_0, edge_index, edge_weight=None):
        x_list = []
        x = x_0
        for layer in self.conv_layers:
            x = layer(x, edge_index, edge_weight=edge_weight)
            x_list.append(x)

        return x_list

    def apply_pool_layer(self, x_0, x_last, batch):
        if self.pool_type == GlobalAttentionSelect:
            # all_batch_nodes * out_ch -> batch * out_ch
            if self.pool_ref == 'last':
                sec = self.pool(x_last, x_last, batch)
            elif self.pool_ref == 'first-last':
                sec = self.pool(torch.cat((x_0, x_last), dim=1),
                                x_last, batch)
            return sec

        elif self.pool_type == gnn.global_mean_pool:
            return self.pool(x_last, batch)

    # edge_weight cambiar por edge weights
    def forward(self, x, edge_index, batch, edge_weight=None):
        x_0 = x
        # ===== 1. Obtain node embeddings
        x_list = self.apply_conv_layers(x_0, edge_index, edge_weight)
        # last layer pointer
        x_last = x_list[-1]
        # ===== 2. Obtain graph embedding
        # all_batch_nodes * out_ch -> batch * out_ch
        return self.apply_pool_layer(x_0, x_last, batch)


class TextFeatures(torch.nn.Module):
    """Class to define a single stylistic feature extraction component.

    Use fully connected neural network"""

    def __init__(self, num_text_features,
                 text_feat_layers_num,
                 h_ch=64, out_ch=64):
        super().__init__()
        if text_feat_layers_num == 0:
            self.feat_layer = torch.nn.Identity()
            self.out_ch = num_text_features
        else:
            self.feat_layer = \
                define_dense_layer(num_text_features, h_ch, out_ch,
                                   text_feat_layers_num,
                                   final_relu=False)
            self.out_ch = out_ch

    def forward(self, x):
        return self.feat_layer(x)


class SiameseNetwork(torch.nn.Module):
    """Implement a Siamese Network.  """

    def __init__(self, raw_components_list: list,
                 pretrain: bool = True,
                 freeze_param: str = 'graph_embeddings',
                 final_out_join: str = 'abs',
                 final_out_layers_num: int = 4,
                 final_out_ch: int = 64) -> None:
        """__init__.

        Args:
            raw_components_list (list): List of the components used to extract
                features. A component should be either:
                    - A dictionary defining class_model and class_args.
                    - Pretrained components. A checkpoint of a SiameseNetwork
                      with single component. In this case, we use the same
                      component that the one in the SiameseNetwork class.
            pretrain (bool): Only relevant in pretrained components. Define
                if the component is used with pretrained weights.
            freeze_param (str): Only relevant in pretrained components. Define
                if the weights of the components remain freezing or will be
                updated during training.
            final_out_join (str): Define how the tensors generated by each
                component are join in a single tensor. (abs, cat or
                use_constractive)
            final_out_layers_num (int): Only relevant if final_out_join is not
                use_constractive. Define number of layers in the
                classification network.
            final_out_ch (int): Only relevant if final_out_join is not
                use_constractive. Define chanels (neurons) used in each layer
                of the classification network.

        Returns:
            None:
        """
        # initialize variables
        super().__init__()
        # TODO assert parameters are correct
        self.pretrain = pretrain
        self.freeze_param = freeze_param
        self.final_out_join = final_out_join
        self.final_out_layers_num = final_out_layers_num
        self.final_out_ch = final_out_ch

        # Define components list, types and output chanels
        self.components_types, self.components_list, out_ch_list = \
            zip(*[self.define_embedding(component)
                for component in raw_components_list])
        # Define module list to manage components within pytorch
        self.components = torch.nn.ModuleList(self.components_list)
        self.out_ch = sum(out_ch_list)

        # ========== Optional consatractive loss
        if self.final_out_join == 'use_constractive':
            self.final_out = None
        else:
            # =============== Out layer
            # This option implements absolute value of difference
            if final_out_join == 'abs':
                in_ch = self.out_ch
            # This option implements concatenation of components tensors
            elif final_out_join == 'cat':
                in_ch = 2 * self.out_ch
            # ========== classification layer
            self.final_out = define_dense_layer(in_ch, final_out_ch, 1,
                                                self.final_out_layers_num,
                                                final_relu=False)

    def vector_reduction(self, out_s: torch.Tensor, out_t: torch.Tensor,
                         join_mode: str) -> torch.Tensor:
        """Function to join the tensor generated by each subnetwork and join
        in a single tensor.

        Args:
            out_s (torch.Tensor): First Tensor
            out_t (torch.Tensor): Second Tensor
            join_mode (str): Define how to join tensors

        Returns:
            torch.Tensor:
        """
        if join_mode == 'abs':
            join_out = torch.abs(out_s - out_t)
        elif join_mode == 'cat':
            join_out = torch.cat([out_s, out_t], dim=1)

        return join_out

    def apply_embedding(self, embedding_type: str,
                        embedding: Union[GBFeatures, TextFeatures],
                        pair: Union[PairData, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """To apply graph component to graph data and stylistic component to
        a tensor.

        Args:
            embedding_type (str): Define type of the component, this define the
                process used.
            embedding (Union[GBFeatures, TextFeatures]): Instance of the
                feature extraction component.
            pair (Union[PairData, torch.Tensor]): Data representing pair of
                graphs or pair of stylistic features.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Resulting tensors for each
                data in pair input.
        """
        if embedding_type == 'text_feat':
            len_pair = pair.shape[1]
            assert len_pair % 2 == 0
            len_pair = int(len_pair / 2)
            s = pair[:, :len_pair]
            t = pair[:, len_pair:]
            out_feat_s = embedding(s)
            out_feat_t = embedding(t)
            return (out_feat_s, out_feat_t)
        # En otro caso es un GCN
        else:
            emb_s = embedding(pair.x_s, pair.edge_index_s,
                              pair.x_s_batch, pair.edge_attr_s)
            emb_t = embedding(pair.x_t, pair.edge_index_t,
                              pair.x_t_batch, pair.edge_attr_t)
            return (emb_s, emb_t)

    def define_embedding(self, component: Union[dict, str]) -> tuple:
        """define_embedding.

        Args:
            component (dict): Two posible flavour of dict:
                - dict of the parameters of the component
                - dict with a pretrained SiameseNetwork using SINGLE component
        """

        # If is a dict with 'class' and 'args'. No pretrained component
        if isinstance(component, dict):
            embedding = define_model(component['class'],
                                     component['args'])
            out_ch = embedding.out_ch

        # If is checkpoint of a pretrained SiameseNetwork
        elif isinstance(component, str):
            # Load the model from checkpoint
            model_dict = load_checkpoint(component, pretrain=self.pretrain)
            # Right now we only support this functionality when the
            # SiameseNetwork has one component
            siamese_model_components = model_dict['model'].components_list
            assert len(siamese_model_components) == 1
            # Use the unique component of the SiameseNetwork
            embedding = siamese_model_components[0]
            out_ch = embedding.out_ch
            if (self.pretrain and self.freeze_param == 'graph_components'):
                # Freeze embedding parameters
                for param in embedding.parameters():
                    param.requires_grad = False
                embedding.eval()

        # Deduce component class
        if isinstance(embedding, GBFeatures):
            embedding_type = 'GCN'
        elif isinstance(embedding, TextFeatures):
            embedding_type = 'text_feat'

        return embedding_type, embedding, out_ch

    def forward(self, item_list: list) -> torch.Tensor:
        """Forward of the class.

        Args:
            item_list (list): List of data pairs

        Returns:
            torch.Tensor: Final score for the pairs
        """
        # Aplica cada módulo a cada elemento de la lista
        components_out = \
            [self.apply_embedding(embedding_type, embedding, pair) for
             (embedding_type, embedding, pair) in
             zip(self.components_types, self.components, item_list)]

        # Concatenar resultado
        out_s = torch.cat([emb[0] for emb in components_out], dim=1)
        out_t = torch.cat([emb[1] for emb in components_out], dim=1)

        join_out = self.vector_reduction(out_s, out_t, self.final_out_join)
        lgts_out = self.final_out(join_out)
        return lgts_out


# Global dict to define actual classes
_class_dict = {'SiameseNetwork': SiameseNetwork,
               'GBFeatures': GBFeatures,
               'TextFeatures': TextFeatures,
               'gnn.LEConv': gnn.LEConv,
               'gnn.GraphConv': gnn.GraphConv,
               'gnn.GCN2Conv': gnn.GCN2Conv,
               'gnn.TAGConv': gnn.TAGConv,
               'GlobalAttentionSelect': GlobalAttentionSelect}


# ============================== Compatibility for old classes ==========
# These classes are used when load old trained models, like the final
# submission of PAN21. All of the following are implemented before

class GCNSiamese2(torch.nn.Module):
    def __init__(self, num_node_features=38,
                 h_ch=64, out_ch=64,
                 pool_att_ch=32, pool_att_layers=2,
                 drop_p=None, pool_ref='last',
                 conv_layers_num=3,
                 conv_type=gnn.GraphConv,
                 conv_args={'alpha': 0.1},
                 conv_residual=False,
                 conv_long_type=None,
                 conv_long_args=[],
                 att_type=None,
                 att_heads=1,
                 pool_type=GlobalAttentionSelect,
                 pool_final_relu=True,
                 final_out_layers_num=1,
                 final_out_ch=None,
                 final_out_join='abs',
                 final_out_join_layers_num=3,
                 text_feat_layers_num=None,
                 num_text_features=0,
                 text_feat_out_ch=32,
                 sparse_mode=False):
        super().__init__()
        self.num_node_features = num_node_features
        self.h_ch = h_ch
        self.out_ch = out_ch
        self.pool_att_ch = pool_att_ch
        self.pool_att_layers = pool_att_layers
        self.drop_p = drop_p
        self.pool_ref = pool_ref
        assert conv_layers_num in [3, 6, 9, 12]
        self.conv_layers_num = conv_layers_num
        self.conv_type = conv_type
        self.conv_args = None if conv_type is None else conv_args
        self.conv_residual = conv_residual
        self.conv_long_type = conv_long_type
        self.conv_long_args = \
            None if conv_long_args is None else conv_long_args
        self.att_type = att_type
        self.att_heads = att_heads
        self.pool_type = pool_type
        self.pool_final_relu = pool_final_relu
        self.final_out_layers_num = final_out_layers_num
        self.final_out_ch = None if final_out_layers_num == 1 else final_out_ch
        self.final_out_join = final_out_join
        self.final_out_join_layers_num = final_out_join_layers_num
        self.text_feat_layers_num = text_feat_layers_num
        self.num_text_features = num_text_features
        self.text_feat_out_ch = text_feat_out_ch
        self.sparse_mode = sparse_mode

        # =============== Node feature extraction layers
        # ========== Conv layers
        # ===== First 3 layers, always
        # Trabaja diferente GCN2Conv
        if self.conv_type is gnn.GCN2Conv:
            self.conv1 = gnn.GraphConv(num_node_features, h_ch)
            self.conv2 = self.conv_type(h_ch, **self.conv_args)
            self.conv_out = self.conv_type(h_ch, **self.conv_args)
        else:
            self.conv1 = self.conv_type(num_node_features, h_ch)
            self.conv2 = self.conv_type(h_ch, h_ch)
            self.conv_out = self.conv_type(h_ch, out_ch)

        self.norm1 = gnn.BatchNorm(h_ch)
        self.norm2 = gnn.BatchNorm(h_ch)
        self.norm_out = gnn.BatchNorm(out_ch)

        # ===== Layers 4-6
        if conv_layers_num > 3:
            # Trabaja diferente GCN2Conv
            if self.conv_type is gnn.GCN2Conv:
                self.conv3 = self.conv_type(h_ch, **self.conv_args)
                self.conv4 = self.conv_type(h_ch, **self.conv_args)
                self.conv5 = self.conv_type(h_ch, **self.conv_args)
            else:
                self.conv3 = self.conv_type(h_ch, h_ch)
                self.conv4 = self.conv_type(h_ch, h_ch)
                self.conv5 = self.conv_type(h_ch, h_ch)

            self.norm3 = gnn.BatchNorm(h_ch)
            self.norm4 = gnn.BatchNorm(h_ch)
            self.norm5 = gnn.BatchNorm(h_ch)

        if conv_layers_num > 6:
            # Trabaja diferente GCN2Conv
            if self.conv_type is gnn.GCN2Conv:
                self.conv6 = self.conv_type(h_ch, **self.conv_args)
                self.conv7 = self.conv_type(h_ch, **self.conv_args)
                self.conv8 = self.conv_type(h_ch, **self.conv_args)
            else:
                self.conv6 = self.conv_type(h_ch, h_ch)
                self.conv7 = self.conv_type(h_ch, h_ch)
                self.conv8 = self.conv_type(h_ch, h_ch)

            self.norm6 = gnn.BatchNorm(h_ch)
            self.norm7 = gnn.BatchNorm(h_ch)
            self.norm8 = gnn.BatchNorm(h_ch)

        if conv_layers_num > 9:
            # Trabaja diferente GCN2Conv
            if self.conv_type is gnn.GCN2Conv:
                self.conv9 = self.conv_type(h_ch, **self.conv_args)
                self.conv10 = self.conv_type(h_ch, **self.conv_args)
                self.conv11 = self.conv_type(h_ch, **self.conv_args)
            else:
                self.conv9 = self.conv_type(h_ch, h_ch)
                self.conv10 = self.conv_type(h_ch, h_ch)
                self.conv11 = self.conv_type(h_ch, h_ch)

            self.norm9 = gnn.BatchNorm(h_ch)
            self.norm10 = gnn.BatchNorm(h_ch)
            self.norm11 = gnn.BatchNorm(h_ch)

        # ========== Conv long layer
        if self.conv_long_type is None:
            pass
        else:
            self.conv_long = self.conv_long_type(out_ch, out_ch,
                                                 *self.conv_long_args)
            self.norm_long = gnn.BatchNorm(out_ch)

        # ========== Attention layer
        if self.att_type is None:
            pass
        elif self.att_type in [gnn.TransformerConv, gnn.GATConv]:
            self.att1 = att_type(out_ch, out_ch, heads=self.att_heads)
        elif self.att_type == gnn.DNAConv:
            self.att1 = att_type(out_ch, heads=self.att_heads)
            # cached not working
        elif self.att_type == gnn.JumpingKnowledge:
            conv_long_layers = 0 if self.conv_type is None else 1
            self.att1 = att_type(mode='lstm', channels=out_ch,
                                 num_layers=conv_layers_num + conv_long_layers)

        # =============== Pooling layer
        if self.pool_type == GlobalAttentionSelect:
            if self.pool_ref in ['first-atten']:
                self.ch_pool = num_node_features + out_ch

            else:
                self.ch_pool = out_ch

            gate_nn = define_dense_layer(self.ch_pool, pool_att_ch, 1,
                                         self.pool_att_layers,
                                         final_relu=self.pool_final_relu)
            self.pool = self.pool_type(gate_nn)

        elif self.pool_type == gnn.global_mean_pool:
            self.pool = self.pool_type

        # =============== Text feat layer
        # Pendiente
        if self.text_feat_layers_num is None:
            self.text_feat_out_ch = 0
        elif self.text_feat_layers_num == 0:
            self.text_feat_out_ch = self.num_text_features
            self.text_feat_layer = torch.nn.Identity()
        else:
            self.text_feat_layer = \
                define_dense_layer(num_text_features, 64, text_feat_out_ch,
                                   text_feat_layers_num,
                                   final_relu=False)

        # =============== Out layer
        if self.final_out_join == 'use_constractive':
            self.reduction = None
            self.final_out = None

        # ===== reduction layer
        if self.final_out_join == 'abs':
            self.reduction = None
            in_ch = self.out_ch + self.text_feat_out_ch
        elif self.final_out_join == 'cat':
            self.reduction = None
            in_ch = 2 * (self.out_ch + self.text_feat_out_ch)

        self.final_out = define_dense_layer(in_ch, final_out_ch, 1,
                                            self.final_out_layers_num,
                                            final_relu=False)

    def layer_conv(self, norm, conv, x, edge_index, edge_weight=None, x_0=None):
        if isinstance(conv, gnn.GCN2Conv):
            x = norm(conv(x, x_0, edge_index, edge_weight=edge_weight))
        else:
            x = norm(conv(x, edge_index, edge_weight=edge_weight))

        x = x.relu()
        return x

    def layer_att(self, x_list, edge_index):
        if self.att_type in [gnn.TransformerConv, gnn.GATConv]:
            # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
            x = self.att1(x_list[-1], edge_index)
        elif self.att_type == gnn.DNAConv:
            # all_batch_nodes * len(x_list) * h_ch -> all_batch_nodes * out_ch
            x_sec = torch.stack(x_list, dim=1)
            x = self.att1(x_sec, edge_index)
        elif self.att_type == gnn.JumpingKnowledge:
            x = self.att1(x_list)

        return x

    def layer_pool(self, x_0, x_long_out, x_att, x_last, batch):
        if self.pool_type == GlobalAttentionSelect:
            # all_batch_nodes * out_ch -> batch * out_ch
            if self.pool_ref == 'last':
                sec = self.pool(x_long_out, x_last, batch)

            elif self.pool_ref == 'atten':
                sec = self.pool(x_att, x_last, batch)

            elif self.pool_ref == 'first-atten':
                sec = self.pool(torch.cat((x_0, x_att), dim=1),
                                x_last, batch)
            return sec

        elif self.pool_type == gnn.global_mean_pool:
            return self.pool(x_last, batch)

    def conv_group(self, x_0, edge_index, edge_weight=None):
        # all_batch_nodes * 38 -> all_batch_nodes * h_ch
        x_1 = self.layer_conv(self.norm1, self.conv1,
                              x_0, edge_index, edge_weight)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_2 = self.layer_conv(self.norm2, self.conv2,
                              x_1, edge_index, edge_weight, x_1)
        if self.conv_layers_num == 3:
            # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
            x_conv_out = self.layer_conv(self.norm_out, self.conv_out,
                                         x_2, edge_index, edge_weight, x_1)
            return [x_1, x_2, x_conv_out]

        # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
        x_3 = self.layer_conv(self.norm3, self.conv3,
                              x_2, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_4 = self.layer_conv(self.norm4, self.conv4,
                              x_3, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_5 = self.layer_conv(self.norm5, self.conv5,
                              x_4, edge_index, edge_weight, x_1)
        if self.conv_layers_num == 6:
            # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
            x_conv_out = self.layer_conv(self.norm_out, self.conv_out,
                                         x_5, edge_index, edge_weight, x_1)
            return [x_1, x_2, x_3, x_4, x_5, x_conv_out]

        # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
        x_6 = self.layer_conv(self.norm6, self.conv6,
                              x_5, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_7 = self.layer_conv(self.norm7, self.conv7,
                              x_6, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_8 = self.layer_conv(self.norm8, self.conv8,
                              x_7, edge_index, edge_weight, x_1)
        if self.conv_layers_num == 9:
            # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
            x_conv_out = self.layer_conv(self.norm_out, self.conv_out,
                                         x_8, edge_index, edge_weight, x_1)
            return [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_conv_out]

        # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
        x_9 = self.layer_conv(self.norm9, self.conv9,
                              x_8, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_10 = self.layer_conv(self.norm10, self.conv10,
                               x_9, edge_index, edge_weight, x_1)
        # all_batch_nodes * h_ch -> all_batch_nodes * h_ch
        x_11 = self.layer_conv(self.norm11, self.conv11,
                               x_10, edge_index, edge_weight, x_1)
        if self.conv_layers_num == 12:
            # all_batch_nodes * h_ch -> all_batch_nodes * out_ch
            x_conv_out = self.layer_conv(self.norm_out, self.conv_out,
                                         x_11, edge_index, edge_weight, x_1)
            return [x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9,
                    x_10, x_11, x_conv_out]

    def forward_one(self, x, edge_index, batch, edge_weight=None):
        x_0 = x
        # ===== 1. Obtain node embeddings
        x_list = self.conv_group(x_0, edge_index, edge_weight)
        # last layer pointer
        x_last = x_list[-1]
        if self.conv_long_type is not None:
            # all_batch_nodes * out_ch -> all_batch_nodes * out_ch
            x_long_out = self.layer_conv(self.norm_long, self.conv_long,
                                         x_last, edge_index, edge_weight)
            x_list = x_list.append(x_long_out)
            x_last = x_long_out
        else:
            x_long_out = x_last

        # attention
        if self.att_type is not None:
            x_att = self.layer_att(x_list, edge_index)
            x_last = x_att
        else:
            x_att = None

        # ===== 2. Obtain graph embedding
        # all_batch_nodes * out_ch -> batch * out_ch
        return self.layer_pool(x_0, x_long_out, x_att,
                               x_last, batch)

    def forward_one_feat(self, tensor):
        return self.text_feat_layer(tensor)

    def forward(self, pair):
        out_s = self.forward_one(pair.x_s, pair.edge_index_s,
                                 pair.x_s_batch, pair.edge_attr_s)
        out_t = self.forward_one(pair.x_t, pair.edge_index_t,
                                 pair.x_t_batch, pair.edge_attr_t)
#         if self.num_text_features > 0:
        if self.text_feat_layers_num is not None:
            out_feat_s = self.forward_one_feat(pair.text_feat_s)
            out_feat_t = self.forward_one_feat(pair.text_feat_t)
            out_s = torch.cat([out_s, out_feat_s], dim=1)
            out_t = torch.cat([out_t, out_feat_t], dim=1)

        join_out = vector_reduction(out_s, out_t, self.final_out_join,
                                    self.reduction)
        lgts_out = self.final_out(join_out)
        return lgts_out


# ==================== Join models ========== retirar

def define_text_feat_layer(text_feat_layers_num, num_text_features):
    if text_feat_layers_num == 0:
        text_feat_layer = torch.nn.Identity()
    else:
        text_feat_layer = \
            define_dense_layer(num_text_features, 64, num_text_features,
                               text_feat_layers_num,
                               final_relu=False)
    return text_feat_layer


def define_reduction_layer(final_out_join, out_ch,
                           final_out_join_layers_num=3):
    # =============== Out layer
    if final_out_join == 'abs':
        reduction = None
        in_ch = out_ch
    elif final_out_join == 'cat':
        reduction = None
        in_ch = 2 * out_ch

    return reduction, in_ch


class SiameseJoin(torch.nn.Module):
    def __init__(self, checkpoint_list,
                 pretrain=True,
                 freeze_param='graph_embeddings',
                 final_out_layers_num=3,
                 final_out_ch=64,
                 final_out_join='abs'):
        super().__init__()
        self.pretrain = pretrain
        self.freeze_param = freeze_param
        self.final_out_layers_num = final_out_layers_num
        self.final_out_ch = final_out_ch
        self.final_out_join = final_out_join

        self.embeddings_types, self.embeddings_list, out_ch_list = \
            zip(*[self.define_embedding(checkpoint)
                for checkpoint in checkpoint_list])
        self.embeddings = torch.nn.ModuleList(self.embeddings_list)
        self.out_ch = sum(out_ch_list)

        self.reduction, in_ch = \
            define_reduction_layer(final_out_join, self.out_ch)

        self.final_out = define_dense_layer(in_ch, final_out_ch, 1,
                                            self.final_out_layers_num,
                                            final_relu=False)

    def define_embedding(self, checkpoint):
        if checkpoint['model_class'] == 'text_feat':
            return ('text_feat',
                    define_text_feat_layer(**checkpoint['args']),
                    checkpoint['args']['num_text_features'])
        # En otro caso debe ser un checkpoint
        else:
            # recover model
            model_dict = load_checkpoint(checkpoint, pretrain=self.pretrain)
            model = model_dict['model']
            if (self.pretrain and self.freeze_param == 'graph_embeddings'):
                # congelamos los parámetros
                for param in model.parameters():
                    param.requires_grad = False
                # congelamos las estadísticas
                model.eval()

            # cambiamos comportamiento de forward
            model.forward = model.forward_one
            # Hacemos nulas las capas que no ocupamos
            if 'final_out' in model._modules.keys():
                model.final_out = None

            if 'reduction' in model._modules.keys():
                model.reduction = None

            return 'GCN', model, model.out_ch

    def vector_reduction(self, out_s, out_t, join_mode, reduction_fn=None):
        if join_mode == 'abs':
            join_out = torch.abs(out_s - out_t)
        elif join_mode == 'cat':
            join_out = torch.cat([out_s, out_t], dim=1)

        return join_out

    def apply_embedding(self, embedding_type, embedding, pair):
        if embedding_type == 'text_feat':
            len_pair = pair.shape[1]
            assert len_pair % 2 == 0
            len_pair = int(len_pair / 2)
            s = pair[:, :len_pair]
            t = pair[:, len_pair:]
            out_feat_s = embedding(s)
            out_feat_t = embedding(t)
            return (out_feat_s, out_feat_t)
        # En otro caso es un GCN
        else:
            emb_s = embedding(pair.x_s, pair.edge_index_s,
                              pair.x_s_batch, pair.edge_attr_s)
            emb_t = embedding(pair.x_t, pair.edge_index_t,
                              pair.x_t_batch, pair.edge_attr_t)
            return (emb_s, emb_t)

    def forward(self, item_list):
        # Aplica cada módulo a cada elemento de la lista
        embeddings_out = \
            [self.apply_embedding(embedding_type, embedding, pair) for
             (embedding_type, embedding, pair) in
             zip(self.embeddings_types, self.embeddings, item_list)]

        # Concatenar resultado
        out_s = torch.cat([emb[0] for emb in embeddings_out], dim=1)
        out_t = torch.cat([emb[1] for emb in embeddings_out], dim=1)

        join_out = self.vector_reduction(out_s, out_t, self.final_out_join,
                                    self.reduction)
        lgts_out = self.final_out(join_out)
        return lgts_out
