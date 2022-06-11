# -*- coding: utf-8 -*-

import os
import re
from collections import Counter
from nltk import tokenize, pos_tag
from unidecode import unidecode
from copy import deepcopy as Deepcopy

import networkx as nx
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import RegexpTokenizer

import torch
from torch_geometric import utils

from common_func import total_size, time_string
    

_treebank = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS',
             'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$',
             'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
             'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']
_classes = _treebank + ['$PUNCT', '$OTHER']
# reduce_classes
_rc_dict = {'full': [],
            'long': ['CD',  # Cardinal number
                     'FW',  # Foreign word
                     'LS',  # List item marker
                     'SYM',  # Symbol
                     'UH',  # Interjection
                     '$OTHER'  # Others
                     ],
            'med': ['CD',  # Cardinal number
                    'FW',  # Foreign word
                    'JJ', 'JJR', 'JJS',  # Adjectives
                    'LS',  # List item marker
                    'NN', 'NNS', 'NNP', 'NNPS',  # Nouns
                    'RB', 'RBR', 'RBS',  # Adverb
                    'SYM',  # Symbol
                    'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',  # verb
                    '$OTHER'  # Others
                    ],
            'short': _classes}

# ============================== Classes to preprocess text ===================

class TextParser:
    """Clase para procesar texto.
    Sustituye caracteres no ascii
    Retira puntuación (opcional)
    Tokeniza y segmenta en enunciados
    Etiqueta POS (part of speech) de acuerdo a treebank con $PUNCT y $OTHER
    Devuelve lista de tuplas
    """

    def __init__(self, char_pre=True,
                 remove_punct=r".,;:!?()[]{}`'\"@#$^&*+-|=~_",
                 # reduce_classes=[],
                 ):
        self.char_pre = char_pre
        self.remove_punct = remove_punct
        # self.reduce_classes = reduce_classes
        self.classes = _classes
        self.punct_all = r" .,;:!?()[]{}`'\"@#$^&*+-|=~_"
        # compile regular expresions
        self.blanks = re.compile(r'\s+')

    def char_preprocess(self, text):
        # sustitute blank with single space
        text2 = self.blanks.sub(' ', text)
        # sustitute non ascii characters
        return unidecode(text2)

    def custom_tokenization(self,text):
        exp1 = re.compile('><')
        exp2 = re.compile('<')
        exp3 = re.compile('>')
        text = exp1.sub("> <", text)
        text = exp2.sub(" <", text)
        text = exp3.sub("> ", text)
        tokenizer = RegexpTokenizer('\w+|<\w*?>|\S+')
        return tokenizer.tokenize(text)

    def define_tagged(self, text):
        """tokenize and pos"""

        # Preprocess characters
        if self.char_pre:
            text = self.char_preprocess(text)

        # Process text to get tags
        # initial pos tag
        #tokens_pos = pos_tag(tokenize.word_tokenize(text))
        tokens_pos = pos_tag(self.custom_tokenization(text))
        # complement pos tag
        text_tag = []
        classes_a = Deepcopy(self.classes)
        classes_a.remove('$OTHER')
        for t in tokens_pos:
            # Remove punctuation in self.remove_punct
            if t[0] in self.remove_punct:
                continue

            # colocar tags de puntuación y otros
            if t[0] in self.punct_all:
                t = (t[0].lower(), '$PUNCT')

            # Add tag to other
            if t[1] not in self.classes:
                t = (t[0].lower(), '$OTHER')

            else:
                t = (t[0].lower(), t[1])

            # Lowercase token
            text_tag.append(t)

        return text_tag


# ============================== Text to graph classes ====================

class ToGraph:
    """Parent class to generate networkx graph from parsed text"""

    def __init__(self, graph_type='Graph'):
        self.graph_type = graph_type
        if graph_type == 'MultiDiGraph':
            self.G = nx.MultiDiGraph()
        elif graph_type == 'MultiGraph':
            self.G = nx.MultiGraph()
        elif graph_type == 'DiGraph':
            self.G = nx.DiGraph()
        elif graph_type == 'Graph':
            self.G = nx.Graph()
        else:
            try:
                self.G = graph_type
                self.graph_type = type(self.G).__name__
            except AttributeError:
                print('graph_type no válida')
                exit()

    def node_label(self, word):
        """To set the name of each node"""

        return word

    def grown_graph(self, text):
        """To add nodes and edges from parsed text"""

        pass

    def final_tune(self, remove_loops=True):
        """Optional final operation in graph"""

        if remove_loops is True:
            self.G.remove_edges_from(nx.selfloop_edges(self.G))

    def define_node_mask(self, measure=nx.closeness_centrality):
        """Se usa en unmasking"""

        # Get measure for all nodes
        nodes_measure = measure(self.G)
        return nodes_measure

    def get_size(self):
        """To debug"""

        return (total_size(self.G.__contains__) +
                total_size(self.G._node))

    def show_graph(self, graph_id=None, dest_folder='ignore_Imagenes'):
        """Método para graficar multigraficas con graphviz. Necesita
        salvar la imagen en archivo"""

        # Exportar a formato agraph
        A = nx.nx_agraph.to_agraph(self.G)
        # prog=[‘neato’|’dot’|’twopi’|’circo’|’fdp’|’nop’]
        A.layout(prog='circo')
        if graph_id is None:
            graph_id = time_string()

        name = self.graph_type + '_' + str(graph_id) + '.png'
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        path = os.path.join(dest_folder,  name)
        print('self.Guardando imagen en ' + name)
        A.draw(path)
        # reading png image file
        im = img.imread(path)
        # show image
        plt.imshow(im)
        plt.show()

    def show_attributes(self):
        print('----------ToGraph.show_attributes method')
        print('graph_type: ', self.graph_type)
        print('Data size(bytes): ', self.get_size())
        print('Nodes:', nx.number_of_nodes(self.G))
        print('Edges:', nx.number_of_edges(self.G))
        print('loops:', nx.number_of_selfloops(self.G))


class ToCoocurrence(ToGraph):
    """Coocurrence graph from parsed text"""

    def __init__(self,
                 reduce_classes=[],
                 graph_type='Graph', weigth_labels=True):
        super().__init__(graph_type)
        self.weigth_labels = weigth_labels
        self.reduce_classes = reduce_classes

    def node_label(self, token):
        if token[1] not in self.reduce_classes:
            return token
        else:
            return (('$' + str(token[1])), token[1])

    def define_edges(self, text_tag):
        """Obtener lista de aristas. Usada para probar formatos"""

        # Coocurrence edges loop
        edges_list = []
        node_pre = self.node_label(text_tag[0])
        for token in text_tag[1:]:
            node_act = self.node_label(token)
            edges_list.append((node_pre, node_act))
            node_pre = node_act

        total = len(edges_list)
        edges = [(node_pre, node_act, (c / total))
                 for (node_pre, node_act), c in Counter(edges_list).items()]
        return edges

    def grown_from_edges(self, edges):
        """Agregar aristas a gráfica. Usada para probar formatos"""

        self.G.add_weighted_edges_from(edges)
        self.final_tune()

    def grown_graph(self, text_tag):
        edges = self.define_edges(text_tag)
        self.grown_from_edges(edges)

    def to_sparse_raw(self):
        """Transformar a matriz dispersa de scipy y lista de nodos"""

        node_names = list(self.G.nodes())
        sparse = nx.convert_matrix.to_scipy_sparse_matrix(self.G)
        return sparse, node_names

    def to_sparse(self, return_node_words=False):
        """Transformar a matriz dispersa de scipy. Usada para probar formatos
        """
        node_pos = [node[1] for node in self.G.nodes()]
        sparse = nx.convert_matrix.to_scipy_sparse_matrix(self.G)
        if return_node_words:
            node_words = [node for node in self.G.nodes()]
            return sparse, node_pos, node_words

        else:
            return sparse, node_pos

    def to_sparse_encoded(self):
        """Transformar a matriz dispersa de scipy con las categorías POS como
        atributos en matiz dispersa también. Usada para probar formatos"""

        sparse, node_pos = self.to_sparse()
        node_attr = encode_pos(node_pos)
        return sparse, node_attr

    def to_data(self, pos_encoder='default'):
        """Transformar a tensor de pytorch como se usaría en torch geometric.
        Usada para probar formatos"""

        nx_graph = self.G
        # verificar que sea graph o digraph
        assert type(nx_graph).__name__ in ['Graph', 'DiGraph']

        # Define graph, edges weight to edge_attr
        weight = 'weight'
        data = utils.from_networkx(nx_graph)
        data_dir = dir(data)
        if weight in data_dir:
            data['edge_attr'] = data[weight]
            delattr(data, weight)

        x_names = list(nx_graph.nodes)
        data['x_names'] = x_names

        # Convert pos in node feature and node class
        x_pos = [n[1] for n in x_names]
        if pos_encoder == 'default':
            pos_encoder = OneHotEncoder(categories=[_classes])

        x_pos = np.array(x_pos).reshape(-1, 1)
        x_sparse = pos_encoder.fit_transform(x_pos)
        x = torch.from_numpy(x_sparse.toarray())
        data['x'] = x
        return data
    
# ============================== Functions to convert graphs and attributes ==

def encode_pos(node_pos):
#     node_pos = [node[1] for node in self.G.nodes()]
    node_pos = np.array(node_pos).reshape(-1, 1)
    pos_encoder = OneHotEncoder(categories=[_classes])
    node_attr = pos_encoder.fit_transform(node_pos)
    return node_attr
