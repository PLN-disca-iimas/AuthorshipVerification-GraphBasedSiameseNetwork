import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk import FreqDist

from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

# ==================== Feature extraction ====================

class FeatureExtractor:
    def __init__(self, parsed):
        self.parsed = parsed

    # Obtener palabras de texto parseado
    @staticmethod
    def get_words(parsed_text: list):
        """ Método para obtener las palabras de texto parseado
        :param parsed_text: Lista de tuplas ('word', 'POS')
        :return: np.ndarray con palabras
        """
        words = np.asarray([pair[0] for pair in parsed_text])

        return words

    # Frequency of function words
    @staticmethod
    def func_words_freq(text_list, lang='english'):
        """ Frequencies of 179 stopwords defined in the NLTK corpus package.

        :param text_list: Lista con tokens de un documento
        :param lang: Idioma de las stopwords
        :return: Vector con las frecuencias, diccionario de frecuencias
        """
        global func_words
        func_words = np.asarray(stopwords.words(lang))
        dictionary = {func_words[i]: 0. for i in range(len(func_words))}

        for word in text_list:
            if word in func_words:
                dictionary[word] += 1

        vec_freq = np.asarray(list(dictionary.values()), dtype=np.double)

        return vec_freq

    # Average number of characters per word
    @staticmethod
    def average_char_word(text_list):
        """ Promedio de caracteres por palabra

        :param text_list: Lista con tokens de un documento
        :return: np.ndarray(letras totales / palabras totales)
        """
        average = np.asarray(len(''.join(text_list)) / len(text_list))

        return average

    # Vocab Richness: The ratio of hapax-legomenon and dis-legomenon
    @staticmethod
    def vocab_richness(text_list):
        """ The ratio of hapax-legomenon and dis-legomenon.

        (Divided by the number of tokens in the document to account for documents of varying lengths)
        :param text_list: Lista de palabras
        :return: The ratio of hapax-legomenon and dis-legomenon
        """
        freq = FreqDist(word for word in text_list)
        hapax = np.asarray([key for key, value in freq.items() if value == 1])
        dis = np.asarray([key for key, value in freq.items() if value == 2])

        if len(dis) == 0 or len(text_list) == 0:
            return np.asarray(0)
        else:
            return np.asarray((len(hapax) / len(dis)) / len(text_list))

    @staticmethod
    def tf_idf_chars(text_list): # No funciona!
        """ Método para extraer un vector conlos valores de
            tf-idf para caracteres en un rango de 1<=n<=6
        """
        tfidf_word = TfidfVectorizer() # Creo que eso no se usa para nada
        X_tfidf_word = tfidf_word.fit_transform(text_list) # Creo que no se usa
        tfidf_char = TfidfVectorizer(analyzer='char', ngram_range=(1, 6), lowercase=False)
        X_tfidf_char = tfidf_char.fit_transform(text_list)
        X_tfidf = sparse.hstack([X_tfidf_word, X_tfidf_char]) # Creo que no se usa
        vector_tf = sparse.hstack(tfidf_char) # Por acá anda el bug
        return vector_tf

    @staticmethod
    def dist_word(text_list):
        """ Método para extraer la distribucion de frecuencia en el
            largo de las palabra 'l' con 1<=l<=n
        """
        lst = []
        for n in range(1, 11):
            aux = pd.DataFrame([FreqDist(len(word)
                                         for word in text_list
                                         if len(word) == n)])
            if aux.empty:
                lst.append(0)
            else:
                lst.append(aux.iloc[0, 0] / len(text_list))
        return lst

    # Extractor principal
    def extract_features(self, text_list):
        """ Método para extraer un vector con todas las características

        Características:
        - Frequency of function words
        - Average number of characters per word
        - Vocab richness
        - Char Dist
        - tf-idf

        :param text_list: Texto (pre-procesado) de entrada
        :return: Vector de características
        """
        words = self.get_words(text_list) if self.parsed else text_list

        vec_freq = self.func_words_freq(words)
        average_ch_per_word = self.average_char_word(words)
        voc_rich = self.vocab_richness(words)
        char_dist = self.dist_word(words)
#         tf_idf_char = self.tf_idf_chars(words)

#         vector_features = np.hstack((vec_freq, average_ch_per_word, voc_rich,
#         char_dist, tf_idf_char))
        vector_features = np.hstack((vec_freq, average_ch_per_word, voc_rich, char_dist))

        return vector_features


# =============== Para transformar en DataFrame ===============


def to_matrix(vec_features_all):
    """ Función para convertir vector de vectores (de características) en un matrix.

    Cada renglón de esta matrix será una columna para el data frame

    :param vec_features_all: Vector de vectores de características
    :return: matrix de características (cada renglón es un vector de características de un texto)
    """
    matrix_columns = tuple([np.expand_dims(vector, axis=0) for vector in vec_features_all])
    matrix_features = np.concatenate(matrix_columns, axis=0)

    return matrix_features


def get_dataframe(ids, matrix):
    """ Función para transformar las ids + features a Dataframe

    :param ids: Lista con las ids de los autores
    :param matrix: Matriz de características estilísticas
    :return: dataframe; renglones: ids+features de un autor; columnas: features de todos los textos
    """

    ids_column = np.expand_dims(ids, axis=0).T
    all_data = np.concatenate((ids_column, matrix), axis=1)

    f_ids = ['ids']
    f_freq = [f'freq_func_word_{i}' for i in range(1, 180)]
    f_average_num_word = ['Average num char/word']
    f_vocab = ['vocab-richness']
    f_char_dis = [f'char_dist_{i}' for i in range(1, 10)]
    f_tf_idf = ['ft-idf']
    feature_name = np.asarray(f_ids + f_freq + f_average_num_word + f_vocab + f_char_dis + f_tf_idf)

    features_dataframe = pd.DataFrame({feature_name[i]: all_data[:, i] for i in range(len(feature_name))})

    return features_dataframe

# ========================== Pruebas ==========================


def experiment():
    from text_to_graph import TextParser

    # Cargamos todos los datos
    corpus = load_text()

    ids = corpus.loc[:, 'id']
    texts = corpus.loc[:, 'text']

    # Pre-procesado y parseado
    parser = TextParser()
    parsed_texts = np.asarray([parser.define_tagged(text) for text in texts], dtype=object)

    # Extración de características
    extractor = FeatureExtractor(parsed=True)
    vec_features_all = np.asarray([extractor.extract_features(parsed_text) for parsed_text in parsed_texts])
    matrix_features = to_matrix(vec_features_all)

    features_dataframe = get_dataframe(ids, matrix_features)

    return features_dataframe


def load_text():
    from jsonlines import open
    from random import randint

    MAX = 0

    path_to_text = '../PAN21_predict/pairs.jsonl'

    ids_texts = {'ids': [], 'pair': []}

    with open(path_to_text) as f:
        i = 0
        for line in f.iter():
            if i <= MAX:
                ids_texts['ids'].append(line['id'])
                ids_texts['pair'].append(line['pair'][randint(0, 1)])
                i += 1
            else:
                break

    ids_texts_df = pd.DataFrame({'id': ids_texts['ids'], 'text': ids_texts['pair']})

    return ids_texts_df


def tf_idf_features(text_dict, special_chars, max_features):
    corpus = text_dict.values()
#     vectorizer_sc = \
#         TfidfVectorizer(vocabulary=special_chars,
#                         analyzer='char')
#     tfidf_sc = vectorizer_sc.fit_transform(corpus)
# 
    vectorizer_ngrams = \
        TfidfVectorizer(analyzer='char', ngram_range=(1, 6), min_df=0.1,
                        max_features=max_features)
    tfidf_ngrams = vectorizer_ngrams.fit_transform(corpus)

    tfidf_features = np.concatenate([tfidf_sc, tfidf_features])
    return tfidf_feat


if __name__ == '__main__':
    print(experiment())
