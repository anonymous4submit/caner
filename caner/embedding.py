import numpy as np
from gensim.models import KeyedVectors
from flair.data import Sentence
from flair.embeddings import FlairEmbeddings
from caner.tool_kit.bert.bert_embedding import BertServer
from caner.tool_kit.bert.extract_features import PoolingStrategy

from abc import ABCMeta, abstractmethod


def embedding_chooser(config):
    """
    Factory method, Decides which subclass should be instantiated
    """
    if config.vec_type == 'word2vec':
        return Word2vecEmbedding(config)
    elif config.vec_type == 'flair':
        return FlairEmbedding(config)
    elif config.vec_type == 'bert':
        return BertEmbedding(config)
    else:
        raise Exception('Make sure vec_type is "fasttext" or "word2vec"')


class Embedding(metaclass=ABCMeta):
    """
    Virtual base classes, defining several basic operations
    """
    def __init__(self, config):
        self.nc = config

    @abstractmethod
    def get_embedding(self, source_list):
        """
        All subclasses return word embedding by override this function
        :param source_list: Two-dimensional array of n*m (n sentences, m words per sentence)
        :return: An n*(m*e) two-dimensional array(n sentences, each sentence is the stitching of m vectors),
                 and a one-dimensional array of length n (the true length of n sentences)
        """
        return np.matrix([[]], dtype='float32'), np.array([], dtype='int32')


class Word2vecEmbedding(Embedding):
    """
    Subclass for generating word2vec embedding (gensim)
    """
    def __init__(self, config):
        Embedding.__init__(self, config)
        self.vec_model = KeyedVectors.load_word2vec_format(config.vec_path, binary=False)
        self.index2word_set = set(self.vec_model.wv.index2word)

    def get_embedding(self, source_list):
        assert self.nc.embedding_size == len(list(self.vec_model['.'])), \
            'The embedding_size is incorrect, please check it:  ' + str(len(list(self.vec_model['.']))) + \
            ' but ' + str(self.nc.embedding_size)
        row_vector_list = []
        len_list = []
        np.random.seed(233333)
        unknown_vec = np.random.randn(self.nc.embedding_size)
        for source_line in source_list:
            i = 0
            row_vector = []
            for source_word in source_line:
                if i < self.nc.seq_length:
                    if source_word in self.index2word_set:
                        row_vector = np.append(row_vector, self.vec_model[source_word])
                    else:
                        row_vector = np.append(row_vector, unknown_vec)
                i += 1
            if len(source_line) < self.nc.seq_length:
                row_vector = np.append(row_vector,
                                       np.zeros((self.nc.embedding_size * (self.nc.seq_length - len(source_line)),),
                                                dtype='float32'))
            row_vector_list.append(row_vector)
            len_list.append(min(len(source_line), self.nc.seq_length))

        return np.matrix(row_vector_list, dtype='float32'), np.array(len_list, dtype='int32')


class FlairEmbedding(Embedding):
    """
    Subclass for generating flair embedding
    Note the distinction with class "FlairEmbeddings" (Flair's own classes, class name have a 's')
    """
    def __init__(self, config):
        Embedding.__init__(self, config)
        self.vec_model = FlairEmbeddings(config.vec_path)

    def get_embedding(self, source_list):
        len_list = []
        sentence_list = [Sentence(' '.join(source_line)) for source_line in source_list]
        row_vector_list = []
        for s_id, sentence in enumerate(sentence_list):
            self.vec_model.embed(sentence)
            assert len(sentence) == len(source_list[s_id])
            row_vector = []
            for i, token in enumerate(sentence):
                if i < self.nc.seq_length:
                    row_vector = np.append(row_vector, np.array(token.embedding, dtype='float32'))
            len_list.append(min(len(source_list[s_id]), self.nc.seq_length))
            if len_list[s_id] < self.nc.seq_length:
                row_vector = np.append(row_vector,
                                       np.zeros((self.nc.embedding_size * (self.nc.seq_length - len_list[s_id]),),
                                                dtype='float32'))
            row_vector_list.append(row_vector)

        return np.matrix(row_vector_list, dtype='float32'), np.array(len_list, dtype='int32')


class BertEmbedding(Embedding):
    # 子类，用于返回bert embedding
    def __init__(self, config):
        Embedding.__init__(self, config)
        self.vec_model = BertServer(self.nc.vec_path, pooling_layer=-1, max_seq_len=self.nc.seq_length,
                                    pooling_strategy=PoolingStrategy.NONE, gpu_used=self.nc.gpu_used)

    def get_embedding(self, source_string):
        # 输入source/split ，返回bert embedding的矩阵

        len_list = [len(x) for x in source_string]
        text_list = [" ".join(x) for x in source_string]

        sentence_list = self.vec_model.get_bert_embedding(text_list)
        row_vector_list = []
        embedding_size = 768

        for i, sentence in enumerate(sentence_list):
            row_vector = []
            count = 0
            for word_emb in sentence[1:1+len_list[i]]:
                row_vector = np.append(row_vector, np.array(word_emb, dtype='float32'))
                count += 1
            if len_list[i] < self.nc.seq_length:
                row_vector = np.append(row_vector,
                                       np.zeros((embedding_size * (self.nc.seq_length - len_list[i]),),
                                                dtype='float32'))
            assert count == len_list[i], \
                'bert生成长度不匹配: ' + str(count) + '/' + str(len_list[i]) + '\n' + text_list[i]
            row_vector_list.append(row_vector)

        return np.matrix(row_vector_list, dtype='float32'), np.array(len_list, dtype='int32')
