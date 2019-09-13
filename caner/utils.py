import os
import logging
import pickle
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from caner.modeling import CanerModel
from caner.embedding import embedding_chooser
from caner.domain_pipeline import get_domain_dict_by_csv, get_domain_target_by_dict

from itertools import chain

logger = logging.getLogger(__name__)


class NetConfig:
    def __init__(self, vec_type='word2vec', vec_path=None, gpu_used=None, embedding_size=128, unit_num=256,
                 dropout_keep_rate=0.5, batch_size=32, seq_length=200, learning_rate=0.01, label_list=None,
                 iter_num=100, train_type='caner', feature_extractor='idcnn', self_attn=False,
                 lambda_value=0.25, mask_flag='O', domain_csv_list=None):
        """
        This class maintain the parameter configuration of the model, and can be persisted to JSON files
        :param vec_type: The type of word embedding，such as 'word2vec', 'fasttext'
        :param vec_path: The path of word embedding model, such as '.../xx.vec', '.../xx.bin'
        :param gpu_used: GPU occupancy rate, None if use CPU
        :param embedding_size: The dimension of word embedding
        :param unit_num: The dimension of LSTM unit num
        :param dropout_keep_rate: The dropout keep rate of model
        :param batch_size: The batch size in training
        :param seq_length: The max length of sentence
        :param learning_rate: The learning rate of optimizer in training
        :param label_list: List of all possible labels, such as: ['O','B-ORG','I-ORG']
        :param iter_num: The num of epochs
        :param train_type: The default type is 'caner'. Set it 'common' if you don't want domain adversarial.
        :param feature_extractor: bilstm or idcnn
        :param self_attn: wether use self attention after feature extractor
        """
        assert train_type in ['caner', 'common'], 'The train type must be "caner" or "common"'
        assert feature_extractor in ['bilstm', 'idcnn'], 'The feature extractor must be "bilstm" or "idcnn"'
        # if train_type == 'caner':
        #     assert domain_csv_list is not None, 'Domain adversarial training must have a list of domain csv'
        self.vec_type = vec_type
        self.vec_path = vec_path
        self.gpu_used = gpu_used
        self.embedding_size = embedding_size
        self.unit_num = unit_num
        self.dropout_keep_rate = dropout_keep_rate
        self.label_list = label_list
        self.output_size = None
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.lr = learning_rate
        self.iter_num = iter_num
        self.train_type = train_type
        self.feature_extractor = feature_extractor
        self.self_attn = self_attn
        self.lambda_value = lambda_value
        self.mask_flag = mask_flag
        self.domain_csv_list = domain_csv_list

    def save_config(self, ner_path):
        config_dict = {"vec_type": self.vec_type, "vec_path": self.vec_path,
                       "embedding_size": self.embedding_size, "unit_num": self.unit_num,
                       "label_list": self.label_list, "feature_extractor": self.feature_extractor}

        with open(os.path.join(ner_path, 'ner_config.json'), 'w', encoding='utf-8') as f:
            json.dump(config_dict, f)
            print('The list of persistence parameters is as follows: ')
            print(config_dict)

    def load_config(self, ner_path):
        with open(os.path.join(ner_path, 'ner_config.json'), 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
            print('Read the parameter list as follows: ')
            print(config_dict)
            try:
                self.vec_type = config_dict["vec_type"]
                self.vec_path = config_dict["vec_path"]
                self.embedding_size = config_dict["embedding_size"]
                self.unit_num = config_dict["unit_num"]
                self.label_list = config_dict["label_list"]
                self.output_size = len(self.label_list)
                self.feature_extractor = config_dict["feature_extractor"]
            except Exception as e:
                print(e)


"""
The following sections are functions for users to call.
If you want to train a model, pleause use train_model()
If you want to predict sentence:
   1. use load_model() to load checkpoint, pickle, .bin, and etc. to memory (## load only once ##)
   2. use predict_string() or predict_file() to predict sentences.

For example:
    caner_model = load_model(...)
    for sentence in sentence_list:
        predict_string('这是一段测试文本', caner_model) 

Please do not：
    for sentence in sentence_list:
        caner_model = load_model(...)
        predict_string('这是一段测试文本', caner_model) 
"""


def train_model(config, ner_folder, source_folder, target_folder,
                val_source_folder=None, val_target_folder=None):
    """
    Train the DANER model
    :param config: Config object of model
    :param ner_folder: Path to save model
    :param source_folder: The path of source files of training dataset
    :param target_folder: The path of BIO tagging files of training dataset
    :param val_source_folder: The path of source files of validation dataset
    :param val_target_folder: The path of BIO tagging files of validation dataset
    :return: The loss of each epoch during training (and validation)
    """
    config.save_config(ner_folder)
    model_path = os.path.join(ner_folder, 'model')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    onehot_model = OneHot()
    onehot_model.fit_label(config.label_list)
    with open(os.path.join(ner_folder, 'onehot_model.pkl'), 'wb') as f:
        pickle.dump(onehot_model, f, True)
    domain_dict = get_domain_dict_by_csv(config.domain_csv_list)

    # load data from folder
    source_list, target_list = _get_train_list(source_folder, target_folder)
    # truncate long sentences
    source_list = _truncate_sentences(source_list, max_len=config.seq_length)
    target_list = _truncate_sentences(target_list, max_len=config.seq_length)
    # get domain target list, for training caner
    domain_target_list = get_domain_target_by_dict(domain_dict, source_list, target_list)
    target_onehot_list = _convert_target_to_onehot(target_list, onehot_model, config.seq_length)
    classed_target_vector = _convert_target_to_onehot(domain_target_list, onehot_model, config.seq_length)
    target_mask = np.array([[0. if x == config.mask_flag else 1. for x in line]
                            for line in target_list], dtype='float32')
    # initialize caner network
    my_model = CanerModel(config)
    # initialize feature model (word embedding model)
    feature_model = embedding_chooser(config)
    if val_source_folder is not None and val_target_folder is not None:
        # load verification set
        val_source_string, val_target_string = _get_train_list(val_source_folder, val_target_folder)
        val_source_string = _truncate_sentences(val_source_string, max_len=config.seq_length)
        val_target_string = _truncate_sentences(val_target_string, max_len=config.seq_length)
        val_target_vector = _convert_target_to_onehot(val_target_string, onehot_model, config.seq_length)
        total_loss = my_model.train_and_eval(model_path, feature_model, source_list, target_onehot_list,
                                    classed_target_vector, target_mask, val_source_string, val_target_vector)
    else:
        total_loss = my_model.train_and_eval(model_path, feature_model,
                                    source_list, target_onehot_list, classed_target_vector, target_mask)
    return total_loss


def load_model(config, ner_folder):
    """
    Read the trained model to memory
    Note that this function should only be called once
    :param config: Config object of model
    :param ner_folder: Path to save model
    :return: caner model
    """

    my_model = CanerModel(config)
    model_path = os.path.join(ner_folder, 'model')
    with open(os.path.join(ner_folder, 'onehot_model.pkl'), 'rb') as f:
        my_model.nc.onehot_model = pickle.load(f)
    my_model.nc.feature = embedding_chooser(config)
    my_model.load_model_to_memory(model_path)
    return my_model


def predict_string(string, model):
    """
    Predict the BIO tagging of the string after a participle, such as '天元 国际 开业 啦'
    :param string: A string to be predicted
    :param model: The loaded caner model
    :return: The prediction results, a list
    """
    if len(string.split()) == 0:
        return []
    config = model.nc
    pred_list = [string.split()]
    pred_len = len(pred_list[0])
    pred_string = _truncate_sentences(pred_list, config.seq_length)
    pred_vector = model.predict(pred_string)
    pred_vector = list(chain.from_iterable(pred_vector))
    predict_label = list(config.onehot_model.decode_label(pred_vector))[:pred_len]
    return predict_label


def predict_file(file_name, model):
    """
    Predict the BIO tagging of the text in file
    :param file_name: The file to be predicted
    :param model: The loaded caner model
    :return: The prediction results, a list
    """
    output = []
    with open(file_name, 'r', encoding='utf-8') as f:
        sentences_list = f.readlines()
        for i, sentence in enumerate(sentences_list):
            output.append(predict_string(sentence, model))
            if i % 1000 == 0 and i > 0:
                print('Predicting file: %d / %d ...' % (i, len(sentences_list)))
    return output


"""
The following are private parts.
It mainly involves data processing, and ordinary users need not pay attention to this parts.
"""


class OneHot(object):
    """
    Responsible for the conversion between BIO tagging and One-Hot
    Make sure you use the same OneHot object for training and prediction
    So after training, please persist OneHot object in pickle
    """
    def __init__(self):
        self.__label_encoder = LabelEncoder()
        self.__onehot_encodeder = OneHotEncoder()

    def encode(self, target_list):
        integer_encoded = self.__label_encoder.fit_transform(np.array(target_list))
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        self.__onehot_encodeder = self.__onehot_encodeder.fit_transform(integer_encoded)
        return self.__onehot_encodeder.toarray()

    def fit_label(self, target_list):
        self.__label_encoder.fit(np.array(target_list))

    def encode_label(self, target_list):
        return self.__label_encoder.transform(np.array(target_list))

    def decode(self, encoder_list):
        return self.__label_encoder.inverse_transform([np.argmax(np.array(encoder_list), axis=1)])

    def decode_label(self, encoder_list):
        return self.__label_encoder.inverse_transform(encoder_list)


def _get_train_list(source_file, target_file):
    """
    Read all files in the source folder and target folder to memory
    :param source_file: the path of source files
    :param target_file: the path of target files
    :return: Parallel data sets
    """
    source_list = []
    target_list = []
    with open(source_file, 'r', encoding="utf-8") as source:
        source_array = source.readlines()
    with open(target_file, 'r', encoding="utf-8") as target:
        target_array = target.readlines()
    for source_line, target_line in zip(source_array, target_array):
        temp_source_list = source_line.split()
        temp_target_list = target_line.split()
        if len(temp_source_list) != len(temp_target_list):
            continue
        source_list.append(temp_source_list)
        target_list.append(temp_target_list)
    logger.info('Source data has been read out, total nums: ' + str(len(source_list)))
    return source_list, target_list


def _get_predict_list(source_file):
    """
    Read and convert the source file
    :param source_file: source sentence file
    :return: sentence list in source file
    """
    with open(source_file, 'r', encoding="utf-8") as source:
        source_list = []
        for s_line in source.readlines():
            source_list.append(s_line.split())
    return source_list


def _truncate_sentences(source_list, max_len):
    """
    truncate long sentences, keep each sentence no longer than max_len
    The redundant part of the sentence will start on another line,
    so choose the appropriate seqlen to avoid truncation as much as possible
    :param source_list: source sentence list
    :param max_len: max sequence length
    :return: truncated sentence list
    """
    new_source_list = []
    for source_line in source_list:
        left = 0
        right = max_len
        while left < len(source_line):
            new_source_list.append(source_line[left:right])
            left = right
            right += max_len
    return new_source_list


def _convert_target_to_onehot(target_list, onehot_obj, max_len):
    """
    Convert the BIO tagging sequences to One-hot sequences
    :param target_list: BIO tagging list
    :param onehot_obj: OneHot object
    :param max_len: max length of BIO tagging
    :return: he corresponding Onehot list
    """
    for i in range(0, len(target_list)):
        if len(target_list[i]) < max_len:
            target_list[i].extend(["O"] * (max_len - len(target_list[i])))
            if target_list[i] is None:
                target_list[i] = ["O"] * max_len
        else:
            if target_list[i] is None:
                target_list[i] = ["O"] * max_len
            else:
                target_list[i] = target_list[i][0:max_len]
    flat_list = [item for sublist in target_list for item in sublist]
    onehot_list = onehot_obj.encode_label(flat_list)
    onehot_list = onehot_list.reshape(-1, max_len)
    return onehot_list

