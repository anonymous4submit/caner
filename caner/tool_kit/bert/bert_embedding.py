# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.python.estimator.estimator import Estimator
from tensorflow.python.estimator.run_config import RunConfig

from caner.tool_kit.bert import tokenization, modeling
from caner.tool_kit.bert.extract_features import model_fn_builder, convert_lst_to_features, PoolingStrategy


def is_valid_input(texts):
    return isinstance(texts, list) and all(isinstance(s, str) and s.strip() for s in texts)


class BertServer():
    def __init__(self, model_dir, gpu_used=0.5, max_seq_len=128, max_batch_size=128,
                 pooling_layer=-2, pooling_strategy=PoolingStrategy.NONE):
        """
        :param config: 配置类
        :param pooling_strategy: None时返回vec序列，分类一般用REDUCE_MEAN，详情见bert-as-service
        """
        assert max_seq_len <= 510, 'max_seq_len不能超过510！'

        self.model_dir = model_dir
        self.config_fp = os.path.join(self.model_dir, 'bert_config.json')
        self.checkpoint_fp = os.path.join(self.model_dir, 'bert_model.ckpt')
        self.vocab_fp = os.path.join(self.model_dir, 'vocab.txt')
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_fp)
        self.max_seq_len = max_seq_len+2  # +2用于容纳[CLS]和[SEP]
        self.max_batch_size = max_batch_size
        self.pooling_layer = pooling_layer
        self.pooling_strategy = pooling_strategy

        self.model_fn = model_fn_builder(
            bert_config=modeling.BertConfig.from_json_file(self.config_fp),
            init_checkpoint=self.checkpoint_fp,
            pooling_strategy=self.pooling_strategy,
            pooling_layer=self.pooling_layer
        )

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # if gpu_used is not None:
        #     config.gpu_options.per_process_gpu_memory_fraction = gpu_used
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        self.estimator = Estimator(self.model_fn, config=RunConfig(session_config=config))
        input_fn = self.input_fn_builder()
        self.predicitons = self.estimator.predict(input_fn, yield_single_examples=False)

        self.msg = None  # 这个全局变量很重要，是生成器的实时输入
        self.closed = False  # 关闭标志，使得生成器在析构函数执行后停止循环

    def __del__(self):
        print('Closed BERT Server ...')
        self.closed = True

    def input_fn_builder(self):
        def gen():
            while not self.closed:
                if is_valid_input(self.msg):
                    tmp_f = list(convert_lst_to_features(self.msg, self.max_seq_len, self.tokenizer))
                    yield {
                        'client_id': 'test',   # 占坑
                        'input_ids': [f.input_ids for f in tmp_f],
                        'input_mask': [f.input_mask for f in tmp_f],
                        'input_type_ids': [f.input_type_ids for f in tmp_f]
                    }

        def input_fn():
            return (tf.data.Dataset.from_generator(
                gen,
                output_types={'input_ids': tf.int32,
                              'input_mask': tf.int32,
                              'input_type_ids': tf.int32,
                              'client_id': tf.string},
                output_shapes={
                    'client_id': (),
                    'input_ids': (None, self.max_seq_len),
                    'input_mask': (None, self.max_seq_len),
                    'input_type_ids': (None, self.max_seq_len)}))

        return input_fn

    def get_bert_embedding(self, news_list):
        """
        传入新闻，用基于char的内置分词后再返回Embedding(序列)
        要保证该函数第一次执行前，图已经建好
        :param news:
        :return:
        """
        answer = []
        for left in range(0, len(news_list), self.max_batch_size):
            self.msg = news_list[left: left+self.max_batch_size]
            cur_pred = next(self.predicitons)['encodes']
            for line in cur_pred:
                answer.append(line)
        return answer
