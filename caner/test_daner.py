import unittest
from caner.utils import *
from caner.domain_pipeline import get_domain_list
from data import DATA_DIR, ORG_A_CSV

NER_ROOT = os.path.join(DATA_DIR, "test_data", "ner_root")
ORG_SOURCE_FILE = os.path.join(DATA_DIR, "test_data", "split_02.txt")
ORG_TARGET_FILE = os.path.join(DATA_DIR, "test_data", "targetH_02.txt")
FLAIR_LM_FILE = os.path.join(DATA_DIR, "best-lm.pt")


class TestCaner(unittest.TestCase):

    def setUp(self):
        self.word2vec_path = os.path.join(DATA_DIR, "test_data", "kd.vec")
        self.label_list = ['O', 'B-ORG', 'I-ORG']
        class_list = get_domain_list()
        for cls in class_list:
            self.label_list.append('ORG-'+cls)
            # self.label_list.append('ORG-'+cls)

    def test_train(self):
        # config = NetConfig(vec_type='bert', vec_path='/Users/chenyuanzhe/PycharmProjects/bert/chinese_L-12_H-768_A-12',
        #                    label_list=self.label_list, seq_length=10, batch_size=2, embedding_size=768, unit_num=100,
        #                    iter_num=10, train_type='caner', feature_extractor='idcnn', self_attn=True)
        config = NetConfig(vec_type='word2vec', vec_path=self.word2vec_path,
                           label_list=self.label_list, seq_length=10, batch_size=2, embedding_size=50, unit_num=64,
                           iter_num=10, train_type='caner', feature_extractor='idcnn',
                           domain_csv_list=[ORG_A_CSV])
        total_loss = train_model(config, NER_ROOT, ORG_SOURCE_FILE, ORG_TARGET_FILE,
                                 ORG_SOURCE_FILE, ORG_TARGET_FILE)
        print(total_loss)

    def test_predict(self):
        config = NetConfig()
        config.load_config(NER_ROOT)
        my_model = load_model(config, NER_ROOT)
        # for i in range(1):
        #     print(predict_file(ORG_SOURCE_FILE, my_model))
        print(predict_string('做个 测试 。', my_model))


if __name__ == '__main__':
    unittest.main()
