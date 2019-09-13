from caner.utils import *

if __name__ == '__main__':
    SOURCE_FOLDER = '../../data/Lattice/Lattice_train.split'
    TARGET_FOLDER = '../../data/Lattice/Lattice_train.target'
    VAL_SOURCE_FOLDER = '../../data/Lattice/Lattice_dev.split'
    VAL_TARGET_FOLDER = '../../data/Lattice/Lattice_dev.target'
    bert_path = '../../embedding/chinese_L-12_H-768_A-12'

    domain_csv = ['../../data/Lattice/LOC_domain.csv',
                  '../../data/Lattice/PER_domain.csv',
                  '../../data/Lattice/ORG_domain.csv']

    NER_FOLDER = 'ner_root'

    test_label_list = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
    class_list = ['A','B','C','D']
    for cls in class_list:
        test_label_list.append('ORG-' + cls)
        test_label_list.append('PER-' + cls)
        test_label_list.append('LOC-' + cls)

    config = NetConfig(vec_type='bert', vec_path=bert_path,
                       label_list=test_label_list,
                       gpu_used=0.5, iter_num=100, embedding_size=768, dropout_keep_rate=0.5,
                       unit_num=256, seq_length=64, batch_size=64, learning_rate=0.001,
                       train_type='caner', feature_extractor='idcnn',
                       lambda_value=0.5, domain_csv_list=domain_csv)

    total_loss = train_model(config, NER_FOLDER, SOURCE_FOLDER, TARGET_FOLDER,
                             VAL_SOURCE_FOLDER, VAL_TARGET_FOLDER)
