from caner.utils import *

if __name__ == '__main__':
    SOURCE_FOLDER = '../../data/FIND2019/FIND_train.split'
    TARGET_FOLDER = '../../data/FIND2019/FIND_train.target'
    VAL_SOURCE_FOLDER = '../../data/FIND2019/FIND_dev.split'
    VAL_TARGET_FOLDER = '../../data/FIND2019/FIND_dev.target'
    bert_path = '../../embedding/chinese_L-12_H-768_A-12'

    cluster_csv = ['../../data/FIND2019/FIND_cluster.csv']

    NER_FOLDER = 'ner_root'

    test_label_list = ['O', 'B-ORG', 'I-ORG']

    class_list = ['Utilities', 'RealEstate', 'Consumer', 'Industry', 'Materials',
                   'Financial', 'InformationTechnology', 'MedicalHealth', 'Telecom']
    for cls in class_list:
        test_label_list.append('ORG-' + cls)

    config = NetConfig(vec_type='bert', vec_path=bert_path,
                       label_list=test_label_list,
                       gpu_used=0.5, iter_num=100, embedding_size=768, dropout_keep_rate=0.5,
                       unit_num=256, seq_length=64, batch_size=64, learning_rate=0.001,
                       train_type='caner', feature_extractor='idcnn',
                       lambda_value=0.5, cluster_csv_list=cluster_csv)

    total_loss = train_model(config, NER_FOLDER, SOURCE_FOLDER, TARGET_FOLDER,
                             VAL_SOURCE_FOLDER, VAL_TARGET_FOLDER)
