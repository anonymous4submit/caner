from caner.utils import *
from caner.metrics import get_ner_metrics

if __name__ == '__main__':
    SOURCE_FILE = '../../data/FIND/FIND_test.split'
    REAL_FILE = '../../data/FIND/FIND_test.target'
    ANSWER_FILE = '../../data/FIND/answer.target'
    NER_FOLDER = 'ner_root'

    # load model
    config = NetConfig()
    config.load_config(NER_FOLDER)
    my_model = load_model(config, NER_FOLDER)
    # predict and write results
    ans = predict_file(SOURCE_FILE, my_model)
    write_text = ''
    for i, line in enumerate(ans):
        if i > 0:
            write_text += '\n'
        write_text += " ".join(line)
    with open(ANSWER_FILE, 'w', encoding='utf-8') as f:
        f.write(write_text)

    # evaluate Prediction results
    get_ner_metrics(SOURCE_FILE, REAL_FILE, ANSWER_FILE)