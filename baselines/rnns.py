from __future__ import absolute_import
from __future__ import print_function
import re
import tarfile
import nltk
import pickle
import numpy as np
import functools
np.random.seed(1337)  # for reproducibility

import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers

# from keras.datasets.data_utils import get_file
# from keras.layers.embeddings import Embedding
# from keras.layers.core import Dense
# from keras.layers import recurrent, Concatenate
# from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences

'''
Trains two recurrent neural networks based upon a story and a question.
The resulting merged vector is then queried to answer a range of bAbI tasks.
The results are comparable to those for an LSTM model provided in Weston et al.:
"Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks"
http://arxiv.org/abs/1502.05698
For the resources related to the bAbI project, refer to:
https://research.facebook.com/researchers/1543934539189348
Notes:
- With default word, sentence, and query vector sizes, the GRU model achieves:
  - 52.1% test accuracy on QA1 in 20 epochs (2 seconds per epoch on CPU)
  - 37.0% test accuracy on QA2 in 20 epochs (16 seconds per epoch on CPU)
In comparison, the Facebook paper achieves 50% and 20% for the LSTM baseline.
- The task does not traditionally parse the question separately. This likely
improves accuracy and is a good example of merging two RNNs.
- The word vector embeddings are not shared between the story and question RNNs.
- See how the accuracy changes given 10,000 training samples (en-10k) instead
of only 1000. 1000 was used in order to be comparable to the original paper.
- Experiment with GRU, LSTM, and JZS1-3 as they give subtly different results.
- The length and noise (i.e. 'useless' story components) impact the ability for
LSTMs / GRUs to provide the correct answer. Given only the supporting facts,
these RNNs can achieve 100% accuracy on many tasks. Memory networks and neural
networks that use attentional processes can efficiently search through this
noise to find the relevant statements, improving performance substantially.
This becomes especially obvious on QA2 and QA3, both far longer than QA1.
'''


# def tokenize(sent):
#     '''Return the tokens of a sentence including punctuation.
#     >>> tokenize('Bob dropped the apple. Where is the apple?')
#     ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
#     '''
#     return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def tokenize(text):
    return nltk.word_tokenize(text.lower())


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(lines, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(lines, only_supporting=only_supporting)
    flatten = lambda data: functools.reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen, vocab_size):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(vocab_size)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


def vectorize(text, word_idx, maxlen=None):
    vets = [word_idx[w] for w in tokenize(text)]
    if maxlen is not None:
        return pad_sequences([vets], maxlen=maxlen)
    else:
        return vets


# def vector2sent(vectors):
#     text = []
#     for vec in vectors:
#         text.append(word_idx.)
#     return text

def construct_GRU(vocab_size, story_maxlen, query_maxlen, rnn_layer=tf.keras.layers.GRU, bidirectional=False, story_embedding_dim=64, question_embedding_dim=64,
                  story_units=128, question_units=128, verbose=False):
    '''
    构建模型
    :param vocab_size:
    :param rnn_layer:
    :param bidirectional:
    :param story_embedding_dim:
    :param question_embedding_dim:
    :param story_units:
    :param question_units:
    :return:
    '''

    # RNN = tf.keras.layers.GRU
    # Bidirectional = False
    # Story_embedding_dim = 64
    # Question_embedding_dim = 64
    # Story_units = 100
    # Question_units = 100

    if verbose: print('RNN / Embed / Story / Question = {}, {}, {}, {}'.format(rnn_layer, story_embedding_dim, question_embedding_dim,
                                                               story_units))
    story_input = tf.keras.layers.Input(shape=(story_maxlen,))
    story_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=story_embedding_dim, mask_zero=True)(
        story_input)
    # story_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    query_input = tf.keras.layers.Input(shape=(query_maxlen,))
    query_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=question_embedding_dim, mask_zero=True)(
        query_input
    )
    # query_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    recurr1 = rnn_layer(units=story_units, return_sequences=False, go_backwards=bidirectional)(story_embeded)
    recurr2 = rnn_layer(units=question_units, return_sequences=False, go_backwards=bidirectional)(query_embeded)
    conc = tf.keras.layers.Concatenate(axis=1)([recurr1, recurr2])

    output = tf.keras.layers.Dense(units=vocab_size, activation='softmax')(conc)
    model = tf.keras.Model([story_input, query_input], output)
    if verbose: model.summary()
    model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()], optimizer='adam') #
    return model

def construct_BiGRU(vocab_size, story_maxlen, query_maxlen, rnn_layer=tf.keras.layers.GRU, bidirectional=True, story_embedding_dim=64, question_embedding_dim=64,
                  story_units=128, question_units=128, verbose=False):
    '''
    构建模型
    :param vocab_size:
    :param rnn_layer:
    :param bidirectional:
    :param story_embedding_dim:
    :param question_embedding_dim:
    :param story_units:
    :param question_units:
    :return:
    '''

    # RNN = tf.keras.layers.GRU
    # Bidirectional = False
    # Story_embedding_dim = 64
    # Question_embedding_dim = 64
    # Story_units = 100
    # Question_units = 100

    if verbose: print('RNN / Embed / Story / Question = {}, {}, {}, {}'.format(rnn_layer, story_embedding_dim, question_embedding_dim,
                                                               story_units))
    story_input = tf.keras.layers.Input(shape=(story_maxlen,))
    story_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=story_embedding_dim, mask_zero=True)(
        story_input)
    # story_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    query_input = tf.keras.layers.Input(shape=(query_maxlen,))
    query_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=question_embedding_dim, mask_zero=True)(
        query_input
    )
    # query_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    recurr1 = rnn_layer(units=story_units, return_sequences=False, go_backwards=bidirectional)(story_embeded)
    recurr2 = rnn_layer(units=question_units, return_sequences=False, go_backwards=bidirectional)(query_embeded)
    conc = tf.keras.layers.Concatenate(axis=1)([recurr1, recurr2])

    output = tf.keras.layers.Dense(units=vocab_size, activation='softmax')(conc)
    model = tf.keras.Model([story_input, query_input], output)
    if verbose: model.summary()
    model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()], optimizer='adam') #
    return model


def readlinesfromfile(dir, task_id):
    '''
    从任务i文件中读取行
    :param dir:
    :param task_id:
    :return: train_lines, test_lines
    '''
    import glob
    train_data_path = glob.glob('%s/qa%d_*_train.txt' % (data_dir, task_id))[0]
    test_data_path = glob.glob('%s/qa%d_*_test.txt' % (data_dir, task_id))[0]
    print('Task: ', task_id)
    print('Reading data from ', train_data_path)
    with open(train_data_path, 'r') as infile:
        train_lines = infile.readlines()
    with open(test_data_path, 'r') as infile:
        test_lines = infile.readlines()
    return train_lines, test_lines


def preprocess(train_lines, test_lines):
    '''
    处理为训练数据
    :param train_lines:
    :param test_lines:
    :return: （字典，_story_maxlen, _query_maxlen），（训练集，测试集）<-([story,question],label)
    '''
    # # Default QA1 with 1000 samples
    # # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # # QA1 with 10,000 samples
    # # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # # QA2 with 1000 samples
    # challenge = data_dir + '/qa1_single-supporting-fact_{}.txt'
    # # QA2 with 10,000 samples
    # # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
    _train = get_stories(train_lines)
    _test = get_stories(test_lines)

    vocab = sorted(
        functools.reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in _train + _test)))
    # Reserve 0 for masking via pad_sequences
    _vocab_size = len(vocab) + 1
    _word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    _story_maxlen = max(map(len, (x for x, _, _ in _train + _test)))
    _query_maxlen = max(map(len, (x for _, x, _ in _train + _test)))

    story_train, question_train, answer_train = vectorize_stories(_train, word_idx=_word_idx, story_maxlen=_story_maxlen, query_maxlen=_query_maxlen, vocab_size=_vocab_size)
    story_test, question_test, answer_test = vectorize_stories(_test, word_idx=_word_idx, story_maxlen=_story_maxlen, query_maxlen=_query_maxlen, vocab_size=_vocab_size)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(story_train.shape))
    print('Xq.shape = {}'.format(question_train.shape))
    print('Y.shape = {}'.format(answer_train.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(_story_maxlen, _query_maxlen))

    pickle.dump(
        (_word_idx, _story_maxlen, _query_maxlen),
        open(model_dir + 'word_idx.pkl', 'wb')
    )

    return (_word_idx, _story_maxlen, _query_maxlen), (
    ([story_train, question_train], answer_train), ([story_test, question_test], answer_test))


def train(model, training_data, testing_data, batch_size=512, epochs=60):
    '''
    执行模型训练
    :param model:compile过的模型
    :param training_data:训练集，结构：([story,question],label)
    :param testing_data:测试集，结构：([story,question],label)
    :param batch_size:
    :param epochs:
    :return:训练好的模型
    '''
    # sentrnn = keras.Sequential()
    # sentrnn.add(Embedding(input_dim=vocab_size, output_dim=EMBED_HIDDEN_SIZE, mask_zero=True,
    #                       input_shape=(story_maxlen, )))
    # sentrnn.add(RNN(units=SENT_HIDDEN_SIZE, return_sequences=False))
    #
    # qrnn = keras.Sequential()
    # qrnn.add(Embedding(input_dim=vocab_size, output_dim=EMBED_HIDDEN_SIZE,
    #                    input_shape=(query_maxlen, )))
    # # qrnn.add(RNN(EMBED_HIDDEN_SIZE, QUERY_HIDDEN_SIZE, return_sequences=False))
    # qrnn.add(RNN(QUERY_HIDDEN_SIZE, return_sequences=False))
    #
    # model = Sequential()
    # merged = Concatenate([sentrnn, qrnn])
    # model.add(Dense(vocab_size, activation='softmax'))(merged)
    #
    # model.summary()
    # model.compile(optimizer='adam', loss='categorical_crossentropy', class_mode='categorical')
    #
    # print('Training')
    # model.fit([X, Xq], Y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS, validation_split=0.05, show_accuracy=True)
    # loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE, show_accuracy=True)
    # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    print('Training...')
    hist = model.fit(training_data[0], training_data[1], batch_size=batch_size, epochs=epochs, validation_split=0.05,
              verbose=2, shuffle=True)
    print(hist.history)
    #model.fit([X, Xq], Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05, verbose=2)

    #loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE, verbose=1)
    # loss, acc = model.evaluate(testing_data[0], testing_data[1], batch_size=batch_size, verbose=1)
    # print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    # model.save(model_dir)
    # print('Model saved...')

    return model



def predict_d():
    word_idx = pickle.load(
        open(model_dir + 'word_idx.pkl', 'rb'))
    import glob
    import numpy as np

    from demo.qa import MemN2N
    from util import parse_babi_task
    """ Initialize app """
    global memn2n, test_story, test_questions, test_qstory
    model_file = 'trained_model/memn2n_model.pklz'

    # Try to load model
    memn2n = MemN2N(data_dir, model_file)
    memn2n.load_model()
    test_story, test_questions, test_qstory = None, None, None
    # Read test data
    print("Reading test data from %s ..." % memn2n.data_dir)
    test_data_path = glob.glob('%s/qa*_*_test.txt' % memn2n.data_dir)
    test_story, test_questions, test_qstory = \
        parse_babi_task(test_data_path, memn2n.general_config.dictionary, False)
    question_idx = np.random.randint(test_questions.shape[1])
    story_idx = test_questions[0, question_idx]
    last_sentence_idx = test_questions[1, question_idx]

    story_txt, question_txt, correct_answer = memn2n.get_story_texts(test_story, test_questions, test_qstory,
                                                                     question_idx, story_idx, last_sentence_idx)
    # Format text
    story_txt = "\n".join(story_txt)
    question_txt += "?"

    print('Case study:')
    print('Story:', story_txt, '\nQuestion:', question_txt, '\nAnswer:', correct_answer)

    model = tf.keras.models.load_model(model_dir)
    predict_answer = model.predict([vectorize(story_txt, word_idx, 552), vectorize(question_txt, word_idx, 5)])
    print(list(word_idx.keys())[np.argmax(predict_answer, axis=1)[0]-1])


def predict(model, story, question):
    '''
    用模型进行预测
    :param model: 模型路径或（模型，字典，故事最长长度，问题最长长度）
    :param story:
    :param question:
    :return: 预测结果
    '''
    if isinstance(model, str):
        print('============\n Loading model:')
        _word_idx, _story_maxlen, _question_maxlen = pickle.load(
            open(model + 'word_idx.pkl', 'rb'))
        _model = tf.keras.models.load_model(model)
    else:
        _model = model[0]
        _word_idx = model[1]
        _story_maxlen = model[2]
        _question_maxlen = model[3]
    _predict_answer = _model.predict([vectorize(story, _word_idx, _story_maxlen),
                                      vectorize(question, _word_idx, _question_maxlen)])
    return list(_word_idx.keys())[np.argmax(_predict_answer, axis=1)[0]]


def execute_training(directory, construct_model, task_ids=None, batch_size=256, epochs=30):
    '''
    coach
    :return:
    '''
    import tensorflow as tf
    from keras import backend as K

    K.clear_session()
    tf.reset_default_graph()

    evluations = []
    if task_ids is None:
        '''jointly train'''
        ltrain = []
        ltest = []
        for i in range(1, 21):
            ltr, lte = readlinesfromfile(directory, i)
            ltrain.extend(ltr)
            ltest.extend(lte)
        (_dict, _story_maxlen, _query_maxlen), data = preprocess(ltrain, ltest)
        _model = construct_model(len(_dict)+1, verbose=False)
        _model = train(_model, data[0], data[1], batch_size=batch_size, epochs=epochs)

        loss, acc = _model.evaluate(data[1][0], data[1][1], batch_size=batch_size, verbose=1)
        print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
        evluations.append(acc)

    else:
        for task_id in task_ids:
            ltrain, ltest = readlinesfromfile(directory, task_id)
            (_dict, _story_maxlen, _query_maxlen), data = preprocess(ltrain, ltest)
            _model = construct_model(len(_dict) + 1, _story_maxlen, _query_maxlen, verbose=False)
            # attention: len(dict) doesn't include padding word 0, thus have to add 1
            _model = train(_model, data[0], data[1], batch_size=batch_size, epochs=epochs)

            loss, acc = _model.evaluate(data[1][0], data[1][1], batch_size=batch_size, verbose=1)
            print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
            evluations.append(acc)
    return evluations


def save_to_csv(_task_ids, _result, save_path):
    import time
    print(time.time())
    print(time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time())))
    ts = time.strftime('%m-%d-%H-%M-%S', time.localtime(time.time()))
    import pandas as pd
    df = pd.DataFrame(_task_ids)
    df['1'] = _result
    df.to_csv(save_path+ts+'.csv', index=False, header=False, sep=',', encoding='utf-8')


if __name__ == '__main__':
    data_dir = 'data/tasks_1-20_v1-2/en-10k'
    model_dir = 'model/gru'
    story_txt = 'John went to the bedroom. Mary journeyed to the bathroom.'
    query_txt = 'Where is John?'
    # tmodel, tdict, story_maxlen, query_maxlen = train()
    # print('train model:')
    # predict_answer = tmodel.predict([vectorize(story_txt, tdict, story_maxlen), vectorize(query_txt, tdict, query_maxlen)])
    # print(list(tdict.keys())[np.argmax(predict_answer, axis=1)[0]])

    # predict()

    times = 5
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en'
        result = []
        task_ids = list(range(1, 21))
        result = execute_training(data_dir, construct_GRU, task_ids, batch_size=128, epochs=30)
        print(result)
        save_to_csv(task_ids, result, './result/gru')
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en-10k'
        result = []
        task_ids = list(range(1, 21))
        result = execute_training(data_dir, construct_GRU, task_ids, batch_size=512, epochs=20)
        print(result)
        save_to_csv(task_ids, result, './result/10k-gru')
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en'
        result = []
        task_ids = list(range(1, 21))
        result = execute_training(data_dir, construct_BiGRU, task_ids, batch_size=128, epochs=30)
        print(result)
        save_to_csv(task_ids, result, './result/bigru')
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en-10k'
        result = []
        task_ids = list(range(1, 21))
        result = execute_training(data_dir, construct_BiGRU, task_ids, batch_size=512, epochs=20)
        print(result)
        save_to_csv(task_ids, result, './result/10k-bigru')
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en'
        result = []
        result = execute_training(data_dir, construct_GRU, batch_size=128, epochs=30)
        print(result)
        save_to_csv(task_ids, result, './result/joint-gru')
    for i in range(times):
        data_dir = 'data/tasks_1-20_v1-2/en-10k'
        result = []
        result = execute_training(data_dir, construct_GRU, batch_size=512, epochs=20)
        print(result)
        save_to_csv(task_ids, result, './result/joint-10k-gru')

