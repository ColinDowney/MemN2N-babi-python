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

#def construct_GRU():



def train():
    RNN = tf.keras.layers.GRU
    Bidirectional = False
    Story_embedding_dim = 64
    Question_embedding_dim = 64
    Story_units = 100
    Question_units = 100
    BATCH_SIZE = 256
    EPOCHS = 30
    print('RNN / Embed / Story / Question = {}, {}, {}, {}'.format(RNN, Story_embedding_dim, Question_embedding_dim,
                                                               Story_units))

    # Default QA1 with 1000 samples
    # challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
    # QA1 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
    # QA2 with 1000 samples
    challenge = data_dir + '/qa1_single-supporting-fact_{}.txt'
    # QA2 with 10,000 samples
    # challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

    with open(challenge.format('train'), 'r') as infile:
        lines = infile.readlines()
    train = get_stories(lines)
    with open(challenge.format('test'), 'r') as infile:
        lines = infile.readlines()
    test = get_stories(lines)

    vocab = sorted(
        functools.reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
    # Reserve 0 for masking via pad_sequences
    vocab_size = len(vocab) + 1
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))

    X, Xq, Y = vectorize_stories(train, word_idx=word_idx, story_maxlen=story_maxlen, query_maxlen=query_maxlen, vocab_size=vocab_size)
    tX, tXq, tY = vectorize_stories(test, word_idx=word_idx, story_maxlen=story_maxlen, query_maxlen=query_maxlen, vocab_size=vocab_size)

    print('vocab = {}'.format(vocab))
    print('X.shape = {}'.format(X.shape))
    print('Xq.shape = {}'.format(Xq.shape))
    print('Y.shape = {}'.format(Y.shape))
    print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

    print('Build model...')

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

    story_input = tf.keras.layers.Input(shape=(story_maxlen,))
    story_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=Story_embedding_dim, mask_zero=True)(
        story_input)
    # story_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    query_input = tf.keras.layers.Input(shape=(query_maxlen,))
    query_embeded = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=Question_embedding_dim, mask_zero=True)(
        query_input
    )
    # query_model = tf.keras.layers.GRU(units=128, return_sequences=False)

    recurr1 = tf.keras.layers.GRU(units=Story_units, return_sequences=False, go_backwards=Bidirectional)(story_embeded)
    recurr2 = tf.keras.layers.GRU(units=Question_units, return_sequences=False, go_backwards=Bidirectional)(query_embeded)
    conc = tf.keras.layers.Concatenate(axis=1)([recurr1, recurr2])

    output = tf.keras.layers.Dense(units=vocab_size, activation='softmax')(conc)
    model = tf.keras.Model([story_input, query_input], output)
    model.summary()
    model.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.CategoricalAccuracy()]) # optimizer='adam',

    print('Training')
    model.fit([X, Xq], Y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.05, verbose=2)

    loss, acc = model.evaluate([tX, tXq], tY, batch_size=BATCH_SIZE, verbose=1)
    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

    model.save(model_dir)

    pickle.dump(
        (word_idx, story_maxlen, query_maxlen),
        open(model_dir + 'word_idx.pkl', 'wb')
    )
    print('Saved')
    return model, word_idx, story_maxlen, query_maxlen


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


def predict(model_path, story, question):
    print('============\n Loading model:')
    _word_idx, _story_maxlen, _question_maxlen = pickle.load(
        open(model_path + 'word_idx.pkl', 'rb'))
    _model = tf.keras.models.load_model(model_path)
    _predict_answer = _model.predict([vectorize(story, _word_idx, _story_maxlen),
                                      vectorize(question, _word_idx, _question_maxlen)])
    return list(_word_idx.keys())[np.argmax(_predict_answer, axis=1)[0]]

if __name__ == '__main__':
    data_dir = 'data/tasks_1-20_v1-2/en'
    model_dir = 'model/gru'
    story_txt = 'John went to the bedroom. Mary journeyed to the bathroom.'
    query_txt = 'Where is John?'
    tmodel, tdict, story_maxlen, query_maxlen = train()
    print('train model:')
    predict_answer = tmodel.predict([vectorize(story_txt, tdict, story_maxlen), vectorize(query_txt, tdict, query_maxlen)])
    print(list(tdict.keys())[np.argmax(predict_answer, axis=1)[0]])

    # predict()


