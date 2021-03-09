import PySimpleGUI as sg
import glob
import numpy as np

from demo.qa import MemN2N
from util import parse_babi_task

sg.theme('DarkAmber')  # Add a touch of color
# All the stuff inside your window.
layout = [[sg.Text('babi数据集算法演示')],
          [sg.Text('故事文本：')],
          [sg.Output(size=(40, 10), key='story')],
          [sg.Text('问题：'), sg.Input(size=(30, 2), key='question')],
          [sg.Text('回答'), sg.Output(size=(30, 2), key='answer')],
          [sg.Button('随机获取故事', key='getstory'), sg.Button('输出答案', key='getanswer')],
          [sg.Button('模型一'), sg.Button('模型二')]
          ]

""" Initialize app """
global memn2n, test_story, test_questions, test_qstory
data_dir = 'data/tasks_1-20_v1-2/en'
model_file = 'trained_model/memn2n_model.pklz'

memn2n = None
test_story, test_questions, test_qstory = None, None, None


if __name__ == '__main__':
    # Create the Window
    window = sg.Window('babi数据集算法演示', layout)
    # Event Loop to process "events" and get the "values" of the inputs

    # Try to load model
    memn2n = MemN2N(data_dir, model_file)
    memn2n.load_model()

    # Read test data
    print("Reading test data from %s ..." % memn2n.data_dir)
    test_data_path = glob.glob('%s/qa*_*_test.txt' % memn2n.data_dir)
    test_story, test_questions, test_qstory = \
        parse_babi_task(test_data_path, memn2n.general_config.dictionary, False)


    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break

        if event == 'getstory':
            question_idx = np.random.randint(test_questions.shape[1])
            story_idx = test_questions[0, question_idx]
            last_sentence_idx = test_questions[1, question_idx]

            story_txt, question_txt, correct_answer = memn2n.get_story_texts(test_story, test_questions, test_qstory,
                                                                             question_idx, story_idx, last_sentence_idx)
            # Format text
            story_txt = "\n".join(story_txt)
            question_txt += "?"

            # return flask.jsonify({
            #     "question_idx": question_idx,
            #     "story": story_txt,
            #     "question": question_txt,
            #     "correct_answer": correct_answer
            # })
            window['story'].update(value=story_txt)
            window['question'].update(value=question_txt)
        elif event == 'getanswer':
            question_idx = question_idx
            user_question = values['question']

            story_idx = test_questions[0, question_idx]
            last_sentence_idx = test_questions[1, question_idx]

            pred_answer_idx, pred_prob, memory_probs = memn2n.predict_answer(test_story, test_questions, test_qstory,
                                                                             question_idx, story_idx, last_sentence_idx,
                                                                             user_question)
            pred_answer = memn2n.reversed_dict[pred_answer_idx]

            window['answer'].update(value=pred_answer)

    window.close()