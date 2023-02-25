import csv
import os

import music21
import numpy as np
import argparse
import xgboost
import pig_utils

from loader_representations import get_path, velocity_piece, notes_piece, finger_piece, finger_nakamura_piece, prob_piece

from utils import strm2map, load_json, save_json


def windowizer(X, win_size=5):
    '''
    Starts with a matrix "X" of dimension N x D. Window size is the number of
    rows viewed at one time by some algorithm. The goal of this function is to
    convert "X" into a matrix where each row is a vector containing the data
    that some window sees. For example, given the matrix X = [[1], [2], [3]] and
    the window size is 2, the returned matrix will be [[1, 2], [2, 3]].
    '''
    num_windows = X.shape[0] - win_size + 1
    X_ans = np.empty((num_windows, win_size * X.shape[1]))
    for win in range(0, num_windows):
        window = X[win:win + win_size, :].reshape(-1,)
        X_ans[win] = window
    X_ans = np.squeeze(np.array(X_ans))
    return X_ans


def get_onset_difficulty(prediction, onsets):
    ans = {}
    for idx, p in enumerate(prediction):
        ans[onsets[idx]] = p
    return ans


def save_PIG_difficulty(alias, model, piece, onset_difficulty, rep):
    path_alias = get_path(alias)
    path_to_save = os.path.join('visualization', model, piece + '.txt')
    r_h_cost = '/'.join(["Fingers", path_alias, piece + '_rh.txt'])
    l_h_cost = '/'.join(["Fingers", path_alias, piece + '_lh.txt'])
    with open(r_h_cost) as csv_file:
        rh = list(csv.reader(csv_file, delimiter="\t"))
    with open(l_h_cost) as csv_file:
        lh = list(csv.reader(csv_file, delimiter="\t"))

    PIG_content = []
    for idx, content in enumerate(sorted(rh + lh, key=lambda a: float(a[1]))):
        new_content = content
        if content[7] != -1 and round(float(content[1]), 2) in onset_difficulty:
            new_content[0] = idx
            new_content.append(round(onset_difficulty[round(float(content[1]), 2)]))
            PIG_content.append(new_content)
        else:
            new_content[0] = idx
            new_content.append(-1)
            PIG_content.append(new_content)

    with open(path_to_save, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        for record in PIG_content:
            writer.writerow(record)


KEY_TO_SEMITONE = {'c': 0, 'c#': 1, 'db': 1, 'd-': 1, 'c##': 2, 'd': 2, 'e--': 2, 'd#': 3, 'eb': 3, 'e-': 3, 'd##': 4, 'e': 4, 'f-': 4, 'e#': 5,
                   'f': 5, 'g--': 5, 'e##': 6, 'f#': 6, 'gb': 6, 'g-': 6, 'f##': 7, 'g': 7, 'a--': 7, 'g#': 8, 'ab': 8, 'a-': 8,
                   'g##': 9, 'a': 9, 'b--': 9,
                   'a#': 10, 'bb': 10, 'b-': 10, 'a##': 11, 'b': 11, 'b#': 12, 'c-': -1, 'x': None}


def an2midi(an):
    a = an[:-1].lower()  # alpha
    n = int(an[-1])  # numeric
    return n * 12 + KEY_TO_SEMITONE[a]

def get_fingering_data(piece):
    name = os.path.splitext(piece)[0]
    return name + '_rh.txt', name + '_lh.txt'

def save_score_difficulty(output, piece, onset_difficulty, rep, appr):
    path_to_save = os.path.join(output, os.path.basename(piece)[:-4] + '.pdf')

    rh_cost, lh_cost = get_fingering_data(piece)
    with open(rh_cost) as csv_file:
        rh = list(csv.reader(csv_file, delimiter="\t"))
    with open(lh_cost) as csv_file:
        lh = list(csv.reader(csv_file, delimiter="\t"))
    h = sorted(rh + lh, key=lambda a: float(a[pig_utils.SPELLED_PITCH_IDX]), reverse=True)

    r_diff = []
    l_diff = []

    print(onset_difficulty)

    for content in sorted(h, key=lambda a: float(a[pig_utils.ONSET_TIME_IDX])):
        onset_time = round(float(content[pig_utils.ONSET_TIME_IDX]), 2)
        cost = content[pig_utils.COST_IDX]

        difficulty = -1

        if content[pig_utils.FINGER_NUMBER_IDX] == '0':
            if cost != -1 and onset_time in onset_difficulty:
                difficulty = round(onset_difficulty[onset_time])
            r_diff.append((cost, difficulty))
        else:
            if cost != -1 and onset_time in onset_difficulty:
                difficulty = round(onset_difficulty[onset_time])
            l_diff.append((cost, difficulty))

    green = '#a1de00'
    yellow = '#f6b100'
    red = '#e30000'
    INTERP = [green, yellow, red, 'white']

    sf = music21.converter.parse(piece)
    rh_om = strm2map(sf.parts[0])
    lh_om = strm2map(sf.parts[1])
    for om, diff_list in zip([rh_om, lh_om], [r_diff, l_diff]):
        for o, (finger, diff) in zip(om, diff_list):
            if 'chord' in o:
                music21_structure = o['chord']
            else:
                music21_structure = o['element']
            o['element'].style.color = INTERP[diff]
            f = music21.articulations.Fingering(finger)
            music21_structure.articulations = [f] + music21_structure.articulation
    sf.write('mxl.pdf', fp=path_to_save)

def load_split(split):
    s = load_json("mikrokosmos/splits.json")[str(split)]
    return {"test": s['ids_test'], "train": s['ids_train']}


def load_split_basename(split):
    s = load_json("mikrokosmos/splits.json")[split]
    ans = {"test": [int(os.path.splitext(os.path.basename(x))[0]) for x in s['ids_test']],
            "train": [int(os.path.splitext(os.path.basename(x))[0]) for x in s['ids_train']]}
    ans['train'].sort()
    ans['test'].sort()
    return ans


def get_feature_representation(rep):
    if rep == "note":
        ans = notes_piece
    elif rep == "finger":
        ans = finger_piece
    elif rep == "finger_nakamura":
        ans = finger_nakamura_piece
    elif rep == "velocity":
        ans = velocity_piece
    elif rep == "prob":
        ans = prob_piece
    else:
        raise "bad representation"
    return ans


def generate_difficulty_xgboost(split, piece):
    appr = 'xgboost'
    subset = 'test'

    for rep in ["note"]:# , "finger", "finger_nakamura", "prob", "velocity"]:
        # variables
        output_dir = f'feedback/{split}/xgboost/{rep}/{subset}'

        os.makedirs(output_dir, exist_ok=True)

        # Already computed results for this piece
        if os.path.exists(os.path.join(output_dir, os.path.splitext(os.path.basename(piece))[0] + '.musicxml')):
            continue

        # Load appropriate model

        # Model trained with window size 9
        model = f"results/xgboost/rep_{rep}/w9/{split}.pkl"

        print(f'scoring difficulty using model: {model} rep: {rep} on piece: {piece}')

        # load piece with representation path, grade, path_alias, xml
        feature_representation = get_feature_representation(rep)
        matrix, onsets = feature_representation(piece, "nak" if rep in ["finger_nakamura", "prob"] else "mikro2")

        matrix = np.array(matrix)
        print(matrix.shape)

        # Window size of trained model is 9
        windows = windowizer(matrix, 9)
        clf = xgboost.XGBClassifier()
        clf.load_model(model)
        prediction = clf.predict(windows)

        # get the values per onset (associate onset time with the prediction)
        onset_difficulty = get_onset_difficulty(prediction, onsets)

        # save the output
        save_score_difficulty(output_dir, piece, onset_difficulty, rep, appr)

def update_json():
    structure = {}
    for d in os.listdir('./feedback'):
        if os.path.isdir('./feedback/' + d):
            structure[d] = load_split_basename(d)

    save_json(structure, "feedback_structure.json")


def save_midis():
    if not os.path.exists('mikrokosmos_midis'):
        os.mkdir('mikrokosmos_midis')

    for path, _ in load_json("mikrokosmos/henle_mikrokosmos.json").items():
            path_xml = f"mikrokosmos/musicxml/{path}.xml"
            path_midi = f"mikrokosmos_midis/{path}.mid"

            sc = music21.converter.parse(path_xml)
            sc.write('midi', fp=path_midi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, required=True)
    args = parser.parse_args()
    
    generate_difficulty_xgboost(10, args.file)
    update_json()
    # save_midis()











