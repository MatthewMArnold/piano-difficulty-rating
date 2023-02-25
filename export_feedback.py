import csv
import os

import music21
import numpy as np
import xgboost

from loader_representations import get_path, velocity_piece, notes_piece, finger_piece, finger_nakamura_piece, prob_piece

from utils import strm2map, load_json, save_json


def salamizer(X, win_size=5):
    X = np.array(X)
    X_ans = []
    hop_size = 1

    for i in range(1, X.shape[0] - win_size + 1, hop_size):
        window = X[i:i + win_size, :].reshape((-1, 1))  # each individual window
        X_ans.append(window)
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
    with open(r_h_cost) as tsv_file:
        r_h = list(csv.reader(tsv_file, delimiter="\t"))
    with open(l_h_cost) as tsv_file:
        l_h = list(csv.reader(tsv_file, delimiter="\t"))

    PIG_content = []
    for idx, content in enumerate(sorted(r_h + l_h, key=lambda a: float(a[1]))):
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


def save_score_difficulty(alias, output, piece, onset_difficulty, rep, appr):

    header = ['']

    is_nakamura = rep in ['prob', 'finger_nakamura']

    path_to_save = os.path.join(output, os.path.basename(piece)[:-4] + '.pdf')
    if not is_nakamura:
        r_h_cost = '/'.join(["Fingers", "pianoplayer", os.path.basename(piece)[:-4] + '_rh.txt'])
        l_h_cost = '/'.join(["Fingers", "pianoplayer", os.path.basename(piece)[:-4] + '_lh.txt'])
        with open(r_h_cost) as tsv_file:
            r_h = list(csv.reader(tsv_file, delimiter="\t"))
        with open(l_h_cost) as tsv_file:
            l_h = list(csv.reader(tsv_file, delimiter="\t"))
        h = sorted(r_h + l_h, key=lambda a: float(a[3]), reverse=True)
    else:
        h_cost = '/'.join(["Fingers", "nakamura", os.path.basename(os.path.basename(piece)[:-4]) + '.txt'])
        with open(h_cost) as tsv_file:
            all_h = list(csv.reader(tsv_file, delimiter="\t"))[1:]
        for idx in range(len(all_h)):
            all_h[idx][3] = an2midi(all_h[idx][3])
        h = sorted(all_h, key=lambda a: float(a[3]), reverse=True)
    r_diff = []
    l_diff = []

    for idx, content in enumerate(sorted(h, key=lambda a: float(a[1]))):
        if content[6] == '0':
            if (is_nakamura or content[7] != -1) and round(float(content[1]), 2) in onset_difficulty:
                r_diff.append((content[7], (round(onset_difficulty[round(float(content[1]), 2)]))))
            else:
                r_diff.append((content[7], -1))
        else:
            if (is_nakamura or content[7] != -1) and round(float(content[1]), 2) in onset_difficulty:
                l_diff.append((content[7], round(onset_difficulty[round(float(content[1]), 2)])))
            else:
                l_diff.append((content[7], -1))
    if appr == 'deepgru':
        INTERP = [
            '#000061',
            '#0000cc',
            '#0000ff',
            '#3333ff',
            '#6666ff',
            '#9999ff',
            '#b3b3ff',
            '#ccccff',
            '#e6e6ff',
            'white'
        ]
    else:
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
            music21_structure.articulations = [f] + music21_structure.articulations
    # sf.write('mxl.pdf', fp=path_to_save)

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
        output = f'feedback/{split}/xgboost/{rep}/{subset}'

        os.makedirs(output, exist_ok=True)

        # Already computed results for this piece
        if os.path.exists(os.path.join(output, os.path.splitext(os.path.basename(piece))[0] + '.musicxml')):
            continue

        # Load appropriate model

        # Model trained with window size 9
        model = f"results/xgboost/rep_{rep}/w9/{split}.pkl"

        print(f'scoring difficulty using model: {model} rep: {rep} on piece: {piece}')

        # load piece with representation path, grade, path_alias, xml
        feature_representation = get_feature_representation(rep)
        matrix, onsets = feature_representation(piece, "nak" if rep in ["finger_nakamura", "prob"] else "mikro2")

        print(len(matrix))

        windows = salamizer(matrix, 9)
        clf = xgboost.XGBClassifier()
        clf.load_model(model)
        prediction = clf.predict(windows)

        # get the values per onset
        # onset_difficulty = get_onset_difficulty(prediction, onsets)

        # save the output
        # save_score_difficulty("mikro2", output, piece, onset_difficulty, rep, appr)

def export_feedback(split):
    pieces_subsets = load_split(split)
    subset = 'test'
    pieces = pieces_subsets[subset]
    pieces = [pieces[1]]

    for piece in pieces:
        generate_difficulty_xgboost(split, piece)

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
    export_feedback(10)
    update_json()
    # save_midis()












