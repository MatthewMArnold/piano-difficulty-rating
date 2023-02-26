import csv
import os

import music21
import numpy as np
import argparse
import xgboost
import pig_utils

from loader_representations import get_path, velocity_piece, notes_piece, finger_piece

from utils import stream2map

def windowize(X, win_size=5):
    '''
    Starts with a matrix 'X' of dimension N x D. Window size is the number of
    rows viewed at one time by some algorithm. The goal of this function is to
    convert 'X' into a matrix where each row is a vector containing the data
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
    '''
    Creates a dict mapping onset time to difficulty prediction. This is
    returned. The values in the map are difficulty ratings ranging from [1, 3].

    @param prediction List of predicted window difficulty. 
    @param onsets A list of (rounded) onset times.
    '''
    # We can derive window size from the difference in sizes of the onset list
    # and prediction.
    windowsz = len(onsets) - len(prediction) + 1
    ans = {}
    for idx, onset in enumerate(onsets):
        # Compute the difficulty prediction for the given onset as the average
        # of all prediction windows that the onset is within.
        window_start = max(0, idx - windowsz + 1)
        window_end = min(idx + 1, len(prediction))

        prediction_avg = np.mean(prediction[window_start:window_end])
        ans[onset] = prediction_avg
    return ans

def save_PIG_difficulty(alias, model, piece, onset_difficulty, rep):
    path_alias = get_path(alias)
    path_to_save = os.path.join('visualization', model, piece + '.txt')
    r_h_cost = '/'.join(['Fingers', path_alias, piece + '_rh.txt'])
    l_h_cost = '/'.join(['Fingers', path_alias, piece + '_lh.txt'])
    with open(r_h_cost) as csv_file:
        rh = list(csv.reader(csv_file, delimiter='\t'))
    with open(l_h_cost) as csv_file:
        lh = list(csv.reader(csv_file, delimiter='\t'))

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

def save_score_difficulty(output_dir, piece, onset_difficulty):
    '''
    Saves score difficulty in mxl.pdf format.

    @param output_dir Directory where score will be outputted to.
    @param piece The musicXML file being scored.
    @param onset_difficulty dict mapping onset start to difficulty rating.
    '''
    rh_cost, lh_cost = get_fingering_data(piece)
    try:
        with open(rh_cost) as csv_file:
            rh = list(csv.reader(csv_file, delimiter='\t'))
    except FileNotFoundError as e:
        print(f'could not find file {rh_cost}')
        rh = []

    try:
        with open(lh_cost) as csv_file:
            lh = list(csv.reader(csv_file, delimiter='\t'))
    except FileNotFoundError as e:
        print(f'could not find file {lh_cost}')
        lh = []

    h = sorted(rh + lh, key=lambda a: float(a[pig_utils.SPELLED_PITCH_IDX]), reverse=True)

    rh_difficulty = []
    lh_difficulty = []

    for content in sorted(h, key=lambda a: float(a[pig_utils.ONSET_TIME_IDX])):
        onset_time = round(float(content[pig_utils.ONSET_TIME_IDX]), 2)
        cost = content[pig_utils.COST_IDX]

        difficulty = -1

        if content[pig_utils.FINGER_NUMBER_IDX] == '0':
            if cost != -1 and onset_time in onset_difficulty:
                difficulty = onset_difficulty[onset_time]
            rh_difficulty.append((cost, difficulty))
        else:
            if cost != -1 and onset_time in onset_difficulty:
                difficulty = onset_difficulty[onset_time]
            lh_difficulty.append((cost, difficulty))

    def linear_interpolate_color(c1, c2, t):
        return [int(x1 + (x2 - x1) * t) for x1, x2 in zip(c1, c2)]
        
    def get_color(diff):
        if diff == -1:
            return 'white'

        green   = [0xa1, 0xde, 0x00]
        yellow  = [0xf6, 0xb1, 0x00]
        red     = [0xe3, 0x00, 0x00]

        if diff < 1:
            c = linear_interpolate_color(green, yellow, diff)
        else:
            c = linear_interpolate_color(yellow, red, diff - 1)

        return '#' + ''.join([hex(x)[2:].zfill(2) for x in c])

    score = music21.converter.parse(piece)
    try:
        rh_om = stream2map(score.parts[0])
    except IndexError as e:
        rh_om = {}
        print('no right hand score')

    try:
        lh_om = stream2map(score.parts[1])
    except IndexError as e:
        lh_om = {}
        print('no left hand score')

    def label_notes(om, diff_list):
        for o, (finger, diff) in zip(om, diff_list):
            if 'chord' in o:
                music21_structure = o['chord']
            else:
                music21_structure = o['element']

            o['element'].style.color = get_color(diff)
            f = music21.articulations.Fingering(finger)
            music21_structure.articulations = [f] + music21_structure.articulations

    label_notes(rh_om, rh_difficulty)
    label_notes(lh_om, lh_difficulty)

    os.makedirs(output_dir, exist_ok=True)
    mxml_path = os.path.join(output_dir, os.path.splitext(os.path.split(piece)[1])[0] + '.pdf')
    score.write('mxl.pdf', fp=mxml_path)
    print(f'written results to {mxml_path}')

def get_feature_representation(rep):
    if rep == 'note':
        ans = notes_piece
    elif rep == 'finger':
        ans = finger_piece
    elif rep == 'velocity':
        ans = velocity_piece
    else:
        raise 'bad representation'
    return ans

def generate_difficulty_xgboost(rep, split, piece, output_dir):
    # Model trained with window size 9
    model = f'results/xgboost/rep_{rep}/w9/{split}.pkl'

    print(f'scoring difficulty using model: {model} rep: {rep} on piece: {piece}')

    # load piece with representation path, grade, path_alias, xml
    feature_representation = get_feature_representation(rep)
    matrix, onsets = feature_representation(piece, 'mikro2')

    matrix = np.array(matrix)

    # Window size of trained model is 9
    windows = windowize(matrix, 9)
    clf = xgboost.XGBClassifier()
    clf.load_model(model)
    prediction = clf.predict(windows)

    # get the values per onset (associate onset time with the prediction)
    onset_difficulty = get_onset_difficulty(prediction, onsets)

    # save the output
    save_score_difficulty(output_dir, piece, onset_difficulty)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type=str, required=True)
    parser.add_argument('-out_dir', type=str, required=True)
    parser.add_argument('-split', type=str, default=10, help='can be note, finger, or velocity')
    parser.add_argument('-rep', type=str, default='note')
    args = parser.parse_args()

    generate_difficulty_xgboost(args.rep, args.split, args.file, args.out_dir)
