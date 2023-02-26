"""
    File name: approach_deepgru.py
    Author: Pedro Ramoneda
    Python Version: 3.7
"""


import csv
import os
import sys

import numpy as np

from utils import load_xmls, load_json, save_json

import pig_utils
# onset time, offset time, spelled pitch, onset velocity, offset velocity, finger number, cost, note ID

def get_fingering_data(piece):
    name = os.path.splitext(piece)[0]
    return name + '_rh.txt', name + '_lh.txt'

NUM_PITCHES = 88

ALIAS_TO_PATH = {
    'mikro1': 'mikrokosmos1',
    'mikro2': 'pianoplayer',
    'nak': 'nakamura',
}

def get_path(alias):
    return ALIAS_TO_PATH[alias]

def rep_raw(alias):
    path_alias = get_path(alias)
    rep = {}
    for grade, path, xml in load_xmls():
        rep[path] = {
            'grade': grade,
            'right_velocity': [],
            'left_velocity': [],
            'right_fingers': [],
            'left_fingers': []
        }
        rh_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_rh.txt'])
        lh_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '_lh.txt'])
        for path_txt, hand in zip([rh_cost, lh_cost], ["right_", "left_"]):
            with open(path_txt) as csv_file:
                read_csv = csv.reader(csv_file, delimiter="\t")
                for l in read_csv:
                    rep[path][hand + 'velocity'] = rep[path][hand + 'velocity'] + [float(l[8])]
                    rep[path][hand + 'fingers'] = rep[path][hand + 'fingers'] + [abs(int(l[7]))]
    save_json(rep, os.path.join('representations', path_alias, 'rep_raw.json'))


def merge_chord_onsets(time_series):
    new_time_series = [list(a) for a in time_series]
    for ii in range(len(time_series)):
        if ii + 1 < len(time_series) and time_series[ii][0] + 0.05 == time_series[ii + 1][0]:
            if ii + 2 < len(time_series) and time_series[ii][0] + 0.1 == time_series[ii + 2][0]:
                if ii + 3 < len(time_series) and time_series[ii][0] + 0.15 == time_series[ii + 3][0]:
                    if ii + 4 < len(time_series) and time_series[ii][0] + 0.2 == time_series[ii + 4][0]:
                        new_time_series[ii][0] = time_series[ii + 4][0]
                    else:
                        new_time_series[ii][0] = time_series[ii + 3][0]
                else:
                    new_time_series[ii][0] = time_series[ii + 2][0]
            else:
                new_time_series[ii][0] = time_series[ii + 1][0]
        else:
            new_time_series[ii][0] = time_series[ii][0]
    return [tuple(a) for a in new_time_series]


def finger2index(f):
    if f > 0:
        index = int(f) + 4
    elif f < 0:
        index = int(f) - 5
    else:  # == 0
        index = -1000
    return index


def velocity_piece(path, alias):
    rh_cost, lh_cost = get_fingering_data(path)

    intermediate_rep = []
    for path_txt in [rh_cost, lh_cost]:
        time_series = []
        with open(path_txt) as csv_file:
            read_csv = csv.reader(csv_file, delimiter="\t")
            for l in read_csv:
                cost = int(l[pig_utils.COST_IDX])
                
                if cost != 0:
                    onset_time = round(float(l[pig_utils.ONSET_TIME_IDX]), 2)
                    note_idx = abs(float(l[pig_utils.NOTE_IDX]))
                    time_series.append((onset_time, cost, note_idx))
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        matrix.append(t)

    return matrix, onsets


def rep_velocity(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = velocity_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_velocity.json'))


def prob_piece(path, alias, xml):
    path_alias = get_path(alias)

    PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

    time_series = []
    with open(PIG_cost) as csv_file:
        read_csv = csv.reader(csv_file, delimiter="\t")
        for l in list(read_csv)[1:]:
            if int(l[7]) != 0:
                time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8])))))
    time_series = time_series[:-3]

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

    onsets = []
    idx = 0
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = intermediate_rep[idx][2]
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = intermediate_rep[j][2]
            j += 1
        idx = j
        matrix.append(t)

    return matrix, onsets


def rep_prob(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = prob_piece(path, alias, xml)

        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }
    save_json(rep, os.path.join('representations', get_path(alias), 'rep_nakamura.json'))


def rep_d_nakamura(alias):
    path_alias = get_path(alias)
    rep = {}
    for grade, path, xml in load_xmls():
        PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

        time_series = []
        with open(PIG_cost) as csv_file:
            read_csv = csv.reader(csv_file, delimiter="\t")
            for l in list(read_csv)[1:]:
                if int(l[7]) != 0:
                    time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8]))), round(float(l[2]), 2)))
        time_series = time_series[:-3]


        # order by onset and create matrix
        matrix = []
        intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

        idx = 0
        while idx < len(intermediate_rep):
            t = [0] * 10
            index = finger2index(intermediate_rep[idx][1])

            t[index] = intermediate_rep[idx][2] / (intermediate_rep[idx][3] - intermediate_rep[idx][0])
            j = idx + 1
            while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
                index = finger2index(intermediate_rep[j][1])
                t[index] = intermediate_rep[j][2] / (intermediate_rep[j][3] - (intermediate_rep[j][0]))
                j += 1
            idx = j
            matrix.append(t)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', path_alias, 'rep_d_nakamura.json'))


def finger_piece(path, _):
    '''
    @return a fingering matrix where each row in the matrix is a 1 hot vector of
        length 10. A 1 in some index indicates the finger is used to play the note.
    '''
    rh_cost, lh_cost = get_fingering_data(path)

    intermediate_rep = []
    for path_txt in [rh_cost, lh_cost]:
        time_series = []
        with open(path_txt) as csv_file:
            read_csv = csv.reader(csv_file, delimiter="\t")
            for l in read_csv:
                cost = int(l[pig_utils.COST_IDX])
                if cost != 0:
                    onset_time = round(float(l[pig_utils.ONSET_TIME_IDX]), 2)
                    note_idx = abs(float(l[pig_utils.NOTE_IDX]))
                    time_series.append((onset_time, cost, note_idx))
        time_series = time_series[:-1]
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = 1.0
            j += 1
        idx = j
        matrix.append(t)

    return matrix, onsets

def rep_finger(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = finger_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_finger.json'))


def finger_nakamura_piece(path, alias, xml):
    path_alias = get_path(alias)
    PIG_cost = '/'.join(["Fingers", path_alias, os.path.basename(xml[:-4]) + '.txt'])

    time_series = []
    with open(PIG_cost) as csv_file:
        read_csv = csv.reader(csv_file, delimiter="\t")
        for l in list(read_csv)[1:]:
            if int(l[7]) != 0:
                time_series.append((round(float(l[1]), 2), int(l[7]), abs(abs(float(l[8])))))
    time_series = time_series[:-3]

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(time_series, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0] * 10
        index = finger2index(intermediate_rep[idx][1])

        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = finger2index(intermediate_rep[j][1])
            t[index] = 1.0
            j += 1
        idx = j
        matrix.append(t)
    return matrix, onsets


def rep_finger_nakamura(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = finger_nakamura_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_finger_nakamura.json'))


def notes_piece(path, _):
    '''
    Extracts notes from the generated PianoPlayer fingering file.

    There must exist PianoPlayer-generated fingering files. If the path to the
    musicXML file is foobar.xml, fingering files must be named foobar_rh.txt
    and foobar_lh.txt and must be located in the same directory as the given
    musicXML file.

    @param path A path to a musicXML file whose notes should be extracted from.
    @param _ Unused. The user may pass in an alias but pianoplayer is the only
        acceptable alias.
    @return a one hot matrix in the form of a 2D list, where each nested list
        in the main list contains a vector of possible pitches, and one index
        in the vector will be set to 1 indicating the pitch played at that time.
        Left and right hand fingerings will be combined in the matrix and ordered
        by note onset time.
    '''
    piece_name = os.path.splitext(path)[0]
    rh_cost = piece_name + '_rh.txt'
    lh_cost = piece_name + '_lh.txt'

    intermediate_rep = []
    for path_txt in [rh_cost, lh_cost]:
        time_series = []
        with open(path_txt) as csv_file:
            read_csv = csv.reader(csv_file, delimiter="\t")
            for l in read_csv:
                # If no cost to play the note, don't include in the matrix
                if int(l[pig_utils.COST_IDX]) != 0:
                    onset_time = round(float(l[pig_utils.ONSET_TIME_IDX]), 2)
                    offset_spelled_pitch = int(l[pig_utils.SPELLED_PITCH_IDX]) - 21
                    time_series.append((onset_time, offset_spelled_pitch))
        intermediate_rep.extend(time_series)

    # order by onset and create matrix
    matrix = []
    intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]

    idx = 0
    onsets = []
    while idx < len(intermediate_rep):
        onsets.append(intermediate_rep[idx][0])
        t = [0.0] * NUM_PITCHES
        index = intermediate_rep[idx][1]
        t[index] = 1.0
        j = idx + 1
        while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
            index = intermediate_rep[j][1]
            t[index] = 1.0
            j += 1
        idx = j
        matrix.append(t)
    return matrix, onsets



if __name__ == '__main__':
    notes_piece('test.midi', 1)

def rep_notes(alias):
    rep = {}
    for grade, path, xml in load_xmls():
        matrix, _ = notes_piece(path, alias, xml)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }

    save_json(rep, os.path.join('representations', get_path(alias), 'rep_note.json'))


def visualize_note_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_note.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_finger_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_finger.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_finger_representation_nakamura(alias="nak", score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_finger_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_velocity_representation(alias, score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_velocity.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def visualize_prob_representation(alias="nak", score='mikrokosmos/musicxml/5.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print('%02.1f' % c, end="|")
        print()


def visualize_d_nakamura(alias="nak", score='mikrokosmos/musicxml/69.xml'):
    data = load_json(os.path.join('representations', get_path(alias), 'rep_d_nakamura.json'))
    matrix = data[score]['matrix']
    for row in np.array(matrix).transpose():
        for c in row:
            print(c, end="|")
        print()


def get_distance_type(last_semitone, current_semitone):
    last_black = (last_semitone % 12) in [1, 3, 6, 8, 10]
    current_black = (current_semitone % 12) in [1, 3, 6, 8, 10]

    if not last_black and not current_black:
        distance_type = 1
    elif last_black and not current_black:
        distance_type = 2
    elif not last_black and current_black:
        distance_type = 3
    else:  # bb
        distance_type = 4

    return distance_type


def rep_distances(alias):
    path_alias = get_path(alias)

    rep = {}
    for grade, path, r_h, l_h in load_xmls():
        rh_cost = '/'.join(["Fingers", path_alias, r_h[:-11] + '_rh.txt'])
        lh_cost = '/'.join(["Fingers", path_alias, l_h[:-11] + '_lh.txt'])

        intermediate_rep = []
        for path_txt, hand in zip([rh_cost, lh_cost], ["right_", "left_"]):
            time_series = []
            with open(path_txt) as csv_file:
                read_csv = csv.reader(csv_file, delimiter="\t")
                for l in read_csv:
                    if int(l[7]) != 0:
                        time_series.append((round(float(l[1]), 2), int(l[7]), abs(float(l[8])), abs(float(l[3]))))
            if alias == 'version_1.0':
                time_series = merge_chord_onsets(time_series[:-10])
            else:
                time_series = time_series[:-10]
            intermediate_rep.extend(time_series)

        # order by onset and create matrix
        matrix = []
        intermediate_rep = [on for on in sorted(intermediate_rep, key=(lambda a: a[0]))]
        # initial semitone: at the beginning the distance is 0
        last_semitone_rh = next(x[3] for x in intermediate_rep if x[1] > 0)
        last_semitone_lh = next(x[3] for x in intermediate_rep if x[1] < 0)
        idx = 0
        while idx < len(intermediate_rep):
            d, dt, t = [0] * 10, [0] * 10, [0] * 10
            index = finger2index(intermediate_rep[idx][1])
            is_r_h = index >= 5
            last_semitone = last_semitone_rh if is_r_h else last_semitone_lh
            t[index] = intermediate_rep[idx][2]
            d[index] = last_semitone - intermediate_rep[idx][3]
            dt[index] = get_distance_type(last_semitone, intermediate_rep[idx][3])
            if is_r_h:
                last_semitone_rh = last_semitone
            else:
                last_semitone_lh = last_semitone

            j = idx + 1
            while j < len(intermediate_rep) and intermediate_rep[idx][0] == intermediate_rep[j][0]:
                index = finger2index(intermediate_rep[j][1])
                is_r_h = index >= 5
                last_semitone = last_semitone_rh if is_r_h else last_semitone_lh
                t[index] = intermediate_rep[j][2]
                d[index] = last_semitone - intermediate_rep[idx][3]
                dt[index] = get_distance_type(last_semitone, intermediate_rep[idx][3])
                if is_r_h:
                    last_semitone_rh = last_semitone
                else:
                    last_semitone_lh = last_semitone
                j += 1
            idx = j
            matrix.append(t + d + dt)
        rep[path] = {
            'grade': grade,
            'matrix': matrix
        }
    save_json(rep, os.path.join('representations', path_alias, 'rep_distance.json'))


def rep_fing_vel_time(alias):
    get_path(alias)


def rep_distances_time(alias):
    get_path(alias)


def rep_merged_time(alias):
    get_path(alias)


def load_rep(klass):
    if klass == "rep_velocity":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_velocity.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_finger":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_finger.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_finger_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_finger_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_prob":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_d_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_d_nakamura.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_note":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_note.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    elif klass == "rep_distance":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_distance.json')
        data = load_json(path)
        ans = ([np.array(x['matrix']) for k, x in data.items()], np.array([x['grade'] for k, x in data.items()]))
    return ans


def load_rep_info(klass):
    if klass == "rep_velocity":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_velocity.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_finger":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_finger.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_finger_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_finger_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_prob":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_d_nakamura":
        path_alias = get_path("nak")
        path = os.path.join('representations', path_alias, 'rep_d_nakamura.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_note":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_note.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    elif klass == "rep_distance":
        path_alias = get_path("mikro2")
        path = os.path.join('representations', path_alias, 'rep_distance.json')
        data = load_json(path)
        ans = np.array([k for k, x in data.items()])
    return ans
