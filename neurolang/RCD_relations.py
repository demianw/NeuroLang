from .interval_algebra import *
from .regions import *
import numpy as np


def intervals_relations_from_boxes(bounding_box, another_bounding_box):
    ''' retrieve interval relations of the axes between two objects '''
    intervals = bounding_box.axis_intervals()
    other_box_intervals = another_bounding_box.axis_intervals()
    return get_intervals_relations(intervals, other_box_intervals)

def get_intervals_relations(intervals, other_box_intervals):
    obtained_relation_per_axis = ['' for _ in range(len(intervals))]
    relations = [before, overlaps, during, meets, starts, finishes, equals]
    for f in relations:
        for i in range(len(obtained_relation_per_axis)):
            if f(intervals[i], other_box_intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0])
            elif f(other_box_intervals[i], intervals[i]):
                obtained_relation_per_axis[i] = str(f.__name__[0] + 'i')
        if np.all(np.array(obtained_relation_per_axis) != ''):
            break
    return tuple(obtained_relation_per_axis)


def is_in_direction(matrix, direction):
    if direction in ['L', 'C', 'R']:
        index = int(np.where(np.array(['L', 'C', 'R']) == direction)[0])
        return np.any(matrix[index] == 1)

    (a, s) = directions_map(direction)
    return matrix[1][a, s] == 1

def directions_map(d):
    return {'SP': (0, 0), 'S': (0, 1), 'SA': (0, 2),
            'P': (1, 0), 'O': (1, 1), 'A': (1, 2),
            'IP': (2, 0), 'I': (2, 1), 'IA': (2, 2)}[d]


def inverse_direction(d):
    return {'SP': 'IA', 'S': 'I', 'SA': 'IP',
            'P': 'A', 'O': 'O', 'A': 'P',
            'IP': 'SA', 'I': 'S', 'IA': 'SP'}[d]


def direction_matrix(bounding_box, another_bounding_box):
    ''' direction matrix of two bounding boxes '''

    res = np.zeros(shape=(3, 3, 3))
    bb_lb, bb_ub = tuple(bounding_box._lb), tuple(bounding_box._ub)
    if len(bb_lb) < 3:
        res[1] = translate_ia_relation(*intervals_relations_from_boxes(bounding_box, another_bounding_box))
        return res

    another_bb_lb, another_bb_ub = tuple(another_bounding_box._lb), tuple(another_bounding_box._ub)
    projected_bb_lb, projected_bb_ub = (another_bb_lb[:1] + bb_lb[1:], another_bb_ub[:1] + bb_ub[1:])
    bb_lr_limits = bounding_box.axis_intervals()[:1]
    another_bb_lr_limits = another_bounding_box.axis_intervals()[:1]
    lr_relation = get_intervals_relations(bb_lr_limits, another_bb_lr_limits)[0]

    projected_intervals = np.array([tuple([projected_bb_lb[i], projected_bb_ub[i]]) for i in range(len(projected_bb_lb))])# get intervals
    another_bounding_box.axis_intervals()
    postant_supinf_relations = get_intervals_relations(projected_intervals, another_bounding_box.axis_intervals())[1:]

    as_matrix = translate_ia_relation(*postant_supinf_relations)

    if lr_relation in ['d', 's', 'f', 'e']:
        res[1] = as_matrix
    elif lr_relation in ['m', 'b']:
        res[0] = as_matrix
    elif lr_relation in ['mi', 'bi']:
        res[2] = as_matrix
    elif lr_relation in ['o', 'fi']:
        res[0] = res[1] = as_matrix
    elif lr_relation in ['oi', 'si']:
        res[1] = res[2] = as_matrix
    elif lr_relation in ['di']:
        res[0] = res[1] = res[2] = as_matrix

    return res


def translate_ia_relation(x, y):
    '''' IA to RCD mapping '''
    if x in ['d', 's', 'f', 'e'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['mi', 'bi']:
        return np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['mi', 'bi']:
        return np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['mi', 'bi']:
        return np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    elif x in ['m', 'b'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])

    elif x in ['fi', 'o'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 1, 0]])
    elif x in ['si', 'oi'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [0, 1, 1]])
    elif x in ['fi', 'o'] and y in ['mi', 'bi']:
        return np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['mi', 'bi']:
        return np.array([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['fi', 'o'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['si', 'oi']:
        return np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    elif x in ['m', 'b'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [1, 0, 0], [1, 0, 0]])

    elif x in ['m', 'b'] and y in ['si', 'oi']:
        return np.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 0, 1]])
    elif x in ['mi', 'bi'] and y in ['si', 'oi']:
        return np.array([[0, 0, 1], [0, 0, 1], [0, 0, 0]])
    elif x in ['di'] and y in ['m', 'b']:
        return np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]])
    elif x in ['di'] and y in ['mi', 'bi']:
        return np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    elif x in ['di'] and y in ['d', 's', 'f', 'e']:
        return np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]])
    elif x in ['d', 's', 'f', 'e'] and y in ['di']:
        return np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    elif x in ['m', 'b'] and y in ['di']:
        return np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    elif x in ['mi', 'bi'] and y in ['di']:
        return np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]])

    elif x in ['o', 'fi'] and y in ['o', 'fi']:
        return np.array([[0, 0, 0], [1, 1, 0], [1, 1, 0]])
    elif x in ['o', 'fi'] and y in ['si', 'oi']:
        return np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
    elif x in ['si', 'oi'] and y in ['o', 'fi']:
        return np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]])
    elif x in ['si', 'oi'] and y in ['si', 'oi']:
        return np.array([[0, 1, 1], [0, 1, 1], [0, 0, 0]])
    elif x in ['o', 'fi'] and y in ['di']:
        return np.array([[1, 1, 0], [1, 1, 0], [1, 1, 0]])
    elif x in ['si', 'oi'] and y in ['di']:
        return np.array([[0, 1, 1], [0, 1, 1], [0, 1, 1]])
    elif x in ['di'] and y in ['fi', 'o']:
        return np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1]])
    elif x in ['di'] and y in ['si', 'oi']:
        return np.array([[1, 1, 1], [1, 1, 1], [0, 0, 0]])
    elif x in ['di'] and y in ['di']:
        return np.array(np.ones(shape=(3, 3)))
