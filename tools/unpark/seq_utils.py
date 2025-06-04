import numpy as np

def generate_sub_seq_info(seq):
    seq = np.asarray(seq)
    change_points = np.where(np.diff(seq) != 0)[0] + 1
    change_points = np.concatenate(([0], change_points, [len(seq)]))
    sequences = [(seq[start], start, end - 1, end - start)
                 for start, end in zip(change_points[:-1], change_points[1:])]
    # print(sequences)
    return sequences

def replace_LL_RR_O_label(seq):
    mapping = {5: 0, 4: 0, 6: 2, 7: 1}
    vectorized_replace = np.vectorize(lambda x: mapping.get(x, x))
    return vectorized_replace(seq)

def replace_short_seq_with_adjacent(seq):
    while True:
        change = False
        sequences = generate_sub_seq_info(seq)
        i = 0
        if len(sequences) > 2:
            while i + 2 < len(sequences):
                if sequences[i][0] == sequences[i+2][0] and sequences[i][0] != sequences[i+1][0] and sequences[i+1][3] < 5:
                    change = True
                    if sequences[i][2] + sequences[i+2][2] > sequences[i][1]:
                        start = sequences[i+1][1]
                        end = sequences[i+1][2]
                        seq[start: end+1] = np.array([sequences[i][0]] * (end - start + 1))
                    else:
                        start1 = sequences[i][1]
                        end1 = sequences[i][2]
                        seq[start1: end1+1] = np.array([sequences[i+1][0]] * (end1 - start1 + 1))
                        start2 = sequences[i+2][1]
                        end2 = sequences[i+2][2]
                        seq[start2: end2+1] = np.array([sequences[i+1][0]] * (end2 - start2 + 1))
                i += 1
        if not change:
            break
    return seq
    
def filter_abnormal_lc_seq(seq):
    while True:
        change = False
        sequences = generate_sub_seq_info(seq)
        i = 0
        if len(sequences) > 1:
            while i + 1 < len(sequences):
                if sequences[i][0] != sequences[i+1][0] and \
                sequences[i][3] < 20 and \
                sequences[i+1][0] == 0 and \
                i >= 1:
                    change = True
                    start = sequences[i][1]
                    end = sequences[i][2]
                    seq[start: end+1] = sequences[i+1][0]
                i += 1
        if not change:
            break
    return seq
    
def replace_discontinuous_subseq(seq):
    sequences = generate_sub_seq_info(seq)
    current_seq = []
    for i in range(len(sequences)):
        if sequences[i][3] < 10:
            current_seq.append(sequences[i])
        else:
            if current_seq:
                seq = replace_discontinuous_subseq_with_majority(seq, np.array(current_seq))
                current_seq = []
    if current_seq:
        seq = replace_discontinuous_subseq_with_majority(seq, np.array(current_seq))
    # print(generate_sub_seq_info(seq))
    return seq

def replace_discontinuous_subseq_with_majority(seq, current_seq):
    labels, counts = np.unique(current_seq[:, 0], return_counts=True)
    majority_num = labels[np.argmax(counts)]
    start = int(current_seq[0, 1])
    end = int(current_seq[-1, 2])
    seq[start:end+1] = majority_num
    return seq

# def replace_discontinuous_subseq_with_majority(seq, current_seq):
#     if len(current_seq) == 2:
#         majority_num = current_seq[1, 0] if current_seq[0, 3] < current_seq[1, 3] else current_seq[0, 0]
#         start, end = current_seq[0, 1], current_seq[1, 2]
#         seq[start:end+1] = majority_num
#     if len(current_seq) == 3:
#         majority_idx = np.argmax(current_seq[:, 3])
#         majority_num = current_seq[majority_idx, 0]
#         start, end = current_seq[0, 1], current_seq[2, 2]
#         seq[start:end+1] = majority_num 
#     return seq

def convert_seq(seq):
    seq = replace_LL_RR_O_label(seq)
    seq = replace_short_seq_with_adjacent(seq)
    seq = filter_abnormal_lc_seq(seq)
    seq = replace_discontinuous_subseq(seq)
    return seq
