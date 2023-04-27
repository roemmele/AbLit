import argparse
import json
import os
import pandas
import numpy
from sklearn.metrics import precision_recall_fscore_support


def binarize_alignment_labels(alignment):

    paired_nums = set([(orig_num, abridged_num)
                       for orig_nums, abridged_nums in zip(alignment['original_segment_nums'],
                                                           alignment['abridged_segment_nums'])
                       for orig_num in orig_nums
                       for abridged_num in abridged_nums]
                      )

    max_orig_seg_num = max([num for nums in alignment['original_segment_nums']
                            for num in nums])
    max_abridged_seg_num = max([num for nums in alignment['abridged_segment_nums']
                                for num in nums])

    binary_labels = {}
    for orig_num in range(max_orig_seg_num + 1):
        for abridged_num in range(max_abridged_seg_num + 1):
            binary_labels[(orig_num,
                           abridged_num)] = (True if (orig_num, abridged_num,) in paired_nums
                                             else False)

    return binary_labels


def get_alignment_labels(alignment):

    paired_nums = set()

    for orig_nums, abridged_nums in zip(alignment['original_segment_nums'],
                                        alignment['abridged_segment_nums']):
        for orig_num in orig_nums:
            if abridged_nums:
                for abridged_num in abridged_nums:
                    paired_nums.add((orig_num, abridged_num,))
            else:
                paired_nums.add((orig_num, None,))

    return paired_nums


def score_chapter_alignment(pred_alignment, gold_alignment):

    pred_labels = get_alignment_labels(pred_alignment)
    gold_labels = get_alignment_labels(gold_alignment)

    n_pred_labels = len(pred_labels)
    n_gold_labels = len(gold_labels)

    n_overlapped_labels = len(pred_labels.intersection(gold_labels))
    precision = n_overlapped_labels / len(pred_labels)
    recall = n_overlapped_labels / len(gold_labels)
    fscore = 2 * ((precision * recall) / (precision + recall))

    return (n_pred_labels, n_gold_labels, precision, recall, fscore)


def get_alignment_from_json(json_obj):
    '''Detect and handle both legacy and new format of json data'''

    alignment = {'original_segment_nums': [],
                 'abridged_segment_nums': []}

    if type(json_obj) == list:  # New format
        prev_orig_seg_num = 0
        prev_abrg_seg_num = 0
        for item in json_obj:

            orig_seg_nums = list(range(prev_orig_seg_num + 1,
                                       prev_orig_seg_num + 1 + len(item['original_segments'])))
            alignment['original_segment_nums'].append(orig_seg_nums)
            prev_orig_seg_num = (prev_orig_seg_num +
                                 len(item['original_segments']))

            abrg_seg_nums = list(range(prev_abrg_seg_num + 1,
                                       prev_abrg_seg_num + 1 + len(item['abridged_segments'])))
            alignment['abridged_segment_nums'].append(abrg_seg_nums)
            prev_abrg_seg_num = (prev_abrg_seg_num +
                                 len(item['abridged_segments']))

    else:  # Legacy format
        if 'original_sentence_idxs' in json_obj:
            alignment['original_segment_nums'] = [[idx + 1 for idx in idxs]
                                                  for idxs in json_obj['original_sentence_idxs']]
        else:
            alignment['original_segment_nums'] = json_obj['original_segment_nums']

        if 'abridged_sentence_idxs' in json_obj:
            alignment['abridged_segment_nums'] = [[idx + 1 for idx in idxs]
                                                  for idxs in json_obj['abridged_sentence_idxs']]
        else:
            alignment['abridged_segment_nums'] = json_obj['abridged_segment_nums']

    return alignment


def evaluate(pred_alignment_dir,
             gold_alignment_dir,
             meta_data_file):

    chapter_scores = {'Book_Id': [],
                      'Chapter_Idx': [],
                      'N_Pred_Labels': [],
                      'N_Gold_Labels': [],
                      'Precision': [],
                      'Recall': [],
                      'F1': []}

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    for book_id, book_info in meta_data.items():

        for chapter_idx in book_info['chapter_idxs']:

            with open(os.path.join(pred_alignment_dir,
                                   book_id,
                                   "{}.json".format(chapter_idx))) as f:
                pred_alignment = get_alignment_from_json(json.load(f))

            with open(os.path.join(gold_alignment_dir,
                                   book_id,
                                   "{}.json".format(chapter_idx))) as f:
                gold_alignment = get_alignment_from_json(json.load(f))

            # import pdb
            # pdb.set_trace()

            (n_pred_labels,
             n_gold_labels,
             precision,
             recall,
             f1) = score_chapter_alignment(pred_alignment,
                                           gold_alignment)

            chapter_scores['Book_Id'].append(book_id)
            chapter_scores['Chapter_Idx'].append(chapter_idx)
            chapter_scores['N_Pred_Labels'].append(n_pred_labels)
            chapter_scores['N_Gold_Labels'].append(n_gold_labels)
            chapter_scores['Precision'].append(precision)
            chapter_scores['Recall'].append(recall)
            chapter_scores['F1'].append(f1)

    # import pdb;pdb.set_trace()
    scores = pandas.DataFrame(chapter_scores)
    output_file = os.path.join(
        os.path.split(os.path.dirname(pred_alignment_dir))[0],
        'scores.csv')

    total_n_pred_labels = scores['N_Pred_Labels'].sum()
    total_n_gold_labels = scores['N_Gold_Labels'].sum()

    score_weights = scores['N_Gold_Labels'] / total_n_gold_labels
    total_precision = numpy.average(scores['Precision'],
                                    weights=score_weights)
    total_recall = numpy.average(scores['Recall'],
                                 weights=score_weights)
    total_f1 = numpy.average(scores['F1'],
                             weights=score_weights)

    scores = pandas.concat([scores,
                            pandas.DataFrame({'Book_Id': ['All'],
                                              'Chapter_Idx': [None],
                                              'N_Pred_Labels': [total_n_pred_labels],
                                              'N_Gold_Labels': [total_n_gold_labels],
                                              'Precision': [total_precision],
                                              'Recall': [total_recall],
                                              'F1': [total_f1]})])
    scores.round(3).to_csv(output_file, index=False)
    print("Saved scores to", output_file)

    print("{:<20}{:>10.3f}".format('Total Weighted Precision', total_precision))
    print("{:<20}{:>10.3f}".format('Total Weighted Recall', total_recall))
    print("{:<20}{:>10.3f}".format('Total Weighted F1', total_f1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Score predicted alignment according to validated gold-standard alignment.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--pred_alignment_dir", "-pred_alignment_dir",
                        help="Directory containing predicted alignment data.\
                        Score output will be saved to this directory.",
                        type=str, required=True)
    parser.add_argument("--gold_alignment_dir", "-gold_alignment_dir",
                        help="Directory containing gold-standard alignment data.",
                        type=str, required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file", help="Path to .json file with book/chapter index.",
                        type=str, required=True)

    args = parser.parse_args()

    evaluate(args.pred_alignment_dir,
             args.gold_alignment_dir,
             args.meta_data_file)
