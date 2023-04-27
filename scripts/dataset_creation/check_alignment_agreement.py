import argparse
import numpy
import pandas
import json
import os
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from evaluate_alignment import binarize_alignment_labels


def fleiss_kappa_score(labels):

    # import pdb
    # pdb.set_trace()

    table = numpy.array([numpy.array(list(rater_labels.values())).astype(int)
                         for rater_labels in labels]).T

    cat_table, _ = aggregate_raters(table)
    score = fleiss_kappa(cat_table)

    return score


def compute_scores(alignments):

    binary_labeled_pairs = [binarize_alignment_labels(alignment)
                            for alignment in alignments]

    scores = {'cohen': [], 'fleiss': []}

    for i, labeled_pairs1 in enumerate(binary_labeled_pairs):
        for labeled_pairs2 in binary_labeled_pairs[i + 1:]:

            assert list(labeled_pairs1.keys()) == list(labeled_pairs2.keys()),\
                "Segment indices in alignments do not match"

            score = cohen_kappa_score(list(labeled_pairs1.values()),
                                      list(labeled_pairs2.values()))
            scores['cohen'].append(score)

    scores['cohen'] = numpy.mean(scores['cohen'])

    fleiss_score = fleiss_kappa_score(labels=binary_labeled_pairs)
    scores['fleiss'] = fleiss_score

    n_binary_labeled_pairs = len(binary_labeled_pairs[0])

    return n_binary_labeled_pairs, scores


def get_diff_pairs(alignments):

    diff_pairs = []
    for i, cur_alignment in enumerate(alignments):

        diffs = []

        other_alignments = [alignment for j, alignment in enumerate(alignments)
                            if i != j]

        # cur_pairs = cur_alignment.values()
        other_pairs = set([(tuple(orig_nums), tuple(abridged_nums),)
                           for alignment in other_alignments
                           for orig_nums, abridged_nums in zip(alignment['original_segment_nums'],
                                                               alignment['abridged_segment_nums'])])

        for (orig_nums, orig_segs,
             abridged_nums, abridged_segs) in zip(cur_alignment['original_segment_nums'],
                                                  cur_alignment['original_segments'],
                                                  cur_alignment['abridged_segment_nums'],
                                                  cur_alignment['abridged_segments']):
            if (tuple(orig_nums), tuple(abridged_nums),) not in other_pairs:
                diff = {'original_segment_nums': orig_nums,
                        'original_segments': orig_segs,
                        'abridged_segment_nums': abridged_nums,
                        'abridged_segments': abridged_segs}
                diffs.append(diff)

        diff_pairs.append(diffs)

    return diff_pairs


def process(alignment_dirs,
            meta_data_file,
            scores_file,
            diffs_file):

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    chapter_scores = {'book_id': [],
                      'chapter_idx': [],
                      'n_binary_labels': [],
                      'mean_pairwise_cohen_kappa': [],
                      'fleiss_kappa': [],
                      }

    all_diff_pairs = {'alignment_id': [],
                      'book_id': [],
                      'chapter_idx': [],
                      'original_segment_nums': [],
                      'original_segments': [],
                      'abridged_segment_nums': [],
                      'abridged_segments': []}

    for book_id, book_info in meta_data.items():

        for chapter_idx in book_info['chapter_idxs']:

            print("Comparing alignments for {} chapter idx {}".format(
                book_id, chapter_idx))

            # import pdb
            # pdb.set_trace()

            alignments = []
            for alignment_dir in alignment_dirs:
                filepath = os.path.join(alignment_dir, book_id,
                                        "{}.json".format(chapter_idx))
                if not os.path.exists(filepath):
                    continue
                with open(filepath) as f:
                    alignments.append(json.load(f))

            if len(alignments) < 2:
                continue

            n_binary_labels, scores = compute_scores(alignments)

            if diffs_file:
                diff_pairs = get_diff_pairs(alignments)
                for pairs, alignment_dir in zip(diff_pairs, alignment_dirs):
                    for pair in pairs:
                        all_diff_pairs['alignment_id'].append(alignment_dir)
                        all_diff_pairs['book_id'].append(book_id)
                        all_diff_pairs['chapter_idx'].append(chapter_idx)
                        all_diff_pairs['original_segment_nums'].append(
                            pair['original_segment_nums'])
                        all_diff_pairs['original_segments'].append(
                            pair['original_segments'])
                        all_diff_pairs['abridged_segment_nums'].append(
                            pair['abridged_segment_nums'])
                        all_diff_pairs['abridged_segments'].append(
                            pair['abridged_segments'])

            chapter_scores['book_id'].append(book_id)
            chapter_scores['chapter_idx'].append(chapter_idx)
            chapter_scores['n_binary_labels'].append(n_binary_labels)
            chapter_scores['mean_pairwise_cohen_kappa'].append(scores['cohen'])
            chapter_scores['fleiss_kappa'].append(scores['fleiss'])

    chapter_scores = pandas.DataFrame(chapter_scores)
    if scores_file:
        chapter_scores.to_csv(scores_file, index=False)
        print("Saved scores to", scores_file)

    if diffs_file:
        all_diff_pairs = pandas.DataFrame(all_diff_pairs)
        # import pdb
        # pdb.set_trace()
        all_diff_pairs['first_orig_seg_num'] = all_diff_pairs[['original_segment_nums',
                                                               'abridged_segment_nums']].apply(
            lambda nums: (nums[0][0] if len(nums[0])
                          else (nums[1][0] if len(nums[1]) else 0)),
            axis=1)

        all_diff_pairs = all_diff_pairs.sort_values(by=['book_id',
                                                        'chapter_idx',
                                                        'first_orig_seg_num'])
        all_diff_pairs = all_diff_pairs.drop(labels='first_orig_seg_num',
                                             axis=1)
        all_diff_pairs.to_csv(diffs_file, index=False)
        print("Saved {} disagreeing alignments to {}".format(len(all_diff_pairs),
                                                             diffs_file))

    total_n_labels = chapter_scores['n_binary_labels'].sum()
    score_weights = chapter_scores['n_binary_labels'] / total_n_labels
    total_mean_cohen_kappa = numpy.average(chapter_scores['mean_pairwise_cohen_kappa'],
                                           weights=score_weights)

    print("\n{:<20}{:>10.3f}".format('Total Weighted Mean Pairwise Cohen Kappa',
                                     total_mean_cohen_kappa))

    total_mean_fleiss_kappa = numpy.average(chapter_scores['fleiss_kappa'],
                                            weights=score_weights)

    print("\n{:<20}{:>10.3f}".format('Total Weighted Mean Fleiss Kappa',
                                     total_mean_fleiss_kappa))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute Cohen's and Fleiss kappa inter-annotator agreement in alignments.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--alignment_dirs", "-alignment_dirs",
                        help="Directory paths for each alignment to be compared (two or more).\
                        When there are more than two alignments, score will be average of all pairwise kappa scores.",
                        type=str, nargs="+", required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file",
                        help="Path to .json file with book/chapter index.",
                        type=str, required=True)
    parser.add_argument("--scores_file", "-scores_file",
                        help="Filepath to .csv where agreement scores will be saved.\
                        If not given, scores will not be saved.",
                        type=str, required=False)
    parser.add_argument("--diffs_file", "-diffs_file",
                        help="Filepath to .csv where disagreeing alignment instances between validators will be saved.\
                        Each row in this file contains an aligned pair that was specified by the indicated validator\
                        but not by other validators.\
                        If filepath not given, disagreements will not be saved.",
                        type=str, required=False)

    args = parser.parse_args()

    process(args.alignment_dirs,
            args.meta_data_file,
            args.scores_file,
            args.diffs_file)
