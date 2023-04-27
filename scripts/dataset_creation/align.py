import argparse
import os
import json
import numpy
import pickle
import pandas
import csv
import collections
from copy import copy
from sklearn.metrics.pairwise import cosine_similarity
from nltk import word_tokenize
import rouge  # pip install py-rouge
from rouge import Rouge


class FastRouge(Rouge):
    '''Enables pre-computing of text tokens to speed up ROUGE scoring'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def text_to_item(self, text):
        sents = self._preprocess_summary_per_sentence(text)
        unigrams_dict, count = self._get_unigrams(sents)
        tokenized_sents = [sent.split() for sent in sents]
        item = {'tokenized_sents': tokenized_sents,
                'unigrams_dict': unigrams_dict,
                'count': count}
        return item

    def get_score(self, item1, item2):

        def _lcs(x, y):
            m = len(x)
            n = len(y)
            vals = collections.defaultdict(int)
            dirs = collections.defaultdict(int)

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        vals[i, j] = vals[i - 1, j - 1] + 1
                        dirs[i, j] = '|'
                    elif vals[i - 1, j] >= vals[i, j - 1]:
                        vals[i, j] = vals[i - 1, j]
                        dirs[i, j] = '^'
                    else:
                        vals[i, j] = vals[i, j - 1]
                        dirs[i, j] = '<'

            return vals, dirs

        def _mark_lcs(mask, dirs, m, n):
            while m != 0 and n != 0:
                if dirs[m, n] == '|':
                    m -= 1
                    n -= 1
                    mask[m] = 1
                elif dirs[m, n] == '^':
                    m -= 1
                elif dirs[m, n] == '<':
                    n -= 1
                else:
                    raise UnboundLocalError('Illegal move')

            return mask

        unigrams_dict1 = copy(item1['unigrams_dict'])
        unigrams_dict2 = copy(item2['unigrams_dict'])

        overlapping_count = 0.0
        for tokens2 in item2['tokenized_sents']:
            hit_mask = [0 for _ in range(len(tokens2))]

            for tokens1 in item1['tokenized_sents']:
                _, lcs_dirs = _lcs(tokens2, tokens1)
                _mark_lcs(hit_mask, lcs_dirs,
                          len(tokens2), len(tokens1))

            for token2_id, val in enumerate(hit_mask):
                if val == 1:
                    token = tokens2[token2_id]
                    if unigrams_dict1[token] > 0 and unigrams_dict2[token] > 0:
                        unigrams_dict1[token] -= 1
                        unigrams_dict2[token2_id] -= 1

                        overlapping_count += 1

        score = Rouge._compute_p_r_f_score(item1['count'],
                                           item2['count'],
                                           overlapping_count,
                                           self.alpha)

        return score


class Alignment():

    def __init__(self,
                 scoring_fn,
                 src_items,
                 tgt_items,
                 src_par_idxs,
                 tgt_par_idxs,
                 size_penalty=0.03,
                 skip_penalty=0.0,
                 min_segment_score=0.0,
                 one_to_one_paragraph=False,
                 max_tgt_lead=10,
                 max_src_trail_percent=0.25):
        self.scoring_fn = scoring_fn

        self.src_items = src_items
        self.n_src_segs = len(src_items[1])
        self.max_src_window = len(src_items)
        self.src_par_idxs = src_par_idxs

        self.tgt_items = tgt_items
        self.n_tgt_segs = len(tgt_items[1])
        self.max_tgt_window = len(tgt_items)
        self.tgt_par_idxs = tgt_par_idxs

        self.size_penalty = size_penalty
        self.skip_penalty = skip_penalty

        self.accum_scores = numpy.zeros((self.n_src_segs,
                                         self.n_tgt_segs))
        self.segment_scores = numpy.zeros((self.n_src_segs,
                                           self.n_tgt_segs))

        self.backtraces = numpy.ones((self.n_src_segs,
                                      self.n_tgt_segs, 2)).astype('int').astype('object') * -1

        self.alignments = [[{} for _ in range(self.n_tgt_segs)]
                           for _ in range(self.n_src_segs)]

        self.min_segment_score = min_segment_score
        self.one_to_one_paragraph = one_to_one_paragraph
        self.max_tgt_lead = max_tgt_lead
        self.max_src_trail_percent = max_src_trail_percent

    def __call__(self):
        for tgt_i in range(self.n_tgt_segs):
            for src_i in range(self.n_src_segs):
                if tgt_i - src_i > self.max_tgt_lead:
                    continue
                if (self.n_src_segs >= 50
                        and (tgt_i + 1) / (src_i + 1) < self.max_src_trail_percent):
                    break
                if src_i % 50 == 0 and tgt_i % 50 == 0:
                    print(src_i, tgt_i)
                self.update(src_i, tgt_i)
        return self.get_best_alignment()

    def update(self, src_i, tgt_i):
        window_sizes = [(src_window_size, tgt_window_size)
                        for src_window_size in range(1, self.max_src_window + 1)
                        for tgt_window_size in range(0, self.max_tgt_window + 1)]
        # Sort by alignnment sizes in order to favor shorter alignments
        window_sizes = sorted(window_sizes,
                              key=lambda window: window[0] + window[1])
        (accum_scores,
         segment_scores,
         backtraces,
         alignments) = zip(*[self.src_to_tgt(src_i, tgt_i,
                                             src_window_size=src_window_size,
                                             tgt_window_size=tgt_window_size).values()
                             for src_window_size, tgt_window_size in window_sizes
                             ])

        best_choice = numpy.argmax(accum_scores)
        self.accum_scores[src_i, tgt_i] = accum_scores[best_choice]
        self.segment_scores[src_i, tgt_i] = segment_scores[best_choice]
        self.backtraces[src_i, tgt_i] = backtraces[best_choice]
        self.alignments[src_i][tgt_i] = alignments[best_choice]

    def src_to_tgt(self, src_i, tgt_i, src_window_size, tgt_window_size):

        alignment_size = max(src_window_size, tgt_window_size)
        src_i_offset = src_window_size - 1
        tgt_i_offset = tgt_window_size - 1
        accum_score = 0.0
        segment_score = 0.0

        if (self.one_to_one_paragraph and
            (len(set(self.src_par_idxs[src_i - src_i_offset: src_i + 1])) > 1
                or len(set(self.tgt_par_idxs[tgt_i - tgt_i_offset: tgt_i + 1])) > 1)):
            accum_score = -numpy.inf
            segment_score = -numpy.inf

        else:
            if (src_i - src_window_size >= 0 and tgt_i - tgt_window_size >= 0):
                accum_score = self.accum_scores[src_i - src_window_size,
                                                tgt_i - tgt_window_size]
                if src_window_size == 0 or tgt_window_size == 0:
                    accum_score -= self.skip_penalty
            if (src_window_size > 0 and tgt_window_size > 0
                    and src_i - src_i_offset >= 0 and tgt_i - tgt_i_offset >= 0):
                src_item = self.src_items[src_window_size][src_i - src_i_offset]
                tgt_item = self.tgt_items[tgt_window_size][tgt_i - tgt_i_offset]
                segment_score = max(0, self.scoring_fn(src_item,
                                                       tgt_item)
                                    - ((alignment_size - 1) * self.size_penalty))
                if segment_score >= self.min_segment_score:
                    accum_score += (segment_score * tgt_window_size)
                else:
                    segment_score = 0.0

        backtrace = (src_i - src_window_size, tgt_i - tgt_window_size)
        alignment = {'src': list(range(src_i - src_i_offset,
                                       min(src_i + 1, self.n_src_segs))),
                     'tgt': list(range(tgt_i - tgt_i_offset,
                                       min(tgt_i + 1, self.n_tgt_segs)))}

        return {'accum_score': accum_score,
                'segment_score': segment_score,
                'backtrace': backtrace,
                'alignment': alignment,
                }

    def get_best_alignment(self):

        end_backtrace = (self.n_src_segs - 1,
                         self.n_tgt_segs - 1)

        best_alignment = []

        src_i, tgt_i = end_backtrace

        while (src_i >= 0 or tgt_i >= 0):

            if ((src_i < 0 and tgt_i >= 0)
                or (src_i >= 0 and tgt_i < 0)
                or (not self.alignments[src_i][tgt_i]['src']
                    and not self.alignments[src_i][tgt_i]['tgt'])):
                for i in range(tgt_i, -1, -1):
                    alignment = {'src': [],
                                 'tgt': [i],
                                 'accum_score': 0.0,
                                 'segment_score': 0.0}
                    best_alignment.insert(0, alignment)
                for i in range(src_i, -1, -1):
                    alignment = {'src': [i],
                                 'tgt': [],
                                 'accum_score': 0.0,
                                 'segment_score': 0.0}
                    best_alignment.insert(0, alignment)
                break

            alignment = {**self.alignments[src_i][tgt_i],
                         'accum_score': self.accum_scores[src_i, tgt_i],
                         'segment_score': self.segment_scores[src_i, tgt_i]}

            best_alignment.insert(0, alignment)

            src_i, tgt_i = self.backtraces[src_i, tgt_i]

        return best_alignment


def align(scoring_method,
          segments_in_dir,
          vectors_in_dir,
          meta_data_file,
          out_dir,
          max_src_window=4,
          max_tgt_window=4,
          size_penalty=0.03,
          skip_penalty=0.0,
          min_segment_score=0.0,
          one_to_one_paragraph=False,
          ngram_size=1):

    if scoring_method == 'string':
        if ngram_size == 1:
            rouge_scorer = FastRouge(stemming=False)

            def scoring_fn(src, tgt):
                return rouge_scorer.get_score(src, tgt)['r']
        else:
            rouge_scorer = Rouge(metrics=['rouge-n'],
                                 max_n=ngram_size,
                                 stemming=False)

            def scoring_fn(src, tgt):
                return rouge_scorer.get_scores(hypothesis=[src],
                                               references=[[tgt]])['rouge-{}'.format(ngram_size)]['r']

    elif scoring_method == 'vector':
        def scoring_fn(src, tgt): return cosine_similarity(src[None],
                                                           tgt[None])[0, 0]

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    auto_output_dir = os.path.join(out_dir, 'json')
    if not os.path.exists(auto_output_dir):
        os.mkdir(auto_output_dir)

    val_sheet_dir = os.path.join(out_dir, 'sheets')
    if not os.path.exists(val_sheet_dir):
        os.mkdir(val_sheet_dir)

    for book, book_info in meta_data.items():

        book_auto_output_dir = os.path.join(auto_output_dir, book)
        if not os.path.exists(book_auto_output_dir):
            os.mkdir(book_auto_output_dir)

        book_val_sheet_dir = os.path.join(val_sheet_dir, book)
        if not os.path.exists(book_val_sheet_dir):
            os.mkdir(book_val_sheet_dir)

        for chapter_idx, chapter_title in zip(book_info['chapter_idxs'], book_info['chapter_titles']):

            aligned_data = {
                'original_paragraph_nums': [],
                'original_segment_nums': [],
                'original_segments': [],
                'abridged_paragraph_nums': [],
                'abridged_segment_nums': [],
                'abridged_segments': [],
                'accumulated_score': [],
                'segment_score': []
            }

            with open(os.path.join(os.path.join(segments_in_dir, 'original',
                                                book, "{}.json".format(chapter_idx)))) as f:
                src_data = json.load(f)
                if 'paragraph_idxs' in src_data:
                    src_data['paragraph_nums'] = [
                        idx + 1 for idx in src_data['paragraph_idxs']]
                    src_data.pop('paragraph_idxs')
                if 'sentences' in src_data:
                    src_data['segments'] = src_data['sentences']
                    src_data.pop('sentences')

            with open(os.path.join(os.path.join(segments_in_dir, 'abridged',
                                                book, "{}.json".format(chapter_idx)))) as f:
                tgt_data = json.load(f)
                if 'paragraph_idxs' in tgt_data:
                    tgt_data['paragraph_nums'] = [
                        idx + 1 for idx in tgt_data['paragraph_idxs']]
                    tgt_data.pop('paragraph_idxs')
                if 'sentences' in tgt_data:
                    tgt_data['segments'] = tgt_data['sentences']
                    tgt_data.pop('sentences')

            if scoring_method == 'string':
                if ngram_size == 1:
                    src_items = {window_size: [rouge_scorer.text_to_item("".join(src_data['segments'][i: i + window_size]))
                                               for i in range(0, len(src_data['segments']))]
                                 for window_size in range(1, max_src_window + 1)}

                    tgt_items = {window_size: [rouge_scorer.text_to_item("".join(tgt_data['segments'][i: i + window_size]))
                                               for i in range(0, len(tgt_data['segments']))]
                                 for window_size in range(1, max_tgt_window + 1)}
                else:
                    src_items = {window_size: ["".join(src_data['segments'][i: i + window_size])
                                               for i in range(0, len(src_data['segments']))]
                                 for window_size in range(1, max_src_window + 1)}

                    tgt_items = {window_size: ["".join(tgt_data['segments'][i: i + window_size])
                                               for i in range(0, len(tgt_data['segments']))]
                                 for window_size in range(1, max_tgt_window + 1)}

            elif scoring_method == 'vector':
                assert vectors_in_dir is not None
                with open(os.path.join(os.path.join(vectors_in_dir, 'original',
                                                    book, "{}.pkl".format(chapter_idx))), 'rb') as f:
                    src_items = {window_size: vectors for window_size, vectors
                                 in pickle.load(f).items() if window_size <= max_src_window}

                with open(os.path.join(os.path.join(vectors_in_dir, 'abridged',
                                                    book, "{}.pkl".format(chapter_idx))), 'rb') as f:

                    tgt_items = {window_size: vectors for window_size, vectors
                                 in pickle.load(f).items() if window_size <= max_tgt_window}

            print("Computing alignment for {} {}...".format(book, chapter_title))
            alignment = Alignment(scoring_fn=scoring_fn,
                                  src_items=src_items,
                                  tgt_items=tgt_items,
                                  src_par_idxs=[
                                      num - 1 for num in src_data['paragraph_nums']],
                                  tgt_par_idxs=[
                                      num - 1 for num in tgt_data['paragraph_nums']],
                                  size_penalty=size_penalty,
                                  skip_penalty=skip_penalty,
                                  min_segment_score=min_segment_score,
                                  one_to_one_paragraph=one_to_one_paragraph)()

            for pair in alignment:
                src_seg_idxs, tgt_seg_idxs = pair['src'], pair['tgt']

                src_par_nums = [src_data['paragraph_nums'][i]
                                for i in src_seg_idxs]
                aligned_data['original_paragraph_nums'].append(src_par_nums)
                aligned_data['original_segment_nums'].append(
                    [idx + 1 for idx in src_seg_idxs])
                aligned_data['original_segments'].append([src_data['segments'][i]
                                                          for i in src_seg_idxs])

                tgt_par_nums = [tgt_data['paragraph_nums'][i]
                                for i in tgt_seg_idxs]
                aligned_data['abridged_paragraph_nums'].append(tgt_par_nums)
                aligned_data['abridged_segment_nums'].append(
                    [idx + 1 for idx in tgt_seg_idxs])
                aligned_data['abridged_segments'].append([tgt_data['segments'][i]
                                                          for i in tgt_seg_idxs])
                aligned_data['accumulated_score'].append(pair['accum_score'])
                aligned_data['segment_score'].append(pair['segment_score'])

            auto_output_file = os.path.join(book_auto_output_dir,
                                            "{}.json".format(chapter_idx))
            with open(auto_output_file, 'w') as f:
                json.dump(aligned_data, f, indent=4)

            rows = []
            for (_, orig_seg_nums, orig_segs,
                 _, abridged_seg_nums, abridged_segs,
                 accum_score, segment_score) in zip(*aligned_data.values()):
                orig_item = "\t".join(["[{}] {}".format(num, seg)
                                       for num, seg in zip(orig_seg_nums, orig_segs)])
                abridged_item = "\t".join(["[{}] {}".format(num, seg)
                                           for num, seg in zip(abridged_seg_nums, abridged_segs)])
                rows.append((orig_item, abridged_item,
                             accum_score, segment_score))
            rows = pandas.DataFrame(rows)
            val_sheet_file = os.path.join(book_val_sheet_dir,
                                          '{}.csv'.format(chapter_idx))
            rows.to_csv(val_sheet_file,
                        header=["Original_Segments",
                                "Abridged_Segments",
                                "Accum_Scores",
                                "Scores",
                                ],
                        index=False,
                        quoting=csv.QUOTE_ALL)

            print("Saved aligned pairs to {} and validation sheet to {}".format(auto_output_file,
                                                                                val_sheet_file))

    print("\nSaved all output alignment data to:", auto_output_dir)
    print("\nSaved all user spreadsheets for alignment validation to:", val_sheet_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute segment alignments between original text and its simplified/abridged version.\
                                                This is done by computing vector similarity between segments and\
                                                applying a dynamic programming algorithm to find the highest-scoring alignment across all segments.\
                                                This alignment may be one-to-one, many-to-one, or many-to-many.\
                                                This script assumes you have run transform.py in order to compute the segment vectors.\
                                                The resulting aligned pairs are grouped by book and then by chapter.)",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--scoring_method", "-scoring_method",
                        help="Approach for scoring similarity between segments.\
                        The 'string' method uses the ROUGE metric directly applied to the text segments.\
                        The 'vector' method calculates cosine similarity between pre-computed segment vectors.",
                        type=str, required=True, choices=['vector', 'string'])
    parser.add_argument("--ngram_size", "-ngram_size",
                        help="If scoring_method == 'string', specify the size n for ngrams that should be used to compute\
                        ROUGE overlap. By default, unigrams (n=1) will be used.",
                        type=int, default=1, required=False, choices=[1, 2, 3, 4])
    parser.add_argument("--segments_in_dir", "-segments_in_dir",
                        help="Directory path containing text sentence-segmented data (the output of split_segments.py).\
                        There is a single .json file per chapter in each original and abridged book",
                        type=str, required=True)
    parser.add_argument("--vectors_in_dir", "-vectors_in_dir",
                        help="Directory path containing segment vectors (the output of transform.py).\
                        There is a single .pkl file per chapter in each original and abridged book.\
                        Not required if the alignment method doesn't require vectors (i.e. string method).",
                        type=str, required=False)
    parser.add_argument("--meta_data_file", "-meta_data_file",
                        help="Path to .json file with book/chapter index.",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir",
                        help="Directory path where the alignment data will be saved (json files)",
                        type=str, required=True)
    parser.add_argument("--max_src_window", "-max_src_window",
                        help="Max number of source text segments allowed in a single aligned pair.\
                        This cannot exceed the max window size of the source vectors in the input data.",
                        type=int, default=4, required=False)
    parser.add_argument("--max_tgt_window", "-max_tgt_window",
                        help="Max number of target text segments allowed in a single aligned pair.\
                        This cannot exceed the max window size of the target vectors in the input data.",
                        type=int, default=4, required=False)
    parser.add_argument("--size_penalty", "-size_penalty",
                        help="Scoring penalty for alignment length, such that longer alignments are penalized more.\
                        This promotes finding shorter alignments.",
                        type=float, default=0.03, required=False)
    parser.add_argument("--skip_penalty", "-skip_penalty",
                        help="Scoring penalty for skipping over segments in alignment (i.e. segment mapped to null).",
                        type=float, default=0.0, required=False)
    parser.add_argument("--min_segment_score", "-min_segment_score",
                        help="Minimum segment similarity score required to consider candidate alignment.\
                        Alignments beneath this score will be pruned from consideration.",
                        type=float, default=0.0, required=False)
    parser.add_argument("--one_to_one_paragraph", "-one_to_one_paragraph",
                        help="Assume a one-to-one paragraph alignment between source and target texts,\
                        such that both the source and target segments in a single alignment pair\
                        must be part of the same paragraph in their respective texts.\
                        Qualitative analysis indicates most gold-standard aligned pairs follow this assumption,\
                        but there are exceptions.",
                        action='store_true', required=False)

    args = parser.parse_args()

    align(args.scoring_method,
          args.segments_in_dir,
          args.vectors_in_dir,
          args.meta_data_file,
          args.out_dir,
          args.max_src_window,
          args.max_tgt_window,
          args.size_penalty,
          args.skip_penalty,
          args.min_segment_score,
          args.one_to_one_paragraph,
          args.ngram_size)
