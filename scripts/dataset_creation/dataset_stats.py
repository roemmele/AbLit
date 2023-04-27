import argparse
import string
import pandas
import numpy
import json
from nltk.tokenize import word_tokenize
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__)))))
from ablit import AblitDataset
from transformers import RobertaTokenizer
from rouge import Rouge  # pip install py-rouge

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

rouge_scorer = Rouge(metrics=['rouge-n'],
                     max_n=1,
                     limit_length=False,
                     stemming=False)


def get_token_ids(tokens):
    tok_counts = {}
    tok_ids = set()
    for tok in tokens:
        if tok not in tok_counts:
            tok_counts[tok] = 0
        else:
            tok_counts[tok] += 1
        tok_ids.add("{}_{}".format(tok, tok_counts[tok]))
    return tok_ids


def get_n_reorders(overlaps):
    n_reorders = 0
    prev_start_char = None
    for span in overlaps:
        if prev_start_char and span.start_char < prev_start_char:
            n_reorders += 1
        prev_start_char = span.start_char

    return n_reorders


def get_rouge_scores(book):

    stats = {'rouge-p': [],
             'rouge-r': [],
             'rouge-f1': []}
    for chapter in book.chapters:
        for row in chapter.rows:
            scores = rouge_scorer.get_scores(hypothesis=[row.abridged.strip()],
                                             references=[[row.original.strip()]])['rouge-1']
            stats['rouge-p'].append(scores['p'])
            stats['rouge-r'].append(scores['r'])
            stats['rouge-f1'].append(scores['f'])

    return stats


def get_rouge_p_bins(rouge_p_scores):
    # Binned scores
    rouge_p_bins = pandas.cut(rouge_p_scores,
                              bins=[-0.0001, 0.0, 0.25, 0.5, 0.75, 0.9999, 1.0])  # .values
    return rouge_p_bins


def get_row_stats(book):
    stats = {
        'n_orig_words': [],
        'n_abrg_words': [],
        'in_orig_not_abrg': [],
        'in_abrg_not_orig': [],
        'in_both_orig_abrg': [],
        'has_remove': [],
        'has_preserve': [],
        'has_add': [],
        'has_reorder': [],
        'is_identical': []
    }

    for chapter in book.chapters:
        for row in chapter.rows:

            orig_toks = [tok.lower()
                         for tok in word_tokenize(row.original.strip())]
            orig_tok_set = get_token_ids(orig_toks)
            stats['n_orig_words'].append(len(orig_tok_set))

            abrg_toks = [tok.lower()
                         for tok in word_tokenize(row.abridged.strip())]
            abrg_tok_set = get_token_ids(abrg_toks)
            stats['n_abrg_words'].append(len(abrg_tok_set))

            tok_intersection = orig_tok_set.intersection(abrg_tok_set)

            orig_not_abrg_tok_set = orig_tok_set - tok_intersection
            stats['in_orig_not_abrg'].append(len(orig_not_abrg_tok_set))

            abrg_not_orig_tok_set = abrg_tok_set - tok_intersection
            stats['in_abrg_not_orig'].append(len(abrg_not_orig_tok_set))

            stats["in_both_orig_abrg"].append(len(tok_intersection))

            stats['has_remove'].append(len(orig_not_abrg_tok_set) > 0)
            stats['has_preserve'].append(len(tok_intersection) > 0)
            stats['has_add'].append(len(abrg_not_orig_tok_set) > 0)

            n_reorders = get_n_reorders(row.overlaps)
            stats['has_reorder'].append(n_reorders > 0)

            stats['is_identical'].append(
                True if abrg_toks == orig_toks else False)

    return stats


def get_row_size_stats(book):
    counts = {'n_1_to_0': 0,
              'n_1_to_1': 0,
              'n_1_to_2+': 0,
              'n_2+_to_2+': 0,
              'n_2+_to_1': 0}

    for chapter in book.chapters:
        for oracle in chapter.rows:
            if oracle.n_original_sentences == 1 and oracle.n_abridged_sentences == 0:
                counts['n_1_to_0'] += 1
            elif oracle.n_original_sentences == 1 and oracle.n_abridged_sentences == 1:
                counts['n_1_to_1'] += 1
            elif oracle.n_original_sentences == 1 and oracle.n_abridged_sentences > 1:
                counts['n_1_to_2+'] += 1
            elif oracle.n_original_sentences > 1 and oracle.n_abridged_sentences > 1:
                counts['n_2+_to_2+'] += 1
            elif oracle.n_original_sentences > 1 and oracle.n_abridged_sentences == 1:
                counts['n_2+_to_1'] += 1

    return counts


def get_size_stats(book):

    chapter_stats = {'n_orig_toks': [],
                     'n_abrg_toks': [],
                     'n_orig_words': [],
                     'n_abrg_words': [],
                     'n_orig_pars': [],
                     'n_abrg_pars': [],
                     'n_orig_sents': [],
                     'n_abrg_sents': [],
                     'n_align_rows': []}

    for chapter in book.chapters:

        n_orig_words = len(word_tokenize(
            chapter.original_version.text))
        n_abrg_words = len(word_tokenize(
            chapter.abridged_version.text))
        chapter_stats['n_orig_words'].append(n_orig_words)
        chapter_stats['n_abrg_words'].append(n_abrg_words)

        n_orig_toks = len(tokenizer.tokenize(
            chapter.original_version.text))
        n_abrg_toks = len(tokenizer.tokenize(
            chapter.abridged_version.text))
        chapter_stats['n_orig_toks'].append(n_orig_toks)
        chapter_stats['n_abrg_toks'].append(n_abrg_toks)

        n_orig_pars = len(list(chapter.original_version.paragraphs))
        n_abrg_pars = len(list(chapter.abridged_version.paragraphs))
        chapter_stats['n_orig_pars'].append(n_orig_pars)
        chapter_stats['n_abrg_pars'].append(n_abrg_pars)

        n_orig_sents = len(list(chapter.original_version.sentences))
        n_abrg_sents = len(list(chapter.abridged_version.sentences))
        chapter_stats['n_orig_sents'].append(n_orig_sents)
        chapter_stats['n_abrg_sents'].append(n_abrg_sents)

        rows = list(chapter.rows)
        chapter_stats['n_align_rows'].append(len(rows))

    return chapter_stats


def accumulate_row_stats(stats):

    accum_stats = {}

    accum_stats["in_orig_not_abrg_rate"] = (sum(stats["in_orig_not_abrg"])
                                            / sum(stats["n_orig_words"]))
    accum_stats["in_abrg_not_orig_rate"] = (sum(stats["in_abrg_not_orig"])
                                            / sum(stats["n_abrg_words"]))
    accum_stats["orig_in_abrg_rate"] = (sum(stats["in_both_orig_abrg"])
                                        / sum(stats["n_abrg_words"]))
    accum_stats["abrg_in_orig_rate"] = (sum(stats["in_both_orig_abrg"])
                                        / sum(stats["n_orig_words"]))

    accum_stats["row_remove_rate"] = (sum(stats["has_remove"])
                                      / len(stats['has_remove']))
    accum_stats["row_preserve_rate"] = (sum(stats["has_preserve"])
                                        / len(stats['has_preserve']))
    accum_stats["row_add_rate"] = (sum(stats["has_add"])
                                   / len(stats["has_add"]))
    accum_stats["row_reorder_rate"] = (sum(stats["has_reorder"])
                                       / len(stats["has_reorder"]))
    accum_stats["is_identical"] = (sum(stats["is_identical"])
                                   / len(stats["is_identical"]))

    return accum_stats


def report(dataset):

    stats_by_book = {}
    corpus_size_stats = []
    corpus_row_stats = []
    corpus_row_size_stats = []
    rouge_stats = []

    for book in dataset.books:

        print(book.book_id)

        stats = {}

        size_stats = pandas.DataFrame(get_size_stats(book))
        corpus_size_stats.append(size_stats)

        stats["book"] = size_stats.sum().to_dict()
        stats["book"]["n_chapters"] = len(size_stats)
        stats["book"]["percent_words"] = round((stats["book"]["n_abrg_words"] /
                                                stats["book"]["n_orig_words"]), 3)
        stats["book"]["percent_sents"] = round((stats["book"]["n_abrg_sents"] /
                                                stats["book"]["n_orig_sents"]), 3)

        stats["chapter"] = {("mean_" + key):
                            round(val, 0) for key, val in size_stats.mean().to_dict().items()}

        stats["chapter"].update({("median_" + key):
                                 round(val, 0) for key, val in size_stats.median().to_dict().items()})

        row_stats = get_row_stats(book)
        corpus_row_stats.append(row_stats)

        stats["align_row"] = {}
        accumed_row_stats = accumulate_row_stats(row_stats)
        stats["align_row"].update({key: round(val, 3)
                                   for key, val in accumed_row_stats.items()})

        row_size_stats = get_row_size_stats(book)
        corpus_row_size_stats.append(row_size_stats)
        row_count = sum(row_size_stats.values())

        stats["align_row"].update({key: round(val / row_count, 3)
                                   for key, val in row_size_stats.items()})

        rouge_scores = pandas.DataFrame(get_rouge_scores(book))
        for metric in rouge_scores:
            stats["align_row"][metric] = round(rouge_scores[metric].mean(), 3)

        rouge_scores['rouge-p-bins'] = get_rouge_p_bins(
            rouge_scores['rouge-p'])
        bin_dist = (rouge_scores.groupby('rouge-p-bins').count()
                    / len(rouge_scores))['rouge-p']
        bin_dist = {'rouge-p-bin-' + str(interval): round(val, 3) for interval, val
                    in bin_dist.to_dict().items()}
        stats['align_row'].update(bin_dist)

        rouge_stats.append(rouge_scores)

        stats_by_book[book.book_id] = stats

    all_stats = stats_by_book
    all_stats["all"] = {}

    corpus_size_stats = pandas.concat([pandas.DataFrame(book_stats)
                                       for book_stats in corpus_size_stats])
    corpus_row_size_stats = pandas.DataFrame(
        corpus_row_size_stats).sum()

    all_stats["all"]["corpus"] = corpus_size_stats.sum().to_dict()
    all_stats["all"]["corpus"]["n_chapters"] = len(corpus_size_stats)
    all_stats["all"]["corpus"]["percent_words"] = round((corpus_size_stats["n_abrg_words"].sum() /
                                                         corpus_size_stats["n_orig_words"].sum()), 3)
    all_stats["all"]["corpus"]["percent_sents"] = round((corpus_size_stats["n_abrg_sents"].sum() /
                                                         corpus_size_stats["n_orig_sents"].sum()), 3)

    all_stats["all"]["chapter"] = {("mean_" + key):
                                   round(val, 0) for key, val in corpus_size_stats.mean().to_dict().items()}

    all_stats["all"]["chapter"].update({("median_" + key):
                                        round(val, 0) for key, val in corpus_size_stats.median().to_dict().items()})

    row_count = corpus_row_size_stats.sum()
    all_stats["all"]["align_row"] = corpus_row_size_stats.divide(
        row_count).round(3).to_dict()

    corpus_row_stats = pandas.concat([pandas.DataFrame(book_stats)
                                      for book_stats in corpus_row_stats]).to_dict(orient='list')
    # import pdb
    # pdb.set_trace()
    accumed_row_stats = accumulate_row_stats(corpus_row_stats)

    all_stats["all"]["align_row"].update({key: round(val, 3)
                                          for key, val in accumed_row_stats.items()})

    rouge_stats = pandas.concat([pandas.DataFrame(book_stats)
                                 for book_stats in rouge_stats])
    for metric in rouge_stats:
        if metric == 'rouge-p-bins':
            bin_dist = (rouge_stats.groupby(metric).count()
                        / len(rouge_stats))['rouge-p']
            bin_dist = {'rouge-p-bin-' + str(interval): round(val, 3) for interval, val
                        in bin_dist.to_dict().items()}
            all_stats["all"]["align_row"].update(bin_dist)
        else:
            all_stats["all"]["align_row"][metric] = round(
                rouge_stats[metric].mean(), 3)

    return all_stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Report various descriptive statistics for Ablit data.")

    parser.add_argument("--ablit_dir", "-ablit_dir",
                        help="Directory path to AbLit dataset.",
                        type=str,
                        required=True)
    parser.add_argument("--output_file", "-output_file",
                        help="Filepath to which stats will be saved.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    stats = {}

    for partition in ("train", "dev", "test",):

        print("processing:", partition)
        dataset = AblitDataset(dirpath=args.ablit_dir,
                               partition=partition)

        stats[partition] = report(dataset)

    with open(args.output_file, "w") as f:
        json.dump(stats, f, indent=4)

    print("Saved stats to:", args.output_file)
