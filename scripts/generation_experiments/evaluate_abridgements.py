import argparse
import os
import json
import pandas
import math
from nltk.tokenize import sent_tokenize, word_tokenize
from rouge import Rouge  # pip install py-rouge
from ablit import AblitDataset

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def get_token_ids(passage, lowercase=True):
    tok_counts = {}
    tok_ids = set()
    if lowercase:
        passage = passage.lower()
    for tok in word_tokenize(passage):
        if tok not in tok_counts:
            tok_counts[tok] = 0
        else:
            tok_counts[tok] += 1
        tok_ids.add("{}_{}".format(tok, tok_counts[tok]))
    return tok_ids


def get_removal_scores(original,
                       pred_abridgement,
                       ref_abridgement):

    orig_tok_ids = get_token_ids(original)
    pred_tok_ids = get_token_ids(pred_abridgement)
    ref_tok_ids = get_token_ids(ref_abridgement)

    ref_rmv_toks = orig_tok_ids - ref_tok_ids
    pred_rmv_toks = orig_tok_ids - pred_tok_ids

    correct_rmv_toks = ref_rmv_toks.intersection(pred_rmv_toks)

    if pred_rmv_toks:
        precision = len(correct_rmv_toks) / len(pred_rmv_toks)
    elif ref_rmv_toks:
        precision = 0.0
    else:
        precision = 1.0

    assert precision <= 1.0

    if ref_rmv_toks:
        recall = len(correct_rmv_toks) / len(ref_rmv_toks)
    elif pred_rmv_toks:
        recall = 0.0
    else:
        recall = 1.0

    assert recall <= 1.0

    if precision + recall:
        f1 = 2.0 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    return {'precision': precision,
            'recall': recall,
            'f1': f1}


def get_preservation_scores(original,
                            pred_abridgement,
                            ref_abridgement):

    orig_tok_ids = get_token_ids(original)
    pred_tok_ids = get_token_ids(pred_abridgement)
    ref_tok_ids = get_token_ids(ref_abridgement)

    ref_prsv_toks = ref_tok_ids.intersection(orig_tok_ids)
    pred_prsv_toks = pred_tok_ids.intersection(orig_tok_ids)

    correct_prsv_toks = ref_prsv_toks.intersection(pred_prsv_toks)

    if pred_prsv_toks:
        precision = len(correct_prsv_toks) / len(pred_prsv_toks)
    elif ref_prsv_toks:
        precision = 0.0
    else:
        precision = 1.0

    assert precision <= 1.0

    if ref_prsv_toks:
        recall = len(correct_prsv_toks) / len(ref_prsv_toks)
    elif pred_prsv_toks:
        recall = 0.0
    else:
        recall = 1.0

    assert recall <= 1.0

    if precision + recall:
        f1 = 2.0 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    return {'precision': precision,
            'recall': recall,
            'f1': f1}


def get_addition_scores(original,
                        pred_abridgement,
                        ref_abridgement):

    orig_tok_ids = get_token_ids(original)
    pred_tok_ids = get_token_ids(pred_abridgement)
    ref_tok_ids = get_token_ids(ref_abridgement)

    ref_added_toks = ref_tok_ids - orig_tok_ids
    pred_added_toks = pred_tok_ids - orig_tok_ids

    correct_added_toks = ref_added_toks.intersection(pred_added_toks)

    if pred_added_toks:
        precision = len(correct_added_toks) / len(pred_added_toks)
    elif ref_added_toks:
        precision = 0.0
    else:
        precision = 1.0

    assert precision <= 1.0

    if ref_added_toks:
        recall = len(correct_added_toks) / len(ref_added_toks)
    elif pred_added_toks:
        recall = 0.0
    else:
        recall = 1.0

    assert recall <= 1.0

    if precision + recall:
        f1 = 2.0 * ((precision * recall) / (precision + recall))
    else:
        f1 = 0.0

    return {'precision': precision,
            'recall': recall,
            'f1': f1}


def rouge_scorer_interface(pred, ref):
    if pred.strip() == ref.strip():
        scores = {'p': 1.0, 'r': 1.0, 'f': 1.0}
        return scores

    scorer = Rouge(metrics=['rouge-n', 'rouge-l'],
                   max_n=4,
                   limit_length=False,
                   stemming=False)
    scores = scorer.get_scores(hypothesis=[pred],
                               references=[[ref]])

    scores = {'rouge-L': scores['rouge-l'],
              'rouge-1': scores['rouge-1'],
              'rouge-2': scores['rouge-2']}

    return scores


def get_rouge_scores(pred_abridgement,
                     ref_abridgement):
    scores = rouge_scorer_interface(pred=pred_abridgement,
                                    ref=ref_abridgement)
    return scores


def get_all_scores(original,
                   pred_abridgement,
                   ref_abridgement):

    scores = {}

    rouge_scores = get_rouge_scores(
        pred_abridgement=pred_abridgement,
        ref_abridgement=ref_abridgement)
    scores['rouge-L-precision'] = rouge_scores['rouge-L']['p']
    scores['rouge-L-recall'] = rouge_scores['rouge-L']['r']
    scores['rouge-L-f1'] = rouge_scores['rouge-L']['f']

    removal_scores = get_removal_scores(
        original=original,
        pred_abridgement=pred_abridgement,
        ref_abridgement=ref_abridgement)
    scores['removal-precision'] = removal_scores['precision']
    scores['removal-recall'] = removal_scores['recall']
    scores['removal-f1'] = removal_scores['f1']

    preservation_scores = get_preservation_scores(
        original=original,
        pred_abridgement=pred_abridgement,
        ref_abridgement=ref_abridgement)
    scores['preservation-precision'] = preservation_scores['precision']
    scores['preservation-recall'] = preservation_scores['recall']
    scores['preservation-f1'] = preservation_scores['f1']

    addition_scores = get_addition_scores(
        original=original,
        pred_abridgement=pred_abridgement,
        ref_abridgement=ref_abridgement)
    scores['addition-precision'] = addition_scores['precision']
    scores['addition-recall'] = addition_scores['recall']
    scores['addition-f1'] = addition_scores['f1']

    scores['pred_length'] = len(word_tokenize(pred_abridgement))
    scores['ref_length'] = len(word_tokenize(ref_abridgement))

    return scores


def restructure_pred_abridged(pred_abridgements):
    restr_pred_abridgements = {}
    for item in pred_abridgements:
        if item['book_id'] not in restr_pred_abridgements:
            restr_pred_abridgements[item['book_id']] = {}
        restr_pred_abridgements[item['book_id']
                                ][item['chapter_idx']] = item['predicted_abridgement']
    return restr_pred_abridgements


def evaluate(ablit_data,
             pred_abridgements):

    pred_abridgements = restructure_pred_abridged(pred_abridgements)

    eval_data = []

    for book in ablit_data.books:
        print("Evaluating items for book {}...".format(book.book_id))
        for chapter in book.chapters:
            pred_abridgement = pred_abridgements[book.book_id][chapter.chapter_idx]
            scores = get_all_scores(original=chapter.original,
                                    pred_abridgement=pred_abridgement,
                                    ref_abridgement=chapter.abridged)
            eval_data.append({'book_id': book.book_id,
                              'chapter_idx': chapter.chapter_idx,
                              **scores})
        #     break
        # break
    return eval_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Score quality of generated abridgements according to automated metrics.")

    parser.add_argument("--ablit_dir", "-ablit_dir",
                        help="Directory path to AbLit dataset.",
                        type=str,
                        required=False)
    parser.add_argument("--partition", "-partition",
                        help="Partition of AbLit dataset (train, dev, or test) associated with predicted abridgements\
                        in -pred_abridged_file.",
                        type=str,
                        required=False,
                        default='test')
    parser.add_argument("--pred_abridged_file", "-pred_abridged_file",
                        help="JSON file with predicted abridgements for each chapter in each book.\
                        JSON object consists of a list of items, each with a 'book_id', 'chapter_idx',\
                        and 'predicted_abridgement' key. Book IDs and chapter indices must match\
                        those specified for ablit data partition.",
                        type=str, required=False)

    args = parser.parse_args()

    with open(args.pred_abridged_file) as f:
        pred_abridgements = json.load(f)

    ablit_data = AblitDataset(dirpath=args.ablit_dir,
                              partition=args.partition)

    eval_data = evaluate(ablit_data=ablit_data,
                         pred_abridgements=pred_abridgements)

    score_output_file = (".".join(args.pred_abridged_file.split(".")[:-1])
                         + ".scores.json")

    eval_data = pandas.DataFrame(eval_data)
    scores = eval_data.drop(
        columns=['chapter_idx']).groupby('book_id').mean().round(
        decimals=3).to_dict(orient='index')
    scores['all'] = eval_data.drop(columns=['chapter_idx']).mean().round(
        decimals=3).to_dict()
    with open(score_output_file, "w") as f:
        json.dump(scores, f, indent=4)

    print("Saved evaluation results to", score_output_file)
    print("SUMMARY:")
    print(scores['all'])
