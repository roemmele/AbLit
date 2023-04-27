import argparse
import json
import numpy
import pandas
from sklearn.metrics import precision_recall_fscore_support

from transformers import RobertaTokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def get_token_ids(toks, lowercase=True):
    tok_counts = {}
    tok_ids = set()
    # for tok in word_tokenize(passage):
    for tok in toks:
        if lowercase:
            tok = tok.lower()
        if tok not in tok_counts:
            tok_counts[tok] = 0
        else:
            tok_counts[tok] += 1
        tok_ids.add("{}_{}".format(tok, tok_counts[tok]))
    return tok_ids


# def get_removal_scores(pred_toks,
#                        ref_toks):

#     pred_mod_toks = get_token_ids(pred_toks)
#     ref_mod_toks = get_token_ids(ref_toks)

#     correct_mod_toks = ref_mod_toks.intersection(pred_mod_toks)

#     if pred_mod_toks:
#         precision = len(correct_mod_toks) / len(pred_mod_toks)
#     else:
#         precision = 1.0

#     assert precision <= 1.0

#     if ref_mod_toks:
#         recall = len(correct_mod_toks) / len(ref_mod_toks)
#     else:
#         recall = 1.0

#     assert recall <= 1.0

#     if precision + recall:
#         f1 = 2.0 * ((precision * recall) / (precision + recall))
#     else:
#         f1 = 0.0

#     return {'precision': precision,
#             'recall': recall,
#             'f1': f1}


# def eval_text(pred_labels, gold_labels, tokens):

#     pred_toks = [tok for tok, label in zip(tokens, pred_labels)
#                  if label == 1]
#     gold_toks = [tok for tok, label in zip(tokens, gold_labels)
#                  if label == 1]

#     scores = get_removal_scores(pred_toks,
#                                 gold_toks)

#     return scores


def restore_truncated_labels(gold_labels,
                             predicted_labels):
    assert len(gold_labels) >= len(predicted_labels)
    if len(gold_labels) > len(predicted_labels):
        restored_labels = [0] * (len(gold_labels) - len(predicted_labels))
        predicted_labels.extend(restored_labels)
    return predicted_labels


def organize_labels_by_book_and_chapter(book_ids,
                                        chapter_idxs,
                                        predicted_labels,
                                        gold_labels):

    labels = {}

    for book_id, chapter_idx, pred_lbls, gold_lbls in zip(book_ids,
                                                          chapter_idxs,
                                                          predicted_labels,
                                                          gold_labels):
        if book_id not in labels:
            labels[book_id] = {}

        if chapter_idx not in labels[book_id]:
            labels[book_id][chapter_idx] = {'gold': [],
                                            'predicted': []}

        labels[book_id][chapter_idx]['gold'].extend(gold_lbls)
        labels[book_id][chapter_idx]['predicted'].extend(pred_lbls)

    return labels


# def organize_tokens_by_book_and_chapter(book_ids,
#                                         chapter_idxs,
#                                         tokens):

#     toks = {}

#     for book_id, chapter_idx, tks in zip(book_ids,
#                                          chapter_idxs,
#                                          tokens):
#         if book_id not in toks:
#             toks[book_id] = {}

#         if chapter_idx not in toks[book_id]:
#             toks[book_id][chapter_idx] = []

#         toks[book_id][chapter_idx].extend(tks)

#     return toks


def evaluate(book_ids,
             chapter_idxs,
             predicted_labels,
             gold_labels,
             tokens):

    labels_by_book_and_chapter = organize_labels_by_book_and_chapter(book_ids,
                                                                     chapter_idxs,
                                                                     predicted_labels,
                                                                     gold_labels)
    # tokens_by_book_and_chapter = organize_tokens_by_book_and_chapter(book_ids,
    #                                                                  chapter_idxs,
    #                                                                  tokens)

    scores = []

    for book_id in labels_by_book_and_chapter:
        for chapter_idx in labels_by_book_and_chapter[book_id]:
            labels = labels_by_book_and_chapter[book_id][chapter_idx]
            # toks = tokens_by_book_and_chapter[book_id][chapter_idx]

            precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels["gold"],
                                                                       y_pred=labels["predicted"],
                                                                       pos_label=1,
                                                                       average='binary')
            # import pdb
            # pdb.set_trace()
            # text_scores = eval_text(labels["predicted"],
            #                         labels["gold"],
            #                         toks)

            scores.append({'book_id': book_id,
                           'chapter_idx': chapter_idx,
                           'n_tokens': len(labels["gold"]),
                           'precision': precision.round(3),
                           'recall': recall.round(3),
                           'f1': f1.round(3)})
            # 'text_precision': round(text_scores["precision"], 3),
            # 'text_recall': round(text_scores["recall"], 3),
            # 'text_f1': round(text_scores["f1"], 3)})

    return scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate the accuracy of the model-predicted labels\
                                     for tokens in a text, where each label indicates whether or not that token appears in\
                                     the corresponding abridged version of the text.")

    parser.add_argument("--gold_labels_file", "-gold_labels_file",
                        help="JSON file containing tokens and gold prediction labels.\
                        This file is the output of make_prediction_dataset.py.\
                        The gold labels are already contained in -output_file but this file additionally\
                        contains the book ID and chapter index for each data instance.",
                        type=str,
                        required=True)
    parser.add_argument("--output_file", "-output_file",
                        help="JSONL file containing predicted labels for corresponding items in -gold_labels_file.\
                        Also happens to contain input tokens and gold labels, but not necessarily book/chapter info.\
                        Scores will be saved to a filepath with this same prefix.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    with open(args.gold_labels_file) as f:
        gold_data = json.load(f)

    with open(args.output_file) as f:
        output_data = {'predicted_labels': []}
        for i, line in enumerate(f):
            item = json.loads(line)
            predicted_labels = restore_truncated_labels(gold_labels=gold_data['labels'][i],
                                                        predicted_labels=item['predicted_labels'])
            output_data['predicted_labels'].append(predicted_labels)

     # Verify that data is consistent across gold and output files
    assert len(gold_data['labels']) == len(output_data['predicted_labels'])

    scores = evaluate(book_ids=gold_data['book_ids'],
                      chapter_idxs=gold_data['chapter_idxs'],
                      predicted_labels=output_data['predicted_labels'],
                      gold_labels=gold_data['labels'],
                      tokens=gold_data['tokens'])

    scores_fp = (".".join(args.output_file.split(".")[:-1])
                 + ".scores.json")

    scores = pandas.DataFrame(scores)
    scores_by_book = scores.drop(
        columns=['chapter_idx']).groupby('book_id').mean().round(
        decimals=3)  # .to_dict(orient='index')
    scores_by_book["n_tokens"] = scores.groupby(
        'book_id')["n_tokens"].sum() * 1.0
    scores_by_book = scores_by_book.to_dict(orient='index')
    scores_by_book['all'] = scores.drop(columns=['chapter_idx']).mean().round(
        decimals=3).to_dict()
    scores_by_book['all']["n_tokens"] = scores["n_tokens"].sum() * 1.0
    scores_by_book["all"]["mean_psg_tokens"] = numpy.mean(
        [len(lbls) for lbls in gold_data['labels']]).round(3)

    with open(scores_fp, "w") as f:
        json.dump(scores_by_book, f, indent=4)

    print("Total Scores:", scores_by_book["all"])
