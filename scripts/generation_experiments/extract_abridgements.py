import argparse
import json
import re
import numpy
from transformers import RobertaTokenizer


def extract_abridgement(tokenizer,
                        tokens,
                        labels,
                        extract_mode,
                        segment_extract_threshold=None):

    abridgement = ""

    for passage_toks, passage_labels in zip(tokens, labels):

        if len(passage_toks) > len(passage_labels):
            # import pdb
            # pdb.set_trace()
            restored_labels = [0] * (len(passage_toks) - len(passage_labels))
            passage_labels.extend(restored_labels)

        if extract_mode == 'tokens':
            psg_abridgement = extract_passage_abridgement_from_token_labels(
                tokenizer=tokenizer,
                passage_toks=passage_toks,
                passage_labels=passage_labels)

        else:
            assert segment_extract_threshold is not None
            psg_abridgement = extract_passage_abridgement_from_segment_labels(
                tokenizer=tokenizer,
                passage_toks=passage_toks,
                passage_labels=passage_labels,
                threshold=segment_extract_threshold)

        abridgement += " " + psg_abridgement

    abridgement = abridgement.strip()

    return abridgement


def extract_passage_abridgement_from_token_labels(tokenizer,
                                                  passage_toks,
                                                  passage_labels):

    # tok_grps = []
    # label_grps = []
    # for tok_i, (tok, label) in enumerate(zip(passage_toks, passage_labels)):
    #     if tok_i == 0:
    #         label_grp = [label]
    #         tok_grp = [tok]
    #         continue
    #     if tok[0] == "Ġ":
    #         label_grps.append(label_grp)
    #         tok_grps.append(tok_grp)
    #         label_grp = []
    #         tok_grp = []
    #     label_grp.append(label)
    #     tok_grp.append(tok)
    #     if tok_i == len(passage_toks) - 1:
    #         label_grps.append(label_grp)
    #         tok_grps.append(tok_grp)

    # tok_count = sum([len(tok_grp) for tok_grp in tok_grps])
    # assert tok_count == len(passage_toks)
    # label_count = sum([len(label_grp)
    #                    for label_grp in label_grps])
    # assert label_count == len(passage_labels)

    # abrg_toks = []
    # for tok_grp, label_grp in zip(tok_grps, label_grps):
    #     max_label = numpy.min(label_grp)
    #     if max_label == 0:
    #         for tok in tok_grp:
    #             abrg_toks.append(tok)

    # import pdb
    # pdb.set_trace()

    abrg_toks = [tok for tok, label in zip(passage_toks, passage_labels)
                 if label == 0]

    abridgement = tokenizer.convert_tokens_to_string(abrg_toks)
    return abridgement


def extract_passage_abridgement_from_segment_labels(tokenizer,
                                                    passage_toks,
                                                    passage_labels,
                                                    threshold=0.5,
                                                    eos_punct=[".", "?", "!", ":", ";"]):

    (segments,
     segment_labels) = split_passage_data_by_segment(tokenizer,
                                                     passage_toks,
                                                     passage_labels,
                                                     eos_punct)

    # import pdb
    # pdb.set_trace()
    abridgement = ""
    for seg, seg_labels in zip(segments, segment_labels):
        inverted_labels = ~(numpy.array(seg_labels).astype(bool))
        # try:
        if sum(inverted_labels) / len(seg_labels) >= threshold:
            if ((abridgement and abridgement[-1] not in (" ", "\n"))
                    and seg[0] not in (" ", "\n")):
                seg = " " + seg
            abridgement += seg
    return abridgement


def split_passage_data_by_segment(tokenizer,
                                  passage_toks,
                                  passage_labels,
                                  eos_punct=[".", "?", "!", ":", ";"]):
    eos_punct = set(eos_punct)

    seg_toks = []
    seg_labels = []
    prev_seg_boundary = 0
    for tok_i, tok in enumerate(passage_toks):
        if tok in eos_punct or tok_i == len(passage_toks) - 1:
            seg_toks.append(passage_toks[prev_seg_boundary:tok_i + 1])
            seg_labels.append(passage_labels[prev_seg_boundary:tok_i + 1])
            prev_seg_boundary = tok_i + 1

    assert all(len(lbls) for lbls in seg_labels)
    assert len(seg_toks) == len(seg_labels)
    segs = [tokenizer.convert_tokens_to_string(toks)
            for toks in seg_toks]
    assert "".join(segs) == tokenizer.convert_tokens_to_string(passage_toks)
    assert [label for labels in seg_labels for label in labels] == passage_labels
    return segs, seg_labels


def extract(tokenizer,
            pred_labels_data,
            extract_mode,
            segment_extract_threshold=None):

    outputs = []

    for book_id, book_data in pred_labels_data.items():

        for chapter_idx, chapter_data in book_data.items():

            abridgement = extract_abridgement(
                tokenizer=tokenizer,
                tokens=chapter_data['tokens'],
                labels=chapter_data['labels'],
                extract_mode=extract_mode,
                segment_extract_threshold=segment_extract_threshold)

            output = {'book_id': book_id,
                      'chapter_idx': chapter_idx,
                      'predicted_abridgement': abridgement}
            outputs.append(output)

    return outputs


def reformat_gold_labels_data(gold_labels_data):
    reformatted_gold_labels_data = {}

    for book_id, chapter_idx, toks, labels in zip(gold_labels_data['book_ids'],
                                                  gold_labels_data['chapter_idxs'],
                                                  gold_labels_data['tokens'],
                                                  gold_labels_data['labels']):
        if book_id not in reformatted_gold_labels_data:
            reformatted_gold_labels_data[book_id] = {}

        if chapter_idx not in reformatted_gold_labels_data[book_id]:
            reformatted_gold_labels_data[book_id][chapter_idx] = {"tokens": [],
                                                                  "labels": []}

        reformatted_gold_labels_data[book_id][chapter_idx]["tokens"].append(
            toks)
        reformatted_gold_labels_data[book_id][chapter_idx]["labels"].append(
            labels)

    return reformatted_gold_labels_data


def reformat_pred_labels_data(pred_labels_data, book_ids, chapter_idxs):

    reformatted_pred_labels_data = {}

    for item_idx, (book_id, chapter_idx) in enumerate(zip(book_ids, chapter_idxs)):
        if book_id not in reformatted_pred_labels_data:
            reformatted_pred_labels_data[book_id] = {}

        if chapter_idx not in reformatted_pred_labels_data[book_id]:
            reformatted_pred_labels_data[book_id][chapter_idx] = {"tokens": [],
                                                                  "labels": []}

        reformatted_pred_labels_data[book_id][chapter_idx]["tokens"].append(
            pred_labels_data[item_idx]["tokens"])
        reformatted_pred_labels_data[book_id][chapter_idx]["labels"].append(
            pred_labels_data[item_idx]["predicted_labels"])

    return reformatted_pred_labels_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--pred_labels_file", "-pred_labels_file",
                        help="File containing input token sequences and predicted labels for each token.\
                        Each line is a json object. If not provided, abridgements will be extracted from the gold data file.",
                        type=str,
                        required=False)

    parser.add_argument("--gold_labels_file", "-gold_labels_file",
                        help="JSON file containing gold token/label data corresponding to output in -pred_labels_file.\
                        If no -pred_labels_file is provided, it is assumed this file should be used to extract the abridgements\
                        (used as an upper bound for comparison to abridgements extracted via model predictions).",
                        type=str,
                        required=True)

    parser.add_argument("--tokenizer", "-tokenizer",
                        help="Path or name of HuggingFace tokenizer used to produce the tokens in -pred_labels_file",
                        type=str,
                        required=False,
                        default='roberta-base')

    parser.add_argument("--extract_mode", "-extract_mode",
                        help="Method for extracting an abridgement from the original input text.\
                        If 'tokens', any token with a prediction label indicating it should be preserved (here, 0)\
                        will be included in the extractive abridgement. If 'segments', input tokens are grouped into sentences (segments),\
                        and sentences that have a minimum proportion of their tokens predicted as 'preserved'\
                        will be included in the abridgement. This minimum proportion is specified by the parameter\
                        -segment_extract_threshold. Segment boundaries are dictated simply by specified punctuation (.?!;:).",
                        type=str,
                        required=True,
                        choices=['tokens', 'segments'])
    parser.add_argument("--segment_extract_threshold", "-segment_extract_threshold",
                        help="If extract_mode = 'segment', specify the minimum proportion of tokens labeled as\
                        'preserved' in a given segment in order for the entire segment to be included in the extractive abridgement.",
                        type=float,
                        required=False)

    parser.add_argument("--output_file", "-output_file",
                        help="Filepath (.json) where predicted abridgements will be saved.\
                        The json object is a list where each item corresponds to an abridgement for a single chapter,\
                        and has a 'book_id', 'chapter_idx', and 'predicted_abridgement' key.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)

    # if args.gold_labels_file:
    with open(args.gold_labels_file) as f:
        gold_labels_data = json.load(f)
        book_ids = gold_labels_data["book_ids"]
        chapter_idxs = gold_labels_data["chapter_idxs"]
        gold_labels_data = reformat_gold_labels_data(gold_labels_data)

    if args.pred_labels_file:
        pred_labels_data = []
        with open(args.pred_labels_file) as f:
            for line in f:
                item = json.loads(line)
                pred_labels_data.append(item)
        pred_labels_data = reformat_pred_labels_data(pred_labels_data,
                                                     book_ids,
                                                     chapter_idxs)
    else:
        pred_labels_data = gold_labels_data

    pred_abridgements = extract(tokenizer=tokenizer,
                                pred_labels_data=pred_labels_data,
                                extract_mode=args.extract_mode,
                                segment_extract_threshold=args.segment_extract_threshold)

    with open(args.output_file, 'w') as f:
        json.dump(pred_abridgements, f, indent=4)

    print("Saved extractive abridgements to", args.output_file)
