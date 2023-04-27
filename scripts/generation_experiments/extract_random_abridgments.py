import argparse
import json
import numpy
from nltk.tokenize import sent_tokenize
from abridgement_dataset import AbridgementDataset

numpy.random.seed(0)


def extract_abridgement(text, extract_mode, select_percent):
    # import pdb
    # pdb.set_trace()

    if extract_mode == 'segments':
        segments = sent_tokenize(text)
        n_selected_segments = int(len(segments) * select_percent)
        selected_seg_idxs = numpy.sort(
            numpy.random.permutation(len(segments))[:n_selected_segments])

        abridgement = ""
        for seg_idx in selected_seg_idxs:
            seg = segments[seg_idx]
            if ((abridgement and abridgement[-1] not in (" ", "\n"))
                    and seg[0] not in (" ", "\n")):
                seg = " " + seg
            abridgement += seg

    elif extract_mode == "tokens":
        # import pdb
        # pdb.set_trace()
        tokens = text.split(" ")
        n_selected_tokens = int(len(tokens) * select_percent)
        selected_tok_idxs = numpy.sort(
            numpy.random.permutation(len(tokens))[:n_selected_tokens])
        abridgement = " ".join([tokens[tok_idx]
                                for tok_idx in selected_tok_idxs])

    return abridgement


def extract(ablit_data, extract_mode, select_percent):

    abrg_items = []

    for book in ablit_data.books:
        for chapter in book.chapters:
            abridgement = extract_abridgement(text=chapter.original,
                                              extract_mode=extract_mode,
                                              select_percent=select_percent)
            item = {'book_id': book.book_id,
                    'chapter_idx': chapter.chapter_idx,
                    'predicted_abridgement': abridgement}
            abrg_items.append(item)

    return abrg_items


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Randomly pick sentences in original text for inclusion in abridgement,\
        as a random baseline for comparison with other abridgement models")

    parser.add_argument("--ablit_dir", "-ablit_dir",
                        help="Directory path to AbLit dataset.",
                        type=str,
                        required=True)
    parser.add_argument("--partition", "-partition",
                        help="Partition of AbLit dataset (train, dev, or test) associated with predicted abridgements\
                        in -abridgements_file.",
                        type=str,
                        required=False,
                        default='test')
    parser.add_argument("--extract_mode", "-extract_mode",
                        help="Method for extracting an abridgement from the original input text.\
                        If 'tokens', randomly select tokens for inclusion in the abridgement.\
                        If 'segments', randomly select full sentences for inclusion in the abridgment.",
                        type=str,
                        required=True,
                        choices=['tokens', 'segments'])
    parser.add_argument("--select_percent", "-select_percent",
                        help="Percentage of original tokens or segments (depending on -extract_mode)\
                        to randomly select for inclusion in abridgement.",
                        type=float,
                        required=True)
    parser.add_argument("--output_file", "-output_file",
                        help="Filepath (.json) where predicted abridgements will be saved.\
                        The json object is a list where each item corresponds to an abridgement for a single chapter,\
                        and has a 'book_id', 'chapter_idx', and 'predicted_abridgement' key.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    ablit_data = AbridgementDataset(dirpath=args.ablit_dir,
                                    partition=args.partition)

    abridged_items = extract(ablit_data=ablit_data,
                             extract_mode=args.extract_mode,
                             select_percent=args.select_percent)

    with open(args.output_file, 'w') as f:
        json.dump(abridged_items, f, indent=4)

    print("Saved random extractive abridgements to", args.output_file)
