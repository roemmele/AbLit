import argparse
import json
from transformers import RobertaTokenizer

# do "pip install ." in ../abridgement_dataset_pkg directory
from abridgement_dataset import AbridgementDataset


def get_tokens_and_labels(tokenizer, text, overlaps, char_offset):
    '''Main function that outputs tokens and labels for a single passage.
    Each binary label corresponds to a token in the original passage; 
    1 indicates that token has been modified (i.e. removed or replaced) from the corresponding abridgement,
    and 0 indicates the token is preserved in the abridgement'''

    # Tokenizer function should strip all trailing (but not leading) whitespace
    def tokenize(text):
        if text and text[-1] == ' ':
            return tokenizer.tokenize(text.rstrip())
        else:
            return tokenizer.tokenize(text)

    tokens = []
    labels = []
    overlaps = sorted([[overlap.start_char - char_offset,
                        overlap.end_char - char_offset]
                       for overlap in overlaps])
    if not overlaps:
        toks = tokenize(text)
        tokens.extend(toks)
        labels.extend([1] * len(toks))
        assert tokenizer.convert_tokens_to_string(
            tokens).rstrip() == text.rstrip()
        return tokens, labels

    for i, (cur_start, cur_end) in enumerate(overlaps):
        if i == 0 and cur_start > 0:
            # If this is the first overlap and it's not at the start of the text,
            # label the preceding tokens as 'modified' (1)
            toks = tokenize(text[:cur_start])
            tokens.extend(toks)
            labels.extend([1] * len(toks))
            prev_end = 0
        elif i > 0:
            # Label the tokens between the last overlap and this overlap as 'modified' (1)
            _, prev_end = overlaps[i - 1]
            if text[prev_end - 1] == ' ':
                toks = tokenize(' ' + text[prev_end:cur_start])
            else:
                toks = tokenize(text[prev_end:cur_start])
            tokens.extend(toks)
            labels.extend([1] * len(toks))

        # Label the tokens in this overlap as 'preserved' (0)
        if cur_start > 0 and text[cur_start - 1] == ' ':
            toks = tokenize(' ' + text[cur_start:cur_end])
        else:
            toks = tokenize(text[cur_start:cur_end])
        tokens.extend(toks)
        labels.extend([0] * len(toks))

        # Finally, if this is the final overlap and it does not cover
        # the rest of the text, label the tokens after this overlap as 'modified' (1)
        if i == len(overlaps) - 1 and cur_end < len(text):
            if text[cur_end - 1] == ' ':
                toks = tokenize(' ' + text[cur_end:])
            else:
                toks = tokenize(text[cur_end:])
            tokens.extend(toks)
            labels.extend([1] * len(toks))

    assert len(tokens) == len(labels)
    assert tokenizer.convert_tokens_to_string(tokens).rstrip() == text.rstrip()
    return tokens, labels


def make_prediction_dataset(tokenizer,
                            input_data,
                            passage_type,
                            chunk_merge_n_segments,
                            chunk_max_n_segments,
                            output_file):

    tokens = []
    labels = []
    book_ids = []
    chapter_idxs = []

    if passage_type == 'chunk':
        assert chunk_merge_n_segments or chunk_max_n_segments,\
            "If passage_type = 'chunk', you must specify either -chunk_merge_n_segments or -chunk_max_n_segments."
    else:
        assert not chunk_merge_n_segments and not chunk_max_n_segments,\
            "Must specify passage_type = 'chunk' if providing a value for -chunk_merge_n_segments or -chunk_max_n_segments"

    for book in input_data.books:
        print("Processing book: {}".format(book.book_id))

        for chapter in book.chapters:

            if passage_type == 'oracle_span':
                passages = chapter.oracle_spans
            elif passage_type == 'paragraph':
                passages = chapter.paragraphs
            elif passage_type == 'segment':
                passages = chapter.segments
            elif passage_type == 'chunk':
                assert chunk_merge_n_segments or chunk_max_n_segments,\
                    "If passage_type = 'chunk', you must specify either -chunk_merge_n_segments or -chunk_max_n_segments."
                passages = chapter.chunks(merge_n_segments=chunk_merge_n_segments,
                                          max_n_segments=chunk_max_n_segments)
            else:
                passages = [chapter]

            for passage in passages:
                toks, lbls = get_tokens_and_labels(tokenizer=tokenizer,
                                                   text=passage.original,
                                                   overlaps=passage.overlaps,
                                                   char_offset=passage.original_start_char)
                tokens.append(toks)
                labels.append(lbls)
                book_ids.append(book.book_id)
                chapter_idxs.append(chapter.chapter_idx)

    dataset = {'tokens': tokens,
               'labels': labels,
               'book_ids': book_ids,
               'chapter_idxs': chapter_idxs}

    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=4)

    print("Saved dataset to", output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Label tokens in the texts according to whether or not\
                                     they appear in their corresponding abridged version, in order to train\
                                     a sequence classification model to predict these labels.")

    parser.add_argument("--data_dir", "-data_dir",
                        help="Directory path to abridgement dataset.",
                        type=str,
                        required=True)
    parser.add_argument("--partition", "-partition",
                        help="Partition of abridgement dataset (train, dev, or test)\
                        for which to assign prediction labels.",
                        type=str,
                        required=True)

    parser.add_argument("--tokenizer", "-tokenizer",
                        help="Path or name of HuggingFace tokenizer to apply to data\
                        (should be consistent with whatever prediction model will be used)",
                        type=str,
                        required=False,
                        default='roberta-base')

    parser.add_argument("--passage_type", "-passage_type",
                        help="Specify how input texts should be split,\
                        such that each resulting split passage will form a token/label sequence\
                        given as input to the prediction model.\
                        Possible types are 'oracle_span', 'segment', 'paragraph', and 'chunk'.\
                        The type 'oracle_span' pertains to the raw result of the alignment task.\
                        In this task, groups of one or more segments in the original text were aligned\
                        to groups of one or more segments in the abridged text, defining the\
                        the narrowest alignment possible. The number of segments in each group is variable.\
                        An oracle span consists of the group of original segments\
                        contained in each of these raw alignments. For instance, an oracle span may consist of two segments in the original text,\
                        that align with one segment in the abridged text.\
                        The term 'oracle' is used because the size of these spans cannot\
                        be directly inferred for any arbitrary input text, in contrast to the other passage types\
                        where the passage size is determined through a function that can be applied to any text\
                        (e.g. a sentence-segmentation algorithm that splits the text into sentence-length passages).\
                        With oracle passages, no additional inference is needed to find its alignment\
                        with an abridged passage, unlike with the 'chunk', 'segment', and 'paragraph' passage types\
                        where heuristics are applied to isolate the exact boundaries\
                        of the abridged passage that aligns to each original passage.\
                        Thus experiments involving oracle alignments provide a useful reference point\
                        for model performance because the ideal size of the input passages covering each alignment is known in advance.\
                        If the passage type is 'chunk', parameter of number of segments per chunk (--chunk_merge_n_segments)\
                        will need to be specified.\
                        Chunks are formed by grouping the text into passages of --chunk_merge_n_segments.\
                        By default, paragraph boundaries will be respected so that a given paragraph\
                        will never be split into smaller passages; chunks merely join paragraphs.\
                        Only paragraphs where the total number of segments between the paragraphs is less\
                        than or equal to --chunk_merge_n_segments will be joined.\
                        It is always possible for chunks to be smaller than\
                        --chunk_merge_n_segments if a grouping of this size cannot be found.\
                        You can specify the parameter -chunk_split_paragraphs, to additionally split paragraphs such\
                        that no chunk will exceed --chunk_merge_n_segments.\
                        If passage_type is not specified, texts will not be divided into passages.",
                        type=str,
                        required=False,
                        choices=['oracle_span', 'segment', 'paragraph', 'chunk', None])
    parser.add_argument("--chunk_merge_n_segments", "-chunk_merge_n_segments",
                        help="If passage_type = 'chunk', minimum number of segments in each chunk.\
                        This parameter is ignored if passage type = 'segment' or 'paragraph', or is None.\
                        This will merge paragraphs where the total number\
                        of segments across the paragraphs is greater than or equal to this parameter.\
                        This parameter does not split paragraphs, it only merges them.\
                        If splitting inside of paragraphs is desired, you should instead\
                        specify the -chunk_max_n_segments parameter.",
                        type=int,
                        required=False)
    parser.add_argument("--chunk_max_n_segments", "-chunk_max_n_segments",
                        help="If passage_type = 'chunk', maximum number of segments in each chunk.\
                        This parameter is ignored if passage type = 'segment' or 'paragraph', or is None.\
                        This parameter will cause paragraphs with fewer than -chunk_max_n_segments to be merged,\
                        and additionally paragraphs will be split into passages of maximum size -chunk_max_n_segments.",
                        type=int,
                        required=False)

    parser.add_argument("--output_file", "-output_file",
                        help="Filepath (.json) where token/label sequence output will be saved.",
                        type=str,
                        required=True)

    args = parser.parse_args()

    input_data = AbridgementDataset(dirpath=args.data_dir,
                                    partition=args.partition)

    tokenizer = RobertaTokenizer.from_pretrained(args.tokenizer)

    make_prediction_dataset(tokenizer=tokenizer,
                            input_data=input_data,
                            passage_type=args.passage_type,
                            chunk_merge_n_segments=args.chunk_merge_n_segments,
                            chunk_max_n_segments=args.chunk_max_n_segments,
                            output_file=args.output_file)
