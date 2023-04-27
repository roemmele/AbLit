import argparse
import os
import json
import string
import re
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')  # Ensure sentence-segmentation model is downloaded


def verify_segment_end_ok(segment):

    tokens = segment.strip().split(" ")

    # List of common English abbreviations and honorifics/titles ending in punctuation,
    abbrevs = ["a.d.", "a.m.", "anon.", "ave.", "b.c.", "blvd.", "br.", "c.", "ca.", "cf.", "cg.", "ct.",
               "def.", "dr.", "drs.", "e.", "e.g.", "et al.", "et seq.", "etc.", "fr.", "ft.", "hon.", "hr.",
               "ibid.", "id.", "inc.", "i.e.", "illus.", "jr.", "kg.", "km.", "lb.", "ln.",
               "m.", "messrs.", "min.", "ml.", "mmes.", "mph.", "mr.", "mrs.", "ms.", "mss.", "mt.",
               "n.", "n.b.", "n.e.", "n.w.", "oz.",
               "p.", "pg.", "p.m.", "pp.", "pr.", "prof.", "p.s.", "pseud.", "ps.", "pt.", "pub.", "q.v.",
               "rd.", "rev.", "rte.", "s.", "sc.", "s.e.", "sec.", "sen.", "sr.", "st.", "s.w.", "s.v.",
               "trans.", "viz.", "vol.", "v.", "vs.", "w."]  # took out "no." : too many false positives

    if tokens[-1].lower() in abbrevs:
        return False

        # Uncomment below to instead require user confirmation of segments with abbreviations,
        # instead of automatically rejecting
        '''
        if tokens[-1].lower() in ["mr.", "mrs.", "dr.", "rev."]:  # Very common errors, just reject
            return False
        is_ok = input(
            "Segment ends in abbreviation: {} | Okay? (y/n): ".format(segment))
        if is_ok.lower() == 'y':
            return True
        else:
            return False
        '''

    return True


def verify_segment_start_ok(segment):
    if all(char in string.punctuation for char in segment.strip()):
        print(
            "Segment consists only of punctuation, marked as not okay: {}".format(segment))
        return False
    return True


def split_into_segments(paragraph):

    segs = []

    paragraph = paragraph.strip()
    nltk_sents = sent_tokenize(paragraph)
    cursor_position = 0
    prev_segment_is_ok = True

    for sent in nltk_sents:

        if segs:
            prev_segment_end_ok = verify_segment_end_ok(segs[-1])

        seg_match = re.match(re.escape(sent) + r"[\s]*",
                             paragraph[cursor_position:])
        assert seg_match,\
            "Error in sentence segmentation for paragraph {}".format(paragraph)

        _, seg_end = seg_match.span()
        seg = paragraph[cursor_position:cursor_position + seg_end]

        cur_segment_start_ok = verify_segment_start_ok(seg)

        if not segs or (prev_segment_end_ok and cur_segment_start_ok):
            segs.append(seg)
        else:
            segs[-1] += seg

        cursor_position += seg_end

    assert cursor_position == len(paragraph),\
        "Error in sentence segmentation for paragraph {}".format(paragraph)

    assert "".join(segs) == paragraph,\
        "Error: some text lost from paragraph during segmentation"

    return segs


def process(in_dir, out_dir, meta_data_file):

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if not os.path.exists(os.path.join(out_dir, 'original')):
        os.mkdir(os.path.join(out_dir, 'original'))

    if not os.path.exists(os.path.join(out_dir, 'abridged')):
        os.mkdir(os.path.join(out_dir, 'abridged'))

    for book_i, (book, book_info) in enumerate(meta_data.items()):

        src_book_in_dir = os.path.join(in_dir, 'original', book)
        src_book_out_dir = os.path.join(out_dir, 'original', book)
        if not os.path.exists(src_book_out_dir):
            os.mkdir(src_book_out_dir)

        tgt_book_in_dir = os.path.join(in_dir, 'abridged', book)
        tgt_book_out_dir = os.path.join(out_dir, 'abridged', book)
        if not os.path.exists(tgt_book_out_dir):
            os.mkdir(tgt_book_out_dir)

        for chapter_i, chapter_title in zip(book_info['chapter_idxs'],
                                            book_info['chapter_titles']):

            print("\nProcessing book {}/{}: {} {}...".format(book_i + 1,
                                                             len(meta_data),
                                                             book,
                                                             chapter_title))

            src_chapter_in_file = os.path.join(src_book_in_dir,
                                               "{}.txt".format(chapter_i))
            src_par_idxs = []
            src_segs = []
            with open(src_chapter_in_file) as f:
                for par_i, paragraph in enumerate(f):
                    segs = split_into_segments(paragraph)
                    src_par_idxs.extend([par_i] * len(segs))
                    src_segs.extend(segs)
                print("Segmented {} sentences across {} paragraphs in source".format(len(src_segs),
                                                                                     par_i + 1))

            src_chapter_out_file = os.path.join(src_book_out_dir,
                                                "{}.json".format(chapter_i))
            with open(src_chapter_out_file, 'w') as f:
                json.dump({'paragraph_nums': [idx + 1 for idx in src_par_idxs],
                           'segments': src_segs},
                          f,
                          indent=4)
                print("Saved segment data for original chapters to {}".format(
                    src_chapter_out_file))

            tgt_chapter_in_file = os.path.join(tgt_book_in_dir,
                                               "{}.txt".format(chapter_i))
            tgt_par_idxs = []
            tgt_segs = []
            with open(tgt_chapter_in_file) as f:
                for par_i, paragraph in enumerate(f):
                    segs = split_into_segments(paragraph)
                    tgt_par_idxs.extend([par_i] * len(segs))
                    tgt_segs.extend(segs)
                print("Segmented {} sentences across {} paragraphs in target".format(len(tgt_segs),
                                                                                     par_i + 1))

            tgt_chapter_out_file = os.path.join(tgt_book_out_dir,
                                                "{}.json".format(chapter_i))
            with open(tgt_chapter_out_file, 'w') as f:
                json.dump({'paragraph_nums': [idx + 1 for idx in tgt_par_idxs],
                           'segments': tgt_segs},
                          f,
                          indent=4)
                print("Saved segment data for abridged chapters to {}".format(
                    tgt_chapter_out_file))

    print("\nSaved all segment data to:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform sentence segmentation on original and abridged chapters\
                                     that have already been processed by split_chapters.py",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_dir", "-in_dir", help="Directory path where raw chapter data is located,\
                        processed according to split_chapters.py: inside are 'original' and 'abridged' folders,\
                        within which are folders named by book, and then one chapter per file inside each book folder.",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir", help="Directory path where sentence-segmented chapter data\
                        will be saved (.json files)",
                        type=str, required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file", help="Path to .json file with book/chapter index.",
                        type=str, required=True)

    args = parser.parse_args()

    process(args.in_dir,
            args.out_dir,
            args.meta_data_file)
