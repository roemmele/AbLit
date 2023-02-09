import argparse
import sys
from ablit import AblitDataset


def check(original_text,
          abridged_text,
          original_version_psgs,
          abridged_version_psgs,
          paired_psgs):

    original_version_psgs = list(original_version_psgs)
    abridged_version_psgs = list(abridged_version_psgs)
    paired_psgs = list(paired_psgs)

    direct_orig_psgs = [psg.text for psg in original_version_psgs]
    direct_orig_text = "".join(direct_orig_psgs)
    assert direct_orig_text == original_text

    inferred_orig_psgs = [psg.original for psg in paired_psgs]
    inferred_orig_text = "".join(inferred_orig_psgs)
    assert inferred_orig_text == original_text

    assert direct_orig_text == inferred_orig_text

    direct_abrg_psgs = [psg.text for psg in abridged_version_psgs]
    direct_abrg_text = "".join(direct_abrg_psgs)
    assert direct_abrg_text == abridged_text

    inferred_abrg_psgs = [psg.abridged for psg in paired_psgs]
    inferred_abrg_text = "".join(inferred_abrg_psgs)
    assert inferred_abrg_text == abridged_text

    assert direct_abrg_text == inferred_abrg_text

    assert original_text == "".join([original_text[psg.start_char:psg.end_char]
                                     for psg in original_version_psgs])
    assert original_text == "".join([original_text[psg.original_start_char:psg.original_end_char]
                                     for psg in paired_psgs])

    assert abridged_text == "".join([abridged_text[psg.start_char:psg.end_char]
                                     for psg in abridged_version_psgs])


def test(dataset):
    for book in dataset.books:
        print("Checking data for book: {}...".format(book.book_title))
        for chapter in book.chapters:
            print("Chapter {}...".format(chapter.chapter_idx))

            print("checking sentences...")
            check(chapter.original,
                  chapter.abridged,
                  chapter.original_version.sentences,
                  chapter.abridged_version.sentences,
                  chapter.sentences)

            print("checking paragraphs...")
            check(chapter.original,
                  chapter.abridged,
                  chapter.original_version.paragraphs,
                  chapter.abridged_version.paragraphs,
                  chapter.paragraphs)

            print("checking chunks with merge_n_sentences=3...")
            check(chapter.original,
                  chapter.abridged,
                  chapter.original_version.chunks(merge_n_sentences=3),
                  chapter.abridged_version.chunks(merge_n_sentences=3),
                  chapter.chunks(merge_n_sentences=3))

            print("checking chunks with max_n_sentences=3...")
            check(chapter.original,
                  chapter.abridged,
                  chapter.original_version.chunks(max_n_sentences=3),
                  chapter.abridged_version.chunks(max_n_sentences=3),
                  chapter.chunks(max_n_sentences=3))

            print("checking row spans...")
            check(chapter.original,
                  chapter.abridged,
                  chapter.original_version.row_spans,
                  chapter.abridged_version.row_spans,
                  chapter.rows)

    print("OK!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", "-data_dir",
                        help="Directory path for dataset.",
                        type=str, required=True)

    args = parser.parse_args()

    dataset = AblitDataset(dirpath=args.data_dir)
    test(dataset)
