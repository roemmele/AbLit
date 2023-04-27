import argparse
import json
import os
import spacy
import random
from transformers import RobertaTokenizer

random.seed(0)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")


def partition_train_dev_test(chapter_idxs, n_test_chapters=5):
    partition = {'train': None,
                 'dev': None,
                 'test': None}
    chapter_idxs.sort()
    partition['dev'] = [chapter_idxs[0]]
    shuffled_idxs = random.sample(chapter_idxs[1:],
                                  k=len(chapter_idxs[1:]))
    partition['test'] = shuffled_idxs[:n_test_chapters]
    partition['train'] = shuffled_idxs[n_test_chapters:]

    return partition


def tokenize(psg):
    start_chars = []
    end_chars = []
    toks = tokenizer.tokenize(psg.lower())
    for i, tok in enumerate(toks):
        if i == 0:
            start_chars.append(0)
            end_chars.append(len(tok))
        elif tok[0] == 'Ġ':
            toks[i] = tok[1:]
            start_chars.append(end_chars[-1] + 1)
            end_chars.append(end_chars[-1] + len(tok))
        else:
            start_chars.append(end_chars[-1])
            end_chars.append(end_chars[-1] + len(tok))

    return toks, start_chars, end_chars


def find_overlaps(orig_segs, abrg_segs):

    orig_psg = "".join(orig_segs)
    abrg_psg = "".join(abrg_segs)

    (orig_toks,
     orig_start_chars,
     orig_end_chars) = tokenize(orig_psg)

    if abrg_psg:
        (abrg_toks,
         abrg_start_chars,
         abrg_end_chars) = tokenize(abrg_psg)
    else:
        abrg_toks = []
        abrg_start_chars = []
        abrg_end_chars = []

    orig_overlap_chars = []
    abrg_overlap_chars = []

    n_abrg_toks = len(abrg_toks)
    abrg_tok_idxs = range(0, n_abrg_toks)

    covered_abrg_chars = set()
    covered_orig_chars = set()

    for tok_window_size in range(n_abrg_toks, 0, -1):

        for abrg_tok_i in abrg_tok_idxs:

            if len(abrg_toks[abrg_tok_i:]) < tok_window_size:
                break

            abrg_subpsg_start_char = abrg_start_chars[abrg_tok_i]
            abrg_subpsg_end_char = abrg_end_chars[abrg_tok_i +
                                                  tok_window_size - 1]

            if not abrg_psg[abrg_subpsg_start_char: abrg_subpsg_end_char].strip():
                continue

            abrg_subpsg_toks = abrg_toks[abrg_tok_i:abrg_tok_i +
                                         tok_window_size]

            if not check_span_is_valid(abrg_segs,
                                       abrg_subpsg_start_char,
                                       abrg_subpsg_end_char):
                continue

            if set(range(abrg_subpsg_start_char,
                         abrg_subpsg_end_char)).intersection(covered_abrg_chars):
                # print("abridged window already covered")
                continue

            matched_orig_chars = check_overlap(orig_toks,
                                               orig_start_chars,
                                               orig_end_chars,
                                               covered_orig_chars,
                                               abrg_subpsg_toks)

            if matched_orig_chars != None:
                matched_orig_start_char, matched_orig_end_char = matched_orig_chars
                orig_overlap_chars.append((matched_orig_start_char,
                                           matched_orig_end_char,))
                covered_orig_chars.update(list(range(matched_orig_start_char,
                                                     matched_orig_end_char)))

                abrg_overlap_chars.append((abrg_subpsg_start_char,
                                           abrg_subpsg_end_char,))
                covered_abrg_chars.update(list(range(abrg_subpsg_start_char,
                                                     abrg_subpsg_end_char)))

        if set(range(abrg_end_chars[-1])) == covered_abrg_chars:
            break

    if orig_overlap_chars:
        sort_order, abrg_overlap_chars = zip(*sorted(enumerate(abrg_overlap_chars),
                                                     key=lambda x: x[1][0]))
        orig_overlap_chars = [orig_overlap_chars[i] for i in sort_order]

    return orig_overlap_chars, abrg_overlap_chars


def check_overlap(orig_toks,
                  orig_start_chars,
                  orig_end_chars,
                  covered_orig_chars,
                  abrg_span_toks):

    for start_idx in range(len(orig_toks)):
        end_idx = start_idx + len(abrg_span_toks)
        if end_idx > len(orig_toks):
            break
        if orig_toks[start_idx:end_idx] == abrg_span_toks:
            matched_orig_start = orig_start_chars[start_idx]
            matched_orig_end = orig_end_chars[end_idx - 1]
            if set(range(matched_orig_start,
                         matched_orig_end)).intersection(covered_orig_chars):
                continue
            return (matched_orig_start, matched_orig_end)

    return None


def check_span_is_valid(segs, start_char, end_char):
    seg_end_chars = []
    for seg in segs:
        if not seg_end_chars:
            seg_end_chars.append(len(seg))
        else:
            seg_end_chars.append(seg_end_chars[-1] + len(seg))

    for seg_end_char in seg_end_chars:
        if start_char >= seg_end_char:
            continue
        if start_char <= seg_end_char and end_char <= seg_end_char:
            return True
        if end_char > seg_end_char:
            return False
    return True


def verify(text,
           paragraph_chars,
           segment_chars,
           row_chars,
           verification_text):

    for version in ('orig', 'abrg'):
        assert text[version] == verification_text[version]

        paragraphs = "".join([text[version][start_char:end_char]
                              for start_char, end_char in paragraph_chars[version]])
        assert paragraphs == verification_text[version]

        segments = "".join([text[version][start_char:end_char]
                            for start_char, end_char in segment_chars[version]])
        assert segments == verification_text[version]

        row_sequences = "".join([text[version][start_char:end_char]
                                 for start_char, end_char in row_chars[version]])
        assert row_sequences == verification_text[version]

    return True


def organize_data(text,
                  par_chars,
                  seg_chars,
                  overlap_chars,
                  row_chars):

    orig_data = {'text': text['orig'],
                 'paragraph_chars': par_chars['orig'],
                 'segment_chars': seg_chars['orig'],
                 'overlap_chars': overlap_chars['orig'],
                 'row_chars': row_chars['orig']}

    abrg_data = {'text': text['abrg'],
                 'paragraph_chars': par_chars['abrg'],
                 'segment_chars': seg_chars['abrg'],
                 'overlap_chars': overlap_chars['abrg'],
                 'row_chars': row_chars['abrg']}

    data = {'original': orig_data,
            'abridged': abrg_data}

    return data


def restructure_chapter_data(data):
    if type(data) == dict:
        return data

    restruct_data = {}

    for item in data:
        for key in item:
            if key not in restruct_data:
                restruct_data[key] = []
            restruct_data[key].append(item[key])

    assert all(len(vals) == len(data) for vals in restruct_data.values())

    return restruct_data


def process(in_dir,
            out_dir,
            meta_data_file,
            text_verify_dir=None,
            assign_partition=True,
            test_set_chapters_per_book=5):

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    new_meta_data_file = os.path.join(out_dir, "meta_data.json")
    assert not os.path.exists(new_meta_data_file),\
        ("File {} already exists. Move it to avoid overwrite, \
         or delete it to confirm overwrite is ok.".format(new_meta_data_file))

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for book_id, book_info in meta_data.items():

        print("Processing data for book {}...".format(book_id))

        book_out_dir = os.path.join(out_dir, book_id)
        if not os.path.exists(book_out_dir):
            os.mkdir(book_out_dir)

        for chapter_idx in book_info['chapter_idxs']:

            print("Processing chapter {}...".format(chapter_idx))

            with open(os.path.join(in_dir,
                                   book_id,
                                   "{}.json".format(chapter_idx))) as f:
                chapter_data = restructure_chapter_data(json.load(f))

            (text,
             par_chars,
             seg_chars,
             overlap_chars,
             row_chars) = process_chapter(chapter_data['original_paragraph_nums'],
                                          chapter_data['original_segments'],
                                          chapter_data['abridged_paragraph_nums'],
                                          chapter_data['abridged_segments'])

            if text_verify_dir:
                # The raw chapter texts have some space at the beginning and end of certain paragraphs.
                # In the processed data this space was removed.
                # So strip this space in the raw text before verification.
                with open(os.path.join(text_verify_dir, 'original',
                                       book_id, "{}.txt".format(chapter_idx))) as f:
                    orig_text = "\n".join([line.strip()
                                           for line in f if line.strip()])
                with open(os.path.join(text_verify_dir, 'abridged',
                                       book_id, "{}.txt".format(chapter_idx))) as f:
                    abrg_text = "\n".join([line.strip()
                                           for line in f if line.strip()])

                verification_text = {'orig': orig_text,
                                     'abrg': abrg_text}

                verify(text, par_chars, seg_chars,
                       row_chars, verification_text)

            final_chapter_data = organize_data(text,
                                               par_chars,
                                               seg_chars,
                                               overlap_chars,
                                               row_chars)

            with open(os.path.join(book_out_dir, "{}.json".format(chapter_idx)), 'w') as f:
                json.dump(final_chapter_data, f, indent=4)

        if assign_partition:
            partition = partition_train_dev_test(book_info['chapter_idxs'],
                                                 n_test_chapters=test_set_chapters_per_book)
            meta_data[book_id]['train_chapter_idxs'] = partition['train']
            meta_data[book_id]['dev_chapter_idxs'] = partition['dev']
            meta_data[book_id]['test_chapter_idxs'] = partition['test']

    with open(new_meta_data_file, 'w') as f:
        json.dump(meta_data, f, indent=4)
        print("Saved meta data file to", new_meta_data_file)


def restore_paragraph_breaks_in_segs(par_nums, segs):
    '''First go through the segments and add line breaks where there are paragraph breaks,
    to make the subsequent processing steps easier.'''

    n_seg_groups = len(segs)
    cur_par_num = None
    prev_nonempty_group_i = None
    prev_seg_i = None

    for group_i in range(n_seg_groups):
        for seg_i, (par_num, seg) in enumerate(zip(par_nums[group_i], segs[group_i])):
            if par_num != cur_par_num:
                if cur_par_num != None:
                    segs[prev_nonempty_group_i][prev_seg_i] += "\n"
                cur_par_num = par_num

            prev_nonempty_group_i = group_i
            prev_seg_i = seg_i

    return segs


def process_chapter(orig_par_nums, orig_segs, abrg_par_nums, abrg_segs):

    orig_segs = restore_paragraph_breaks_in_segs(orig_par_nums, orig_segs)
    abrg_segs = restore_paragraph_breaks_in_segs(abrg_par_nums, abrg_segs)

    row_chars = {'orig': [],
                 'abrg': []}

    par_nums = {'orig': orig_par_nums,
                'abrg': abrg_par_nums}

    segs = {'orig': orig_segs,
            'abrg': abrg_segs}

    cur_rel_overlap_chars = {'orig': [],
                             'abrg': []}

    par_chars = {'orig': [],
                 'abrg': []}
    seg_chars = {'orig': [],
                 'abrg': []}

    overlap_chars = {'orig': [],
                     'abrg': []}

    text = {'orig': "",
            'abrg': ""}

    cur_par_num = {'orig': None,
                   'abrg': None}

    n_seg_groups = len(segs['orig'])

    for group_i in range(n_seg_groups):

        (orig_rel_overlap_chars,
         abrg_rel_overlap_chars) = find_overlaps(segs['orig'][group_i],
                                                 segs['abrg'][group_i])

        cur_rel_overlap_chars['orig'] = orig_rel_overlap_chars
        cur_rel_overlap_chars['abrg'] = abrg_rel_overlap_chars

        for version in ('orig', 'abrg'):

            # Convert relative char indices to absolute based on paragraph
            overlap_chars[version].extend(
                [(start_char + len(text[version]),
                  end_char + len(text[version]),)
                 for start_char, end_char in cur_rel_overlap_chars[version]])

            n_segs = len(segs[version][group_i])

            if not n_segs:
                if not row_chars[version]:
                    row_chars[version].append((0, 0,))
                else:
                    row_chars[version].append((row_chars[version][-1][-1],
                                               row_chars[version][-1][-1],))

            for seg_i, (par_num, seg) in enumerate(zip(par_nums[version][group_i],
                                                       segs[version][group_i])):
                if par_num != cur_par_num[version]:
                    if cur_par_num[version] != None:
                        if not par_chars[version]:
                            par_chars[version].append((0,
                                                       len(text[version]),))
                        else:
                            par_chars[version].append((par_chars[version][-1][-1],
                                                       len(text[version]),))

                    cur_par_num[version] = par_num

                text[version] += seg
                if not seg_chars[version]:
                    seg_chars[version].append((0,
                                               len(text[version]),))
                else:
                    seg_chars[version].append((seg_chars[version][-1][-1],
                                               len(text[version]),))

                if seg_i == n_segs - 1:
                    if not row_chars[version]:
                        row_chars[version].append((0,
                                                   len(text[version]),))
                    else:
                        row_chars[version].append((row_chars[version][-1][-1],
                                                   len(text[version]),))

            if group_i == n_seg_groups - 1:
                if not par_chars[version]:
                    par_chars[version].append((0,
                                               len(text[version]),))
                elif len(text[version]) > par_chars[version][-1][-1]:
                    par_chars[version].append((par_chars[version][-1][-1],
                                               len(text[version]),))

    return (text, par_chars, seg_chars,
            overlap_chars, row_chars)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply final processing steps to validation data.\
                                     This involves converting the paragraph and segment info into character indices\
                                     used to extract them directly from the raw chapter text, \
                                     as well as finding overlapping spans between the original and abridged texts.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_dir", "-in_dir",
                        help="Directory path for validation data (.json files).",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir",
                        help="Directory path for where to save finalized (i.e. packaged) data.",
                        type=str, required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file",
                        help="Path to .json file with book/chapter index.",
                        type=str, required=True)
    parser.add_argument("--text_verify_dir", "-text_verify_dir",
                        help="For optional sanity check purposes, provide directory of raw texts associated with validation data.\
                        Will ensure that text in packed data matches the raw texts.",
                        type=str, required=False)
    parser.add_argument("--assign_partition", "-assign_partition",
                        help="Assign chapters to train, dev, and test sets.\
                        These chapter indices will be specified in the saved meta_data.json file.\
                        Note that dev set is already fixed to first chapter index (0) in each book.",
                        action='store_true', required=False)
    parser.add_argument("--test_set_chapters_per_book", "-test_set_chapters_per_book",
                        help="If partition=True, specify number of chapters per book to assign to test set.",
                        type=int, required=False, default=5)

    args = parser.parse_args()

    process(args.in_dir,
            args.out_dir,
            args.meta_data_file,
            args.text_verify_dir,
            args.assign_partition,
            args.test_set_chapters_per_book)
