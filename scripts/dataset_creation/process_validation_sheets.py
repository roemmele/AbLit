import argparse
import pandas
import re
import json
import os


def validate_chapter_alignment(orig_data,
                               abridged_data,
                               user_aligned_seg_nums):

    validated_alignment = {
        'original_paragraph_nums': [],
        'original_segment_nums': [],
        'original_segments': [],
        'abridged_paragraph_nums': [],
        'abridged_segment_nums': [],
        'abridged_segments': []
    }

    for orig_seg_nums, abridged_seg_nums in user_aligned_seg_nums:
        orig_seg_idxs = [seg_num - 1 for seg_num in orig_seg_nums]
        # try:
        orig_par_nums = [orig_data['paragraph_nums'][seg_idx]
                         for seg_idx in orig_seg_idxs]
        # except:
        #     import pdb
        #     pdb.set_trace()
        orig_segs = [orig_data['segments'][seg_idx]
                     for seg_idx in orig_seg_idxs]

        validated_alignment['original_paragraph_nums'].append(orig_par_nums)
        validated_alignment['original_segment_nums'].append(orig_seg_nums)
        validated_alignment['original_segments'].append(orig_segs)

        abridged_seg_idxs = [seg_num - 1 for seg_num in abridged_seg_nums]
        abridged_par_nums = [abridged_data['paragraph_nums'][seg_idx]
                             for seg_idx in abridged_seg_idxs]
        abridged_segs = [abridged_data['segments'][seg_idx]
                         for seg_idx in abridged_seg_idxs]

        validated_alignment['abridged_paragraph_nums'].append(
            abridged_par_nums)
        validated_alignment['abridged_segment_nums'].append(
            abridged_seg_nums)
        validated_alignment['abridged_segments'].append(abridged_segs)

    check_completeness(validated_alignment, orig_data, abridged_data)

    return validated_alignment


def check_completeness(alignment, orig_data, abridged_data):

    def par_nums_are_complete(aligned_nums, max_num):

        unique_aligned_nums = []
        for nums in aligned_nums:
            for num in nums:
                if (not unique_aligned_nums
                        or unique_aligned_nums[-1] != num):
                    unique_aligned_nums.append(num)

        # import pdb
        # pdb.set_trace()
        return unique_aligned_nums == list(range(1, max_num + 1))

    orig_pars_are_complete = par_nums_are_complete(alignment['original_paragraph_nums'],
                                                   max_num=orig_data['paragraph_nums'][-1])
    assert orig_pars_are_complete,\
        "Error: Incomplete original paragraph nums: {}".format(
            alignment['original_paragraph_nums'])

    abridged_pars_are_complete = par_nums_are_complete(alignment['abridged_paragraph_nums'],
                                                       max_num=abridged_data['paragraph_nums'][-1])
    assert abridged_pars_are_complete,\
        "Error: Incomplete abridged paragraph indices: {}".format(
            alignment['abridged_paragraph_nums'])

    def seg_nums_are_complete(aligned_nums, max_num):

        unique_aligned_nums = []
        for nums in aligned_nums:
            for num in nums:
                if not unique_aligned_nums or unique_aligned_nums[-1] != num:
                    unique_aligned_nums.append(num)

        return unique_aligned_nums == list(range(1, max_num + 1))

    if not all(len(group) for group in alignment['original_segment_nums']):
        print("Error: unaligned abridged segment")
        import pdb
        pdb.set_trace()

    orig_segs_are_complete = seg_nums_are_complete(alignment['original_segment_nums'],
                                                   len(orig_data['segments']))

    if not orig_segs_are_complete:
        print("Error: Incomplete original segment numbers")
        print(alignment['original_segment_nums'])
        import pdb
        pdb.set_trace()

    # assert orig_segs_are_complete,\
    #     "Error: Incomplete original segment numbers: {}".format(
    #         alignment['original_segment_nums'])

    abridged_segs_are_complete = seg_nums_are_complete(alignment['abridged_segment_nums'],
                                                       len(abridged_data['segments']))
    # assert abridged_segs_are_complete,\
    #     "Error: Incomplete abridged segment numbers: {}".format(
    #         alignment['abridged_segment_nums'])
    if not abridged_segs_are_complete:
        print("Error: Incomplete abridged segment numbers")
        print(alignment['abridged_segment_nums'])
        import pdb
        pdb.set_trace()


def parse_user_alignment(orig_items, abridged_items):

    aligned_seg_nums = []

    for i, (orig_item, abridged_item) in enumerate(zip(orig_items, abridged_items)):
        if pandas.isnull(orig_item) or not orig_item.strip():
            orig_seg_nums = []
        else:
            orig_item = re.findall(r'(\[[0-9]+\])(.*)', orig_item)
            if not orig_item:
                print(
                    "WARNING: can't parse original segment info in row {}:{}".format(i + 1, original_item))
                orig_seg_nums = []
            else:
                orig_seg_nums = [int(seg_num.strip()[1:-1])
                                 for seg_num, _ in orig_item]

        if pandas.isnull(abridged_item) or not abridged_item.strip():
            abridged_seg_nums = []
        else:
            abridged_item = re.findall(r'(\[[0-9]+\])(.*)', abridged_item)
            if not abridged_item:
                print(
                    "WARNING: can't parse abridged segment info in row {}:{}".format(i + 1, abridged_item))
                abridged_seg_nums = []
            else:
                abridged_seg_nums = [int(seg_num.strip()[1:-1])
                                     for seg_num, _ in abridged_item]

        if not (len(orig_seg_nums) + len(abridged_seg_nums)):
            #print("WARNING: empty row:", i)
            continue

        aligned_seg_nums.append((orig_seg_nums, abridged_seg_nums))

    return aligned_seg_nums


def restructure_data(data):
    # import pdb
    # pdb.set_trace()
    restruct_data = []
    for i in range(len(data['original_paragraph_nums'])):
        item = {key: data[key][i] for key in data}
        restruct_data.append(item)
    assert all(len(vals) == len(restruct_data)
               for vals in data.values())
    return restruct_data


def process(sheet_dir,
            segment_data_dir,
            meta_data_file,
            out_dir):

    fileprefix_template = "{}.{}.valid.sheet"

    with open(meta_data_file) as f:
        meta_data = json.load(f)

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for book_id, book_info in meta_data.items():

        for chapter_idx in book_info['chapter_idxs']:

            chapter_filenames = [filename for filename in os.listdir(sheet_dir)
                                 if filename.startswith(fileprefix_template.format(book_id, chapter_idx))]

            for filename in chapter_filenames:

                print("Computing alignment for {} chapter idx {} in file {}".format(
                    book_id, chapter_idx, filename))

                if len(filename.split(".")) == 6:
                    validator_id = filename.split(".")[-2]
                    validator_out_dir = os.path.join(out_dir, validator_id)
                else:
                    validator_out_dir = os.path.join(out_dir)

                if not os.path.exists(validator_out_dir):
                    os.mkdir(validator_out_dir)

                validator_book_out_dir = os.path.join(
                    validator_out_dir, book_id)

                if not os.path.exists(validator_book_out_dir):
                    os.mkdir(validator_book_out_dir)

                user_output = pandas.read_excel(os.path.join(sheet_dir, filename),
                                                engine='openpyxl')

                user_output = user_output.rename(columns={"Original_Sentences": "Original_Segments",
                                                          "Abridged_Sentences": "Abridged_Segments"},
                                                 errors='ignore')

                user_aligned_seg_nums = parse_user_alignment(user_output['Original_Segments'].values,
                                                             user_output['Abridged_Segments'].values)

                with open(os.path.join(os.path.join(segment_data_dir,
                                                    'original',
                                                    book_id,
                                                    "{}.json".format(chapter_idx)))) as f:
                    orig_data = json.load(f)
                    if 'paragraph_idxs' in orig_data:
                        orig_data['paragraph_nums'] = [idx + 1 for idx
                                                       in orig_data['paragraph_idxs']]
                        orig_data.pop('paragraph_idxs')
                    if 'sentences' in orig_data:
                        orig_data['segments'] = orig_data['sentences']
                        orig_data.pop('sentences')

                with open(os.path.join(os.path.join(segment_data_dir,
                                                    'abridged',
                                                    book_id,
                                                    "{}.json".format(chapter_idx))), 'rb') as f:
                    abridged_data = json.load(f)
                    if 'paragraph_idxs' in abridged_data:
                        abridged_data['paragraph_nums'] = [idx + 1 for idx
                                                           in abridged_data['paragraph_idxs']]
                        abridged_data.pop('paragraph_idxs')
                    if 'sentences' in abridged_data:
                        abridged_data['segments'] = abridged_data['sentences']
                        abridged_data.pop('sentences')

                aligned_data = validate_chapter_alignment(orig_data,
                                                          abridged_data,
                                                          user_aligned_seg_nums)

                aligned_data.pop('original_segment_nums')
                aligned_data.pop('abridged_segment_nums')
                aligned_data = restructure_data(aligned_data)

                with open(os.path.join(validator_book_out_dir,
                                       '{}.json'.format(chapter_idx)), 'w') as f:
                    json.dump(aligned_data, f, indent=4)
                    print("Saved validated aligned pairs to {}".format(os.path.join(validator_book_out_dir,
                                                                                    '{}.json'.format(chapter_idx))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process output of user-validated alignments in order to finalize gold-standard alignment dataset.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--sheet_dir", "-sheet_dir",
                        help="Directory path containing human-aligned spreadsheet data\
                        (.xlsx files with prefix [BOOK_NAME].[CHAPTER_IDX])",
                        type=str, required=True)
    parser.add_argument("--segment_data_dir", "-segment_data_dir",
                        help="Directory path containing sentence-segmented data\
                        (folders for each book, .json files for each chapter containing paragraph numbers and segments)",
                        type=str, required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file",
                        help="Path to .json file with book/chapter index.",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir",
                        help="Directory where validated alignments will be saved.",
                        type=str, required=True)

    args = parser.parse_args()

    process(args.sheet_dir,
            args.segment_data_dir,
            args.meta_data_file,
            args.out_dir)
