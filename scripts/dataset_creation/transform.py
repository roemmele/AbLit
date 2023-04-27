import argparse
import os
import json
import numpy
import pickle
import string
import torch
from transformers import AutoTokenizer, AutoModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HuggingFaceVectorizer():

    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    def __call__(self, segments, batch_size=128, max_segment_length=None):
        segment_vectors = []
        with torch.no_grad():
            for batch_idx in range(0, len(segments), batch_size):
                batch_data = self.tokenizer(segments[batch_idx:batch_idx + batch_size],
                                            padding=True,
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            truncation=True,
                                            max_length=max_segment_length).to(device)
                batch_segment_vectors = self.model(**batch_data)[0]
                batch_segment_vectors = (batch_segment_vectors *
                                         batch_data.attention_mask[:, :, None])
                segment_vectors.extend(
                    torch.sum(batch_segment_vectors, axis=1).cpu().numpy())
        segment_vectors = numpy.stack(segment_vectors, axis=0)
        return segment_vectors


def transform(in_dir,
              out_dir,
              meta_data_file,
              vectorizer_name,
              max_src_window=4,
              max_tgt_window=4):

    vectorizer = HuggingFaceVectorizer(model_name=vectorizer_name)

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

        for chapter_i in book_info['chapter_idxs']:
            print("\nProcessing book {}/{}: {} {}...".format(book_i + 1,
                                                             len(meta_data),
                                                             book,
                                                             book_info['chapter_titles'][chapter_i]))

            src_chapter_in_file = os.path.join(src_book_in_dir,
                                               "{}.json".format(chapter_i))

            with open(src_chapter_in_file) as f:
                data = json.load(f)

            if 'sentences' in data:
                src_segs = data['sentences']
            elif 'segments' in data:
                src_segs = data['segments']
            else:
                assert False, "No segment info found in source data"

            print("Encoding source vectors for {} segments...".format(len(src_segs)))
            src_vectors = {window_size: vectorizer(["".join(src_segs[i: i + window_size])
                                                    for i in range(0, len(src_segs))])
                           for window_size in range(1, max_src_window + 1)}

            src_chapter_out_file = os.path.join(src_book_out_dir,
                                                "{}.pkl".format(chapter_i))
            with open(src_chapter_out_file, 'wb') as f:
                pickle.dump(src_vectors, f)
                print("Saved processed data for original chapters to {}".format(
                    src_chapter_out_file))

            tgt_chapter_in_file = os.path.join(tgt_book_in_dir,
                                               "{}.json".format(chapter_i))
            with open(tgt_chapter_in_file) as f:
                data = json.load(f)

            if 'sentences' in data:
                tgt_segs = data['sentences']
            elif 'segments' in data:
                tgt_segs = data['segments']
            else:
                assert False, "No segment info found in target data"

            print("Encoding target vectors for {} segments...".format(len(tgt_segs)))
            tgt_vectors = {window_size: vectorizer(["".join(tgt_segs[i: i + window_size])
                                                    for i in range(0, len(tgt_segs))])
                           for window_size in range(1, max_tgt_window + 1)}

            tgt_chapter_out_file = os.path.join(tgt_book_out_dir,
                                                "{}.pkl".format(chapter_i))
            with open(tgt_chapter_out_file, 'wb') as f:
                pickle.dump(tgt_vectors, f)
                print("Saved processed data for abridged chapters to {}".format(
                    tgt_chapter_out_file))

    print("\nSaved all segment vector data to:", out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Apply pretrained model to encode sentence-segmented text as vectors.\
                                    These vectors are used for a separate alignment script.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_dir", "-in_dir",
                        help="Directory path where segment text data is located ,\
                        processed according to split_segments.py (.json files): inside are 'original' and 'abridged' folders,\
                        within which are folders named by book, and then one chapter per file inside each book folder.",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir",
                        help="Directory path where segment vector data will be saved in form of .pkl file.",
                        type=str, required=True)
    parser.add_argument("--meta_data_file", "-meta_data_file", help="Path to .json file with book/chapter index.",
                        type=str, required=True)
    parser.add_argument("--vectorizer", "-vectorizer",
                        help="Name of model used for encoding segment vectors,\
                        based on HuggingFace model names.",
                        type=str, required=True)
    parser.add_argument("--max_src_window", "-max_src_window",
                        help="Max number of source text segments to encode in single vector",
                        type=int, default=4, required=False)
    parser.add_argument("--max_tgt_window", "-max_tgt_window",
                        help="Max number of target text segments to encode in single vector",
                        type=int, default=4, required=False)

    args = parser.parse_args()

    transform(args.in_dir,
              args.out_dir,
              args.meta_data_file,
              args.vectorizer,
              args.max_src_window,
              args.max_tgt_window)
