import os
import json
import argparse
import math
import re
import torch
import nltk
from nltk.tokenize import sent_tokenize
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

nltk.download('punkt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_segments(paragraph):

    segments = []

    nltk_sents = sent_tokenize(paragraph)
    cursor_position = 0

    for i, sent in enumerate(nltk_sents):

        seg_match = re.match(re.escape(sent) + r"[\s]*",
                             paragraph[cursor_position:])
        assert seg_match,\
            "Error in sentence segmentation for paragraph {}".format(paragraph)

        _, seg_end = seg_match.span()

        segments.append(paragraph[cursor_position: cursor_position + seg_end])

        cursor_position += seg_end

    assert "".join(segments) == paragraph

    return segments


def get_chunks(text,
               merge_n_segments=None,
               max_n_segments=None):

    assert merge_n_segments or max_n_segments,\
        "To get chunks, you must specify either the merge_n_segments or max_n_segments parameter."

    paragraphs = text.split("\n")
    paragraphs = [(par + "\n" if i < len(paragraphs) - 1 else par)
                  for i, par in enumerate(paragraphs)]

    chunks = []

    if max_n_segments:
        for par_i, par in enumerate(paragraphs):
            segs = get_segments(par)
            if len(segs) > max_n_segments:
                for i in range(0, len(segs), max_n_segments):
                    chunks.append(segs[i:i + max_n_segments])
            else:
                if chunks and len(chunks[-1]) + len(segs) <= max_n_segments:
                    chunks[-1].extend(segs)
                else:
                    chunks.append(segs)

    else:
        for par_i, par in enumerate(paragraphs):
            segs = get_segments(par)
            prev_is_too_short = (chunks
                                 and len(chunks[-1]) < merge_n_segments)
            cur_is_trailing_and_too_short = (chunks
                                             and par_i == len(paragraphs) - 1
                                             and len(segs) < merge_n_segments)
            if prev_is_too_short or cur_is_trailing_and_too_short:
                chunks[-1].extend(segs)
            else:
                chunks.append(segs)

        while (len(chunks) > 1
               and len(chunks[-1]) < merge_n_segments):
            # import pdb
            # pdb.set_trace()
            fin_chunk = chunks.pop(-1)
            chunks[-1].extend(fin_chunk)

    chunks = ["".join(chunk_segs) for chunk_segs in chunks]

    assert "".join(chunks) == text

    return chunks


def split_passages(text,
                   passage_type,
                   chunk_merge_n_segments=None,
                   chunk_max_n_segments=None,):

    if passage_type == 'segment':
        psgs = []
        pars = text.split("\n")
        for i, par in enumerate(pars):
            if i < len(pars) - 1:
                par += "\n"
            psgs.extend(get_segments(par))

    elif passage_type == 'paragraph':
        pars = text.split("\n")
        psgs = [(par + "\n" if i < len(pars) - 1 else par)
                for i, par in enumerate(pars)]

    elif passage_type == 'chunk':
        psgs = get_chunks(text,
                          merge_n_segments=chunk_merge_n_segments,
                          max_n_segments=chunk_max_n_segments)

    elif passage_type == None:
        psgs = [text]

    else:
        assert False, "Invalid passage type"

    assert "".join(psgs) == text

    return psgs


def get_data_iterator_for_ablit(dirpath,
                                partition,
                                passage_type=None,
                                chunk_merge_n_segments=None,
                                chunk_max_n_segments=None):
    from abridgement_dataset import AbridgementDataset

    data_interface = AbridgementDataset(dirpath, partition)

    for book_i, book in enumerate(data_interface.books):
        for chapter in book.chapters:

            if passage_type == 'segment':
                psgs = chapter.segments
            elif passage_type == 'paragraph':
                psgs = chapter.paragraphs
            elif passage_type == 'chunk':
                psgs = chapter.chunks(merge_n_segments=chunk_merge_n_segments,
                                      max_n_segments=chunk_max_n_segments)
            elif passage_type == None:
                psgs = [chapter]
            else:
                assert False, "Invalid passage type"

            item = {'passages': [psg.original for psg in psgs],
                    'book_id': book.book_id,
                    'chapter_idx': chapter.chapter_idx,
                    # 'reference_abridgement': chapter.abridged
                    }

            yield item

        # if book_i == 1:
        #     break


def get_data_iterator_for_file(filepath,
                               passage_type=None,
                               chunk_merge_n_segments=None,
                               chunk_max_n_segments=None,):

    with open(filepath) as f:
        data_from_file = json.load(f)

    for item_i, item in enumerate(data_from_file):
        assert 'text' in item, "Input file must be .json object consisting of list\
                                with 'text' entry in each list item"

        psgs = split_passages(text=item['text'],
                              passage_type=passage_type,
                              chunk_merge_n_segments=chunk_merge_n_segments,
                              chunk_max_n_segments=chunk_max_n_segments)

        item = {key: val for key, val in item.items()
                if key != 'text'}
        item['passages'] = psgs

        yield item


def load_pipeline(tokenizer_path, model_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    return tokenizer, model


def generate(passages,
             tokenizer,
             model,
             max_source_length=1024,
             max_target_length=1024,
             do_sample=False,
             n_beams=1,
             temperature=1.0,
             top_p=1.0,
             batch_size=16):

    if tokenizer.name_or_path.startswith("t5"):
        passages = ["summarize: " + psg for psg in passages]

    generated = []
    n_batches = int(math.ceil(len(passages) / batch_size))
    for i, batch_idx in enumerate(range(0, len(passages), batch_size)):

        if (i + 1) % 5 == 0:
            print("Completed batch {}/{}".format(i + 1, n_batches))

        batch_passages = passages[batch_idx:batch_idx + batch_size]
        batch_input_ids = tokenizer(batch_passages,
                                    max_length=max_source_length,
                                    return_tensors="pt",
                                    padding=True,
                                    truncation=True).input_ids.to(device)
        batch_output_ids = model.generate(batch_input_ids,
                                          max_length=max_target_length,
                                          do_sample=do_sample,
                                          num_beams=n_beams,
                                          temperature=temperature,
                                          top_p=top_p,
                                          no_repeat_ngram_size=3
                                          )
        batch_gen_texts = [tokenizer.decode(output_ids,
                                            skip_special_tokens=True)
                           for output_ids in batch_output_ids]
        generated.extend(batch_gen_texts)

    return generated


def generation_for_input_file(input_file,
                              tokenizer,
                              model,
                              output_file,
                              passage_type=None,
                              chunk_merge_n_segments=None,
                              chunk_max_n_segments=None,
                              max_source_length=1024,
                              max_target_length=1024,
                              decoding_method='greedy',
                              n_beams=1,
                              temperature=1.0,
                              top_p=1.0,
                              batch_size=16):

    data_iterator = get_data_iterator_for_file(filepath=input_file,
                                               passage_type=passage_type,
                                               chunk_merge_n_segments=chunk_merge_n_segments,
                                               chunk_max_n_segments=chunk_max_n_segments)
    # import pdb
    # pdb.set_trace()

    items = []

    print("Generating abridgements...")
    for item_i, item in enumerate(data_iterator):

        print("Item {}...".format(item_i + 1))

        passages = item.pop('passages')

        generated = generate(passages=passages,
                             tokenizer=tokenizer,
                             model=model,
                             max_source_length=max_source_length,
                             max_target_length=max_target_length,
                             do_sample=True if decoding_method == 'sample' else False,
                             n_beams=n_beams,
                             temperature=temperature,
                             top_p=top_p,
                             batch_size=batch_size)

        #item['text'] = "".join(passages)
        item['predicted_abridgement'] = "".join(generated)
        items.append(item)

    with open(output_file, 'w') as f:
        json.dump(items, f, indent=4)

    print("Saved generated abridgements to", output_file)


def generation_for_ablit_dataset(input_dir,
                                 partition,
                                 tokenizer,
                                 model,
                                 output_file,
                                 passage_type=None,
                                 chunk_merge_n_segments=None,
                                 chunk_max_n_segments=None,
                                 max_source_length=1024,
                                 max_target_length=1024,
                                 decoding_method='greedy',
                                 n_beams=1,
                                 temperature=1.0,
                                 top_p=1.0,
                                 batch_size=16):

    data_iterator = get_data_iterator_for_ablit(
        dirpath=input_dir,
        partition=partition,
        passage_type=passage_type,
        chunk_merge_n_segments=chunk_merge_n_segments,
        chunk_max_n_segments=chunk_max_n_segments)

    outputs = []

    print("Generating abridgements...")
    for chapter in data_iterator:

        print("Book {}, Chapter {}...".format(chapter['book_id'],
                                              chapter['chapter_idx']))

        # book_output_dir = os.path.join(output_dir, chapter['book_id'])
        # if not os.path.exists(book_output_dir):
        #     os.mkdir(book_output_dir)

        generated = generate(passages=chapter['passages'],
                             tokenizer=tokenizer,
                             model=model,
                             max_source_length=max_source_length,
                             max_target_length=max_target_length,
                             do_sample=True if decoding_method == 'sample' else False,
                             n_beams=n_beams,
                             temperature=temperature,
                             top_p=top_p,
                             batch_size=batch_size)

        output = {
            'book_id': chapter['book_id'],
            'chapter_idx': chapter['chapter_idx'],
            'predicted_abridgement': "".join(generated)
        }

        outputs.append(output)

    with open(output_file, 'w') as f:
        json.dump(outputs, f, indent=4)

    print("Saved generated abridgements to", output_file)


def generation(ablit_dir,
               input_file,
               ablit_partition,
               tokenizer,
               model,
               output_file,
               passage_type,
               chunk_merge_n_segments,
               chunk_max_n_segments,
               max_source_length,
               max_target_length,
               decoding_method,
               n_beams,
               temperature,
               top_p,
               batch_size):

    if ablit_dir:
        generation_for_ablit_dataset(input_dir=ablit_dir,
                                     partition=ablit_partition,
                                     tokenizer=tokenizer,
                                     model=model,
                                     output_file=output_file,
                                     passage_type=passage_type,
                                     chunk_merge_n_segments=chunk_merge_n_segments,
                                     chunk_max_n_segments=chunk_max_n_segments,
                                     max_source_length=max_source_length,
                                     max_target_length=max_target_length,
                                     decoding_method=decoding_method,
                                     n_beams=n_beams,
                                     temperature=temperature,
                                     top_p=top_p,
                                     batch_size=batch_size)
    else:
        assert input_file
        generation_for_input_file(input_file=input_file,
                                  tokenizer=tokenizer,
                                  model=model,
                                  output_file=output_file,
                                  passage_type=passage_type,
                                  chunk_merge_n_segments=chunk_merge_n_segments,
                                  chunk_max_n_segments=chunk_max_n_segments,
                                  max_source_length=max_source_length,
                                  max_target_length=max_target_length,
                                  decoding_method=decoding_method,
                                  n_beams=n_beams,
                                  temperature=temperature,
                                  top_p=top_p,
                                  batch_size=batch_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Use a trained generation model to generate abridgements of input texts.")

    '''Data loading arguments'''
    parser.add_argument("--ablit_dir", "-ablit_dir",
                        help="Directory path to AbLit abridgement dataset.",
                        type=str,
                        required=False)
    parser.add_argument("--partition", "-partition",
                        help="Partition of AbLit dataset (train, dev, or test) for which to generate abridgments.",
                        type=str,
                        required=False,
                        default='test')
    parser.add_argument("--input_file", "-input_file",
                        help="As an alternative to loading the AbLit dataset,\
                        provide one or more .json files containing\
                        the texts for which to generate abridgements.\
                        Each .json object should consist of a list,\
                        where each list item has a 'text' key (along with any optional other keys identifying the list).",
                        type=str,
                        required=False)

    '''Model path arguments'''
    parser.add_argument("--tokenizer_path", "-tokenizer_path",
                        help="Path to HuggingFace tokenizer.",
                        type=str,
                        required=True)
    parser.add_argument("--model_path", "-model_path",
                        help="Path to trained HuggingFace model.",
                        type=str,
                        required=True)

    '''Data preparation arguments'''
    parser.add_argument("--passage_type", "-passage_type",
                        help="Specify how input texts should be split,\
                        such that each resulting split passage will given as input to the generation model.\
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

    '''Model generation arguments'''
    parser.add_argument("--max_source_length", "-max_source_length",
                        help="Maximum length of input passages provided to model, provided to HuggingFace tokenizer.",
                        type=int,
                        required=False,
                        default=1024)
    parser.add_argument("--max_target_length", "-max_target_length",
                        help="Maximum length of abridgements generated by model, provided to HuggingFace generation function.",
                        type=int,
                        required=False,
                        default=1024)
    parser.add_argument("--decoding_method", "-decoding_method",
                        help="Method for performing generation: either 'sample' or 'greedy'.\
                        Defaults to greedy.",
                        type=str,
                        required=False,
                        choices=['greedy', 'sample'],
                        default='greedy')
    parser.add_argument("--n_beams", "-n_beams",
                        help="If decoding method = 'greedy', number of beams to use for beam search.\
                        By default, n_beams=1, meaning beam search will not be used.",
                        type=int,
                        required=False,
                        default=1)
    parser.add_argument("--temperature", "-temperature",
                        help="If decoding method = 'sample', specify temperature parameter for sampling.\
                        Defaults to 1.0.",
                        type=float,
                        required=False,
                        default=1.0)
    parser.add_argument("--top_p", "-top_p",
                        help="If decoding method = 'sample', specify probablility parameter for nucleus sampling.\
                        Defaults to 1.0.",
                        type=float,
                        required=False,
                        default=1.0)
    parser.add_argument("--batch_size", "-batch_size",
                        help="Batch size for generation function.",
                        type=int,
                        required=False,
                        default=16)

    parser.add_argument("--output_file", "-output_file",
                        help="If input is provided via input_file,\
                        provide filepath for .json file to which the generate abridgements will be saved.\
                        Each .json object will consist of a list,\
                        where each list item has a 'predicted_abridgement' key\
                        along with any additional key values (either those contained in the input file,\
                        or book/chapter identifiers if using the AbLit data).",
                        type=str,
                        required=False)

    args = parser.parse_args()

    print("Loading tokenizer and model...")
    tokenizer, model = load_pipeline(tokenizer_path=args.tokenizer_path,
                                     model_path=args.model_path)

    generation(ablit_dir=args.ablit_dir,
               input_file=args.input_file,
               ablit_partition=args.partition,
               tokenizer=tokenizer,
               model=model,
               output_file=args.output_file,
               passage_type=args.passage_type,
               chunk_merge_n_segments=args.chunk_merge_n_segments,
               chunk_max_n_segments=args.chunk_merge_n_segments,
               max_source_length=args.max_source_length,
               max_target_length=args.max_target_length,
               decoding_method=args.decoding_method,
               n_beams=args.n_beams,
               temperature=args.temperature,
               top_p=args.top_p,
               batch_size=args.batch_size)
