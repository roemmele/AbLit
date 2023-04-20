import json
import os


class VersionChapter():
    def __init__(self,
                 version,
                 text,
                 paragraph_chars,
                 sentence_chars,
                 row_span_chars,
                 overlap_chars):

        self.version = version
        self.text = text
        self.paragraph_chars = paragraph_chars
        self.sentence_chars = sentence_chars
        self.row_span_chars = row_span_chars
        self.overlap_chars = overlap_chars

        self.n_sentences_per_paragraph = self.get_n_sentences_per_psg(
            self.paragraph_chars)
        self.n_sentences_per_row_span = self.get_n_sentences_per_psg(
            self.row_span_chars)

        self.sentence_boundary_set = set(end for _, end
                                         in self.sentence_chars)

    def get_n_sentences_per_psg(self, chars):
        n_segs = [len(self.get_sentence_chars_in_psg(start, end))
                  for start, end in chars]
        return n_segs

    def get_sentence_chars_in_psg(self, start, end):
        seg_chars = []
        for i, (seg_start, seg_end) in enumerate(self.sentence_chars):
            if seg_start >= start and seg_end <= end:
                seg_chars.append([seg_start, seg_end])
            elif seg_start >= end:
                break
        return seg_chars

    def get_chunks(self,
                   merge_n_sentences=None,
                   max_n_sentences=None):

        assert merge_n_sentences or max_n_sentences,\
            "To get chunks, you must specify either the merge_n_sentences or max_n_sentences parameter."

        chunk_chars = []
        chunk_pars_i = []
        chunk_n_segs = []

        if max_n_sentences:
            for par_i, (par_start, par_end) in enumerate(self.paragraph_chars):
                seg_chars = self.get_sentence_chars_in_psg(par_start, par_end)
                if len(seg_chars) > max_n_sentences:
                    for i in range(0, len(seg_chars), max_n_sentences):
                        chunk_chars.append(seg_chars[i:i + max_n_sentences])
                        chunk_pars_i.append([par_i])
                        chunk_n_segs.append(len(chunk_chars[-1]))
                else:
                    if chunk_chars and chunk_n_segs[-1] + len(seg_chars) <= max_n_sentences:
                        chunk_chars[-1].extend(seg_chars)
                        chunk_pars_i[-1].append(par_i)
                        chunk_n_segs[-1] += len(seg_chars)
                    else:
                        chunk_chars.append(seg_chars)
                        chunk_pars_i.append([par_i])
                        chunk_n_segs.append(len(seg_chars))

            assert all(n_segs <= max_n_sentences
                       for n_segs in chunk_n_segs)

        else:
            for par_i, (par_start, par_end) in enumerate(self.paragraph_chars):
                seg_chars = self.get_sentence_chars_in_psg(par_start, par_end)
                prev_is_too_short = (chunk_chars
                                     and chunk_n_segs[-1] < merge_n_sentences)
                cur_is_trailing_and_too_short = (chunk_chars
                                                 and par_i == len(self.paragraph_chars) - 1
                                                 and len(seg_chars) < merge_n_sentences)
                if prev_is_too_short or cur_is_trailing_and_too_short:
                    chunk_chars[-1].extend(seg_chars)
                    chunk_pars_i[-1].append(par_i)
                    chunk_n_segs[-1] += len(seg_chars)
                else:
                    chunk_chars.append(seg_chars)
                    chunk_pars_i.append([par_i])
                    chunk_n_segs.append(len(seg_chars))

            while (len(chunk_chars) > 1
                   and len(chunk_chars[-1]) < merge_n_sentences):
                # import pdb
                # pdb.set_trace()
                fin_chars = chunk_chars.pop(-1)
                fin_pars_i = chunk_pars_i.pop(-1)
                fin_n_segs = chunk_n_segs.pop(-1)

                chunk_chars[-1].extend(fin_chars)
                chunk_pars_i[-1].extend(fin_pars_i)
                chunk_n_segs[-1] += fin_n_segs

            assert (len(chunk_n_segs) == 1 or all(n_segs >= merge_n_sentences
                                                  for n_segs in chunk_n_segs))

        chunk_chars = [[chars[0][0], chars[-1][-1]] for chars in chunk_chars]

        return chunk_chars, chunk_pars_i, chunk_n_segs

    @ property
    def paragraphs(self):

        return (VersionParagraph(version=self.version,
                                 idx=idx,
                                 n_sentences=n_sentences,
                                 start_char=start,
                                 end_char=end,
                                 text=self.text[start:end])
                for (idx,
                     (n_sentences,
                      (start, end)))
                in enumerate(zip(self.n_sentences_per_paragraph,
                                 self.paragraph_chars)))

    def chunks(self,
               merge_n_sentences=None,
               max_n_sentences=None):

        (chunk_chars,
         chunk_pars_i,
         chunk_n_segs) = self.get_chunks(merge_n_sentences=merge_n_sentences,
                                         max_n_sentences=max_n_sentences)

        return (VersionChunk(version=self.version,
                             idx=idx,
                             paragraph_idxs=pars_i,
                             n_sentences=n_segs,
                             start_char=start,
                             end_char=end,
                             text=self.text[start:end])
                for (idx, ((start, end),
                           pars_i,
                           n_segs))
                in enumerate(zip(chunk_chars, chunk_pars_i, chunk_n_segs)))

    @ property
    def sentences(self):

        return (VersionSentence(version=self.version,
                                idx=idx,
                                start_char=start,
                                end_char=end,
                                text=self.text[start:end])
                for idx,
                (start, end)
                in enumerate(self.sentence_chars))

    @ property
    def row_spans(self):
        return (VersionRowSpan(version=self.version,
                               idx=idx,
                               n_sentences=n_sentences,
                               start_char=start,
                               end_char=end,
                               text=self.text[start:end])
                for idx,
                (n_sentences,
                 (start, end))
                in enumerate(zip(self.n_sentences_per_row_span,
                                 self.row_span_chars)))

    @ property
    def start_char(self):
        return 0

    @ property
    def end_char(self):
        return len(self.text)


class VersionPassage():

    def __init__(self,
                 version,
                 start_char,
                 end_char,
                 text,
                 idx=None):

        self.idx = idx
        self.version = version
        self.start_char = start_char
        self.end_char = end_char
        self.text = text

    def __repr__(self):
        return '''Passage(
        version={},
        start_char={},
        end_char={},
        text="{}"
        )'''.format(
            self.version,
            self.start_char,
            self.end_char,
            self.text)


class VersionParagraph(VersionPassage):

    def __init__(self, *args, **kwargs):
        self.n_sentences = kwargs.pop('n_sentences')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''Paragraph(
        version={}
        idx={},
        n_sentences={},
        start_char={},
        end_char={},
        text="{}{}"
        )'''.format(self.version,
                    self.idx,
                    self.n_sentences,
                    self.start_char,
                    self.end_char,
                    self.text[:1000],
                    '...' if len(self.text) > 1000 else '')


class VersionChunk(VersionPassage):

    def __init__(self, *args, **kwargs):
        self.n_sentences = kwargs.pop('n_sentences')
        self.paragraph_idxs = kwargs.pop('paragraph_idxs')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''Chunk(
        version={},
        paragraph_idxs={},
        n_sentences={},
        start_char={},
        end_char={},
        text="{}{}",
        )'''.format(self.version,
                    self.paragraph_idxs,
                    self.n_sentences,
                    self.start_char,
                    self.end_char,
                    self.text[:1000],
                    '...' if len(self.text) > 1000 else '')


class VersionSentence(VersionPassage):

    def __repr__(self):
        return '''Sentence(
        version={},
        idx={},
        start_char={},
        end_char={},
        text="{}"
        )'''.format(self.version,
                    self.idx,
                    self.start_char,
                    self.end_char,
                    self.text)


class VersionRowSpan(VersionPassage):

    def __init__(self, *args, **kwargs):
        self.n_sentences = kwargs.pop('n_sentences')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''RowSpan(
        version={},
        idx={},
        n_sentences={},
        start_char={},
        end_char={},
        text="{}"
        )'''.format(self.version,
                    self.idx,
                    self.n_sentences,
                    self.start_char,
                    self.end_char,
                    self.text)


class Chapter():

    def __init__(self,
                 filepath,
                 book_id,
                 book_title,
                 chapter_idx,
                 chapter_title):
        self.filepath = filepath
        self.book_id = book_id
        self.book_title = book_title
        self.chapter_idx = chapter_idx
        self.chapter_title = chapter_title

        with open(self.filepath) as f:
            self.data = json.load(f)

        self.original_version = VersionChapter(
            version='original',
            text=self.data['original']['text'],
            paragraph_chars=self.data['original']['paragraph_chars'],
            sentence_chars=self.data['original']['segment_chars'],
            row_span_chars=self.data['original']['row_chars'],
            overlap_chars=self.data['original']['overlap_chars'])

        self.abridged_version = VersionChapter(
            version='abridged',
            text=self.data['abridged']['text'],
            paragraph_chars=self.data['abridged']['paragraph_chars'],
            sentence_chars=self.data['abridged']['segment_chars'],
            row_span_chars=self.data['abridged']['row_chars'],
            overlap_chars=self.data['abridged']['overlap_chars'])

        self.overlaps = self.get_overlaps(
            [[0, len(self.original_version.text)]])[0]

        self.row_overlaps = self.get_overlaps(
            self.original_version.row_span_chars)

        self.paragraph_overlaps = self.get_overlaps(
            self.original_version.paragraph_chars)
        self.paragraph_abridgements = self.get_abridgements(
            self.original_version.paragraph_chars)

        self.sentence_overlaps = self.get_overlaps(
            self.original_version.sentence_chars)
        self.sentence_abridgements = self.get_abridgements(
            self.original_version.sentence_chars)

    def get_overlaps(self, chars):

        overlaps = []
        for start, end in chars:
            overlaps.append(self.get_overlaps_for_psg(start, end))
        return overlaps

    def update_abridgement_chars(self,
                                 idx,
                                 overlaps,
                                 chars_lookup):

        overlap_set = set([char for overlap in overlaps
                           for char in range(overlap[0], overlap[1])])

        if not chars_lookup:
            chars_lookup.update({char: idx for char in overlap_set})
            return chars_lookup  # , False

        conflict_range = range(overlaps[0][0],
                               max(chars_lookup) + 1)
        conflict_chars = sorted(list(conflict_range))

        if not conflict_chars:
            chars_lookup.update({char: idx for char in overlap_set})
            return chars_lookup  # , False

        these_chars_are_in_new = set([char for char in conflict_chars
                                      if char in overlap_set])
        these_chars_are_in_old = set([char for char in conflict_chars
                                      if char in chars_lookup])

        new_and_old_chars = sorted(list(these_chars_are_in_new.union(
            these_chars_are_in_old)))

        are_these_chars_old_only = []
        n_contiguous_old = 0
        for char in new_and_old_chars:
            if char not in overlap_set and char in chars_lookup:
                are_these_chars_old_only.append(n_contiguous_old + 1)
                n_contiguous_old += 1
            else:
                are_these_chars_old_only.append(0)
                n_contiguous_old = 0

        contiguous_old_len, contiguous_old_end_i = max((char, i + 1) for i, char
                                                       in enumerate(are_these_chars_old_only))
        contiguous_old_start_i = contiguous_old_end_i - contiguous_old_len
        if contiguous_old_len > contiguous_old_start_i:
            if contiguous_old_end_i < len(new_and_old_chars):
                new_psg_starts_at = new_and_old_chars[contiguous_old_end_i]
            else:
                new_psg_starts_at = None
        else:
            new_psg_starts_at = new_and_old_chars[0]

        if new_psg_starts_at != None:
            for char in new_and_old_chars:
                if char in these_chars_are_in_new:
                    if char >= new_psg_starts_at:
                        chars_lookup[char] = idx
                elif char >= new_psg_starts_at:
                    chars_lookup.pop(char)

        for char in overlap_set - set(new_and_old_chars):
            chars_lookup[char] = idx

        return chars_lookup

    def smooth_abridgements(self, abridgements, next_start, next_end):

        prev_end = abridgements[-1][-1]

        boundary_chars = list(self.abridged_version.sentence_boundary_set.intersection(
            range(prev_end,
                  next_start + 1))
        )

        if boundary_chars:
            next_start = boundary_chars[0]

            for char in range(prev_end, next_start):

                for i in range(len(abridgements) - 1, -1, -1):
                    if abridgements[i][0] != abridgements[i][1]:
                        abridgements[i] = [abridgements[i][0],
                                           char + 1]
                        abridgements[i + 1:] = [[char + 1, char + 1]
                                                for j in range(i + 1, len(abridgements))]
                        break

        return abridgements

    def reassemble_psgs_from_chars_lookup(self, idxs, chars_lookup):

        reversed_chars_lookup = {}

        for char, idx in chars_lookup.items():
            if idx in reversed_chars_lookup:
                reversed_chars_lookup[idx].append(char)
            else:
                reversed_chars_lookup[idx] = [char]

        psgs = []

        for idx in idxs:
            if idx not in reversed_chars_lookup:
                if idx == 0:
                    psg = [0, 0]
                else:
                    psg = [psgs[-1][-1], psgs[-1][-1]]
                psgs.append(psg)
            else:
                chars = reversed_chars_lookup[idx]
                psgs.append([min(chars), max(chars) + 1])

        return psgs

    def get_abridgements(self, chars):

        chars_lookup = {}

        for idx, (start, end) in enumerate(chars):

            overlaps = self.get_abridged_overlaps_for_psg(start, end)

            if overlaps:
                chars_lookup = self.update_abridgement_chars(idx=idx,
                                                             overlaps=overlaps,
                                                             chars_lookup=chars_lookup)

        psgs = self.reassemble_psgs_from_chars_lookup(idxs=range(len(chars)),
                                                      chars_lookup=chars_lookup)

        abridgements = []
        for i, psg in enumerate(psgs):
            st, en = chars[i]
            if i == 0:
                abridgement_start = 0
            else:
                abridgements = self.smooth_abridgements(abridgements,
                                                        next_start=psg[0],
                                                        next_end=psg[1])
                abridgement_start = abridgements[-1][-1]

            if i == len(psgs) - 1:
                abridgement_end = len(self.abridged_version.text)
            else:
                abridgement_end = psg[1]

            abridgements.append([abridgement_start, abridgement_end])

        return abridgements

    def get_overlaps_for_psg(self, start, end):

        def get_overlap_chars(overlap_chars):
            psg_overlap_chars = []

            for i, (overlap_start, overlap_end) in enumerate(overlap_chars):
                if overlap_start >= start and overlap_end <= end:
                    psg_overlap_chars.append([overlap_start, overlap_end])
                elif overlap_start >= end:
                    break
            return psg_overlap_chars

        psg_overlap_chars = get_overlap_chars(
            self.original_version.overlap_chars)

        overlaps = [
            VersionPassage(version='original',
                           text=self.original_version.text[overlap_start:overlap_end],
                           start_char=overlap_start,
                           end_char=overlap_end)
            for overlap_start, overlap_end in psg_overlap_chars]

        return overlaps

    def get_abridged_overlaps_for_psg(self, start, end):
        abridged_psg_overlap_chars = []

        for i, (overlap_start, overlap_end) in enumerate(self.original_version.overlap_chars):
            if overlap_start >= start and overlap_end <= end:
                abridged_psg_overlap_chars.append(
                    self.abridged_version.overlap_chars[i])

        if abridged_psg_overlap_chars:
            return abridged_psg_overlap_chars
        else:
            return None

    @ property
    def original(self):
        return self.original_version.text

    @ property
    def abridged(self):
        return self.abridged_version.text

    @ property
    def original_start_char(self):
        return 0

    @ property
    def original_end_char(self):
        return len(self.original_version.text)

    @ property
    def paragraphs(self):

        for i, orig_par in enumerate(self.original_version.paragraphs):
            abrg_start, abrg_end = self.paragraph_abridgements[i]
            yield Paragraph(idx=orig_par.idx,
                            n_original_sentences=orig_par.n_sentences,
                            original_start_char=orig_par.start_char,
                            original_end_char=orig_par.end_char,
                            original=orig_par.text,
                            abridged_start_char=abrg_start,
                            abridged_end_char=abrg_end,
                            abridged=self.abridged_version.text[abrg_start:abrg_end],
                            overlaps=self.paragraph_overlaps[i])

    def chunks(self,
               merge_n_sentences=None,
               max_n_sentences=None):

        (chunk_chars,
         chunk_pars_i,
         chunk_n_segs) = self.original_version.get_chunks(merge_n_sentences=merge_n_sentences,
                                                          max_n_sentences=max_n_sentences)

        chunk_overlaps = self.get_overlaps(chunk_chars)

        chunk_abridgements = self.get_abridgements(chunk_chars)

        return (Chunk(idx=idx,
                      original_paragraph_idxs=pars_i,
                      n_original_sentences=n_segs,
                      original_start_char=start,
                      original_end_char=end,
                      original=self.original_version.text[start:end],
                      abridged_start_char=abrg_start,
                      abridged_end_char=abrg_end,
                      abridged=self.abridged_version.text[abrg_start:abrg_end],
                      overlaps=overlaps)
                for (idx,
                     ((start, end),
                      pars_i,
                      n_segs,
                      overlaps,
                      (abrg_start, abrg_end)))
                in enumerate(zip(chunk_chars,
                                 chunk_pars_i,
                                 chunk_n_segs,
                                 chunk_overlaps,
                                 chunk_abridgements)))

    @ property
    def sentences(self):

        for i, orig_sentence in enumerate(self.original_version.sentences):
            abrg_start, abrg_end = self.sentence_abridgements[i]
            yield Sentence(idx=orig_sentence.idx,
                           original_start_char=orig_sentence.start_char,
                           original_end_char=orig_sentence.end_char,
                           original=orig_sentence.text,
                           abridged_start_char=abrg_start,
                           abridged_end_char=abrg_end,
                           abridged=self.abridged_version.text[abrg_start:abrg_end],
                           overlaps=self.sentence_overlaps[i])

    @ property
    def rows(self):

        for i, orig_span in enumerate(self.original_version.row_spans):
            abrg_start, abrg_end = self.abridged_version.row_span_chars[i]
            yield Row(idx=orig_span.idx,
                      n_original_sentences=orig_span.n_sentences,
                      original_start_char=orig_span.start_char,
                      original_end_char=orig_span.end_char,
                      original=self.original_version.text[orig_span.start_char:orig_span.end_char],
                      n_abridged_sentences=self.abridged_version.n_sentences_per_row_span[i],
                      abridged_start_char=abrg_start,
                      abridged_end_char=abrg_end,
                      abridged=self.abridged_version.text[abrg_start:abrg_end],
                      overlaps=self.row_overlaps[i])

    def __repr__(self):
        return '''Chapter(
        book_id={},
        book_title={},
        chapter_idx={},
        chapter_title={},
        original="{}{}"
        abridged="{}{}"
        )'''.format(
            self.book_id,
            self.book_title,
            self.chapter_idx,
            self.chapter_title,
            self.original_version.text[:1000],
            '...' if len(self.original_version.text) > 1000 else '',
            self.abridged_version.text[:1000],
            '...' if len(self.abridged_version.text) > 1000 else '')


class Passage():

    def __init__(self,
                 original_start_char,
                 original_end_char,
                 original,
                 abridged,
                 abridged_start_char,
                 abridged_end_char,
                 overlaps,
                 idx=None):

        self.original_start_char = original_start_char
        self.original_end_char = original_end_char
        self.original = original

        self.abridged_start_char = abridged_start_char
        self.abridged_end_char = abridged_end_char
        self.abridged = abridged

        self.overlaps = overlaps
        self.idx = idx

    def __repr__(self):
        return '''Passage(
        original_start_char={},
        original_end_char={},
        original="{}",
        abridged_start_char={},
        abridged_end_char={},
        abridged="{}",
        )'''.format(
            self.original_start_char,
            self.original_end_char,
            self.original,
            self.abridged_start_char,
            self.abridged_end_char,
            self.abridged)


class Row(Passage):

    def __init__(self, *args, **kwargs):
        self.n_original_sentences = kwargs.pop('n_original_sentences')
        self.n_abridged_sentences = kwargs.pop('n_abridged_sentences')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''Row(
        idx={},
        n_original_sentences={},
        original_start_char={},
        original_end_char={},
        original="{}",
        n_abridged_sentences={},
        abridged_start_char={},
        abridged_end_char={},
        abridged="{}",
        )'''.format(self.idx,
                    self.n_original_sentences,
                    self.original_start_char,
                    self.original_end_char,
                    self.original,
                    self.n_abridged_sentences,
                    self.abridged_start_char,
                    self.abridged_end_char,
                    self.abridged)


class Paragraph(Passage):

    def __init__(self, *args, **kwargs):
        self.n_original_sentences = kwargs.pop('n_original_sentences')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''Paragraph(
        idx={},
        n_original_sentences={},
        original_start_char={},
        original_end_char={},
        original="{}{}",
        abridged_start_char={},
        abridged_end_char={},
        abridged="{}{}"
        )'''.format(self.idx,
                    self.n_original_sentences,
                    self.original_start_char,
                    self.original_end_char,
                    self.original[:1000],
                    '...' if len(self.original) > 1000 else '',
                    self.abridged_start_char,
                    self.abridged_end_char,
                    self.abridged[:1000] if self.abridged else '',
                    '...' if (self.abridged and len(self.abridged) > 1000) else '')


class Chunk(Passage):

    def __init__(self, *args, **kwargs):
        self.n_original_sentences = kwargs.pop('n_original_sentences')
        self.original_paragraph_idxs = kwargs.pop('original_paragraph_idxs')
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return '''Chunk(
        idx={},
        original_paragraph_idxs={},
        n_original_sentences={},
        original_start_char={},
        original_end_char={},
        original="{}{}",
        abridged_start_char={},
        abridged_end_char={},
        abridged="{}{}"
        )'''.format(self.idx,
                    self.original_paragraph_idxs,
                    self.n_original_sentences,
                    self.original_start_char,
                    self.original_end_char,
                    self.original[:1000],
                    '...' if len(self.original) > 1000 else '',
                    self.abridged_start_char,
                    self.abridged_end_char,
                    self.abridged[:1000] if self.abridged else '',
                    '...' if (self.abridged and len(self.abridged) > 1000) else '')


class Sentence(Passage):

    def __repr__(self):
        return '''Sentence(
        idx={},
        original_start_char={},
        original_end_char={},
        original="{}",
        abridged_start_char={},
        abridged_end_char={},
        abridged="{}"
        )'''.format(self.idx,
                    self.original_start_char,
                    self.original_end_char,
                    self.original,
                    self.abridged_start_char,
                    self.abridged_end_char,
                    self.abridged)


class Book():

    def __init__(self,
                 dirpath,
                 book_id,
                 book_title,
                 author,
                 chapter_titles,
                 chapter_idxs):

        self.dirpath = dirpath
        self.book_id = book_id
        self.book_title = book_title
        self.author = author
        self.chapter_titles = chapter_titles
        self.chapter_idxs = chapter_idxs
        self.chapter_filepaths = [os.path.join(self.dirpath,
                                               "{}.json".format(chapter_idx))
                                  for chapter_idx in self.chapter_idxs]
        self.chapters = [self.chapter(i)
                         for i in range(len(self.chapter_idxs))]

    def chapter(self, i):
        return Chapter(filepath=self.chapter_filepaths[i],
                       book_id=self.book_id,
                       book_title=self.book_title,
                       chapter_idx=self.chapter_idxs[i],
                       chapter_title=self.chapter_titles[i])

    def get_chapter_by_idx(self, chapter_idx):

        for i, idx in enumerate(self.chapter_idxs):
            if idx == chapter_idx:
                return self.chapters[i]

        assert False, "chapter_idx {} not in book {}".format(
            chapter_idx, self.book_id)

    def __repr__(self):
        return '''Book(
        dirpath={},
        id={},
        book_title={},
        author={},
        chapters={}
        )'''.format(
            self.dirpath,
            self.book_id,
            self.book_title,
            self.author,
            {chapter_idx: chapter_title for chapter_idx, chapter_title
             in zip(self.chapter_idxs, self.chapter_titles)}
        )


class AblitDataset():

    def __init__(self, dirpath, partition=None):
        self.dirpath = dirpath
        self.partition = partition
        with open(os.path.join(self.dirpath, 'meta_data.json')) as f:
            meta_data = json.load(f)
        self.load_books(meta_data)

    def load_books(self, meta_data):
        self.books = []
        for book_id, book_info in meta_data.items():
            if self.partition:
                partition_idxs = book_info["{}_chapter_idxs".format(
                    self.partition)]
            else:
                partition_idxs = book_info['chapter_idxs']

            partition_titles = [book_info['chapter_titles'][chapter_idx]
                                for chapter_idx in partition_idxs]

            book = Book(
                dirpath=os.path.join(self.dirpath, book_id),
                book_id=book_id,
                book_title=book_info['book_title'],
                author=book_info['author'],
                chapter_titles=partition_titles,
                chapter_idxs=partition_idxs
            )

            self.books.append(book)

    def get_book_by_id(self, book_id):

        for book in self.books:
            if book.book_id == book_id:
                return book

        assert False, "book_id {} not in dataset".format(book_id)

    def __repr__(self):
        return '''AblitDataset(
        dirpath={},
        partition={},
        books={},
        )'''.format(self.dirpath,
                    self.partition,
                    [{book.book_id: {'book_title': book.book_title,
                                     'author': book.author}}
                     for book in self.books]
                    )
