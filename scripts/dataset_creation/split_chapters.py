import os
import argparse
import re
import json
import string


def split_chapters(lines,
                   section_prefixes=["Book", "BOOK", "Volume", "Volume"],
                   end_prefixes=["*** END", "***END", "The End", "THE END"],
                   chapter_prefixes=['Chapter', 'CHAPTER'],
                   other_prefixes=['Prologue', 'PROLOGUE',
                                   'Epilogue', 'EPILOGUE' 'Preface', 'PREFACE']):

    chapters = {}

    def finalize_chapter(chapter):
        chapter['chapter_title'] = chapter['chapter_title']
        chapter['book_title'] = chapter['book_title']
        chapter['chapter_text'] = re.sub('[ \t]{2,}',
                                         ' ',
                                         chapter['chapter_text']).strip()

        chapters[(chapter['book_num'],
                  chapter['book_title'],
                  chapter['chapter_num'],
                  chapter['chapter_title'])] = chapter['chapter_text']

    def init_new_chapter(book_num=None, book_title=""):

        chapter = {'book_num': book_num,
                   'book_title': book_title,
                   'chapter_num': None,
                   'chapter_title': "",
                   'chapter_text': ""}

        return chapter

    cur_chapter = init_new_chapter()

    for line in lines:
        if any(line.strip().startswith(prefix) for prefix in end_prefixes):
            if cur_chapter['chapter_num'] != None:
                print(
                    "*** DETECTED END OF CHAPTER {} ***\n".format(cur_chapter['chapter_num']))
                finalize_chapter(cur_chapter)
                cur_chapter = init_new_chapter()
                break

        if any(line.strip().startswith(prefix) for prefix in section_prefixes):
            if cur_chapter['chapter_num'] != None:
                if not cur_chapter['chapter_text']:
                    print(
                        "No detected book text. Table of contents? {}".format(
                            cur_chapter['chapter_num']))
                else:
                    finalize_chapter(cur_chapter)
                    cur_chapter = init_new_chapter()

            line_tokens = line.replace("--", " ").strip().split()  # Hack
            try:
                cur_chapter['book_num'] = line_tokens[1]
                cur_chapter['book_num'] = re.findall("[0-9a-zA-Z-]+",
                                                     cur_chapter['book_num'])[0]
            except:
                raise ValueError(
                    "Book without number detected:".format(line.strip()))
            # Chapter number is roman numeral or words
            if not cur_chapter['book_num'].isnumeric():
                try:
                    cur_chapter['book_num'] = roman_to_int(
                        cur_chapter['book_num'])
                except:
                    try:
                        cur_chapter['book_num'] = text_to_int(
                            cur_chapter['book_num'])
                    except:
                        raise ValueError(
                            "Can't parse book number in".format(line.strip()))

            cur_chapter['book_num'] = int(cur_chapter['book_num'])
            cur_chapter['book_title'] = ""

            if len(line_tokens) > 2:  # there might be a book name on this same line
                print("Title name detected in book title header: {}".format(
                    line_tokens))
                try:
                    if line_tokens[-1][-1] == ".":
                        line_tokens[-1] = line_tokens[-1][:-1]
                    title = " ".join(line_tokens[2:]).strip()
                    if title[-1] == ".":
                        title = title[:-1]
                    cur_chapter['book_title'] += title
                except:
                    raise(
                        ValueError, "Failed to detect book title in line {}".format(line))

        elif (any(line.strip().startswith(prefix) for prefix in chapter_prefixes)
                or any(line.strip() == prefix for prefix in other_prefixes)):

            line = line.strip()
            if (cur_chapter['book_num'] and cur_chapter['chapter_num'] == None):
                if cur_chapter['chapter_text']:
                    print("Warning: skipping text given between book\
                          heading {} and chapter heading {}". format(cur_chapter['chapter_title'],
                                                                     line.strip()))
                    cur_chapter['chapter_text'] = ""
            else:
                if cur_chapter['chapter_num'] != None:
                    if not cur_chapter['chapter_text']:
                        # No chapter text for previously detected chapter (probably just table of contents)
                        print(
                            "No detected chapter text. Table of contents? {}".format(
                                cur_chapter['chapter_num']))
                    else:
                        finalize_chapter(cur_chapter)
                    cur_chapter = init_new_chapter(cur_chapter['book_num'],
                                                   cur_chapter['book_title'])

            matched_prefix = [prefix for prefix in other_prefixes if
                              line.strip() == prefix]
            if matched_prefix:  # Non-numbered section
                print("Detected a non-numbered chapter: {}".format(line.strip()))
                matched_prefix = matched_prefix[0]
                cur_chapter['chapter_num'] = 0
                title = matched_prefix[0] + matched_prefix[1:].lower()
                if title[-1] == ".":
                    title = title[:-1]
                cur_chapter['chapter_title'] += title.strip()

            else:  # Must parse chapter number
                line_tokens = line.strip().split()
                print("Detected chapter heading: {}".format(line.strip()))
                try:
                    cur_chapter['chapter_num'] = line_tokens[1]
                    cur_chapter['chapter_num'] = re.findall(
                        "[0-9a-zA-Z-]+", cur_chapter['chapter_num'])[0]
                except:
                    raise ValueError(
                        "Chapter without number detected:".format(line.strip()))
                # Chapter number is roman numeral or words
                if not cur_chapter['chapter_num'].isnumeric():
                    try:
                        cur_chapter['chapter_num'] = roman_to_int(
                            cur_chapter['chapter_num'])
                    except:
                        try:
                            cur_chapter['chapter_num'] = text_to_int(
                                cur_chapter['chapter_num'])
                        except:
                            raise ValueError(
                                "Can't parse chapter number in".format(line.strip()))

                cur_chapter['chapter_num'] = int(cur_chapter['chapter_num'])

                if len(line_tokens) > 2:  # there might be a chapter name on this same line
                    print("Title name detected in chapter header: {}".format(
                        line_tokens))
                    try:
                        title = " ".join(line_tokens[2:]).strip()
                        if title[-1] == ".":
                            title = title[:-1]
                        cur_chapter['chapter_title'] = title.strip()
                    except:
                        raise(
                            ValueError, "Failed to detect chapter title in line {}".format(line))

        # Continue looking for title
        elif (cur_chapter['chapter_num'] != None
              and cur_chapter['chapter_title'] == ""
              and not cur_chapter['chapter_text']):

            if not line.strip():
                continue
            elif line.strip() == "-":  # Hack
                cur_chapter['chapter_text'] = " "
            # no end-of-line punctuation; assume this line is a title
            elif (re.findall("[0-9a-zA-Z]+", line.strip())
                  and line.strip()[-1] not in string.punctuation):
                print("Title name in line AFTER chapter header: {}".format(
                    line.strip()))
                cur_chapter['chapter_title'] += line.strip()
            else:  # End of line punctuation, assume no title for chapter
                print("No extra title name found for chapter: {}".format(
                    cur_chapter['chapter_num']))
                cur_chapter['chapter_text'] += line

        elif cur_chapter['chapter_num'] != None:  # Append or skip text
            if not line.strip():
                continue
            else:
                cur_chapter['chapter_text'] += line

    if (cur_chapter['chapter_num'] != None
            or cur_chapter['book_num'] != None):
        raise ValueError("Did not detect end of book")

    return chapters


def text_to_int(textnum, numwords={}):
    '''Convert word representation of number to integer.
    Borrowed from https://stackoverflow.com/questions/493174/is-there-a-way-to-convert-number-words-to-integers'''
    if not numwords:
        units = [
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty",
                "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):
            numwords[word] = (1, idx)
        for idx, word in enumerate(tens):
            numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales):
            numwords[word] = (10 ** (idx * 3 or 2), 0)

    textnum = textnum.lower().replace("-", " ")
    current = result = 0
    for word in textnum.split():
        if word not in numwords:
            raise Exception("Illegal word: " + word)

        scale, increment = numwords[word]
        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current


def roman_to_int(numeral):
    '''Convert a Roman numeral to an integer.
    Borrowed from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch03s24.html'''
    if not isinstance(numeral, type("")):
        raise TypeError("expected string, got %s" % type(numeral))
    numeral = numeral.upper()
    numeral_map = {'M': 1000, 'D': 500, 'C': 100,
                   'L': 50, 'X': 10, 'V': 5, 'I': 1}
    converted_num = 0
    for i in range(len(numeral)):
        try:
            value = numeral_map[numeral[i]]
            # If the next place holds a larger number, this value is negative
            if i + 1 < len(numeral) and numeral_map[numeral[i + 1]] > value:
                converted_num -= value
            else:
                converted_num += value
        except KeyError:
            raise ValueError(
                'input is not a valid Roman numeral: %s' % numeral)
    return converted_num


def normalize_text(text):
    text = re.sub(r"[‘’]", r"'", text)
    text = re.sub(r"[“”]", r'"', text)
    text = re.sub(r'[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]',
                  '-', text)
    text = re.sub(r'[^\x00-\x7F]', r'', text)
    return text


def resolve_unmatched_titles(src_titles, tgt_titles):

    resolved_src_titles = {}
    for src_title in src_titles:
        if src_title not in tgt_titles:
            for tgt_title in tgt_titles:
                match = title_match(src_title, tgt_title)
                if match == tgt_title:
                    resolved_src_titles[src_title] = tgt_title

    resolved_tgt_titles = {}
    for tgt_title in tgt_titles:
        if tgt_title not in src_titles:
            for src_title in src_titles:
                match = title_match(src_title, tgt_title)
                if match == src_title:
                    resolved_tgt_titles[tgt_title] = src_title

    print("SOURCE TITLE RESOLUTIONS:", resolved_src_titles)
    print("TARGET TITLE RESOLUTIONS:", resolved_tgt_titles)

    return resolved_src_titles, resolved_tgt_titles


def convert_title_to_string(title):

    title_string = ""

    (book_num, book_title,
     chapter_num, chapter_title) = title

    if book_num != None:
        title_string += "Book {} - ".format(book_num)
        if book_title:
            title_string += "{} | ".format(book_title)

    if chapter_num > 0:
        title_string += "Chapter {}".format(chapter_num)

    if chapter_title:
        if chapter_num == 0:
            title_string += "{}".format(chapter_title)
        else:
            title_string += ": {}".format(chapter_title)

    return title_string


def title_match(title1, title2):

    if title1 == title2:
        return title1

    (book_num1, book_title1,
     chapter_num1, chapter_title1) = title1

    (book_num2, book_title2,
     chapter_num2, chapter_title2) = title2

    if chapter_num1 == chapter_num2:
        if book_num1 != book_num2:
            return None

        if chapter_title1 and chapter_title2:
            # Favor titles that are not all caps
            if chapter_title1.isupper() and not chapter_title2.isupper():
                return title2
            elif chapter_title2.isupper() and not chapter_title1.isupper():
                return title1

        return title1 if len(chapter_title1) >= len(chapter_title2) else title2

    return None


def process(in_dir, out_dir):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with open(os.path.join(in_dir, "meta_data.json")) as f:
        meta_data = json.load(f)

    src_in_dir = os.path.join(in_dir, 'original')
    tgt_in_dir = os.path.join(in_dir, 'abridged')

    for book_filename in meta_data:
        chapters = {}

        if not os.path.isfile(os.path.join(src_in_dir, book_filename)):
            print("WARNING: book with filename {} not found in {}. Skipping...".format(
                book_filename, src_in_dir))
            continue

        with open(os.path.join(src_in_dir, book_filename)) as f:

            # (Some?) source texts have mid-sentence single line breaks and
            # double line breaks for paragraph boundaries.
            # Fix this to remove undesired line breaks and
            # normalize paragraph boundaries to have only one line break
            # This problem doesn't seem to appear in the abridged texts.
            text = normalize_text(f.read())
            text = re.sub('[\n]{3,}', "\n\n", text).split("\n\n")
            # Double line breaks indicate paragraphs, and single line breaks are erroneous
            # lines = [paragraph.replace("\n", " ") + "\n" for paragraph in text]
            lines = [re.sub("[\n]+", " ", paragraph) +
                     "\n" for paragraph in text]
            print("SPLITTING CHAPTERS FOR ORIGINAL BOOK {}...".format(book_filename))
            src_chapters = split_chapters(lines)

        with open(os.path.join(tgt_in_dir, book_filename)) as f:
            lines = [normalize_text(line) for line in f.readlines()]
            print("SPLITTING CHAPTERS FOR ABRIDGED BOOK {}...".format(book_filename))
            tgt_chapters = split_chapters(lines)

        (resolved_src_titles,
         resolved_tgt_titles) = resolve_unmatched_titles(src_chapters.keys(),
                                                         tgt_chapters.keys())

        src_chapters = {resolved_src_titles[title] if title in resolved_src_titles else title:
                        chapters for title, chapters in src_chapters.items()}
        src_chapters = {convert_title_to_string(title): chapter
                        for title, chapter in src_chapters.items()}
        tgt_chapters = {resolved_tgt_titles[title] if title in resolved_tgt_titles else title:
                        chapters for title, chapters in tgt_chapters.items()}
        tgt_chapters = {convert_title_to_string(title): chapter
                        for title, chapter in tgt_chapters.items()}

        assert src_chapters.keys() == tgt_chapters.keys(),\
            "Error: Mismatched chapter titles between source and target.\
            lonely source items: {}, lonely target items: {}".format(src_chapters.keys() - tgt_chapters.keys(),
                                                                     tgt_chapters.keys() - src_chapters.keys())

        if not os.path.exists(os.path.join(out_dir, 'original')):
            os.mkdir(os.path.join(out_dir, 'original'))
        src_chapters_dir = os.path.join(
            out_dir,
            'original',
            book_filename.split(".")[0]
        )
        if not os.path.exists(src_chapters_dir):
            os.mkdir(src_chapters_dir)

        if not os.path.exists(os.path.join(out_dir, 'abridged')):
            os.mkdir(os.path.join(out_dir, 'abridged'))
        tgt_chapters_dir = os.path.join(
            out_dir,
            'abridged',
            book_filename.split(".")[0]
        )
        if not os.path.exists(tgt_chapters_dir):
            os.mkdir(tgt_chapters_dir)

        meta_data[book_filename]['chapter_titles'] = []
        meta_data[book_filename]['chapter_idxs'] = []

        for chapter_idx, chapter_title in enumerate(src_chapters):
            meta_data[book_filename]['chapter_titles'].append(chapter_title)
            meta_data[book_filename]['chapter_idxs'].append(chapter_idx)

            src_chapter = src_chapters[chapter_title]

            src_chapter_fp = os.path.join(src_chapters_dir,
                                          "{}.txt".format(chapter_idx))
            with open(src_chapter_fp, 'w') as f:
                f.write(src_chapter.rstrip())

            print("Saved original version of {} {} to {}".format(
                book_filename.split(".")[0],
                chapter_title,
                src_chapter_fp))

            tgt_chapter = tgt_chapters[chapter_title]
            tgt_chapter_fp = os.path.join(tgt_chapters_dir,
                                          "{}.txt".format(chapter_idx))
            with open(tgt_chapter_fp, 'w') as f:
                f.write(tgt_chapter.rstrip())
            print("Saved abridged version of {} {} to {}".format(
                book_filename.split(".")[0],
                chapter_title,
                tgt_chapter_fp))

    meta_data = {book_filename.split(".")[0]: book_info
                 for book_filename, book_info in meta_data.items()}

    with open(os.path.join(out_dir, "meta_data.json"), 'w') as f:
        json.dump(meta_data, f, indent=4)
        print("Saved meta data file to", os.path.join(out_dir, "meta_data.json"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split the raw .txt downloads for the original and abridged books into chapters, which can further be processed by align.py\
                                     The resulting original and abridged data directories have subdirectory for each book, and each file inside a subdirectory corresponds to a chapter.\
                                     The rules for parsing chapters are not fully consistent across books so there is some book-specific logic here.\
                                     There's also some funkiness with line breaks in the original texts that causes line breaks to appear in the middle of sentences, so that is handled.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--in_dir", "-in_dir", help="Directory path where downloaded books are located\
                        (contains folders labeled 'original' and 'abridged' as well as meta_data.csv with titles and author names",
                        type=str, required=True)
    parser.add_argument("--out_dir", "-out_dir", help="Directory path where chapters (both original and abridged) will be saved,\
                        along with meta_data.json description.",
                        type=str, required=True)

    args = parser.parse_args()

    process(args.in_dir, args.out_dir)
