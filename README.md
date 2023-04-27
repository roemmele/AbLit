# AbLit: *Ab*ridged versions of English *Lit*erature books


This repo contains the data and code featured in the paper [AbLit: A Resource for Analyzing and Generating Abridged Versions of English Literature](https://arxiv.org/pdf/2302.06579.pdf). The author Emma Laybourn wrote [abridged versions of 10 classic books](http://www.englishliteratureebooks.com/classicnovelsabridged.html). AbLit annotates the alignment between passages in these abridgements with their original version.

#### The dataset is now available on HuggingFace datasets: [huggingface.co/datasets/roemmele/ablit](https://huggingface.co/datasets/roemmele/ablit). (There is also an abridgement generation model available on the [Hub](https://huggingface.co/roemmele/ablit-bart-base), which is initialized from bart-base and fine-tuned on the AbLit dataset, as described in the paper.)

#### Alternatively, you can access the data directly from this repo. Because AbLit can provide aligned passages of different lengths (e.g. sentences, paragraphs, multi-paragraph chunks), we've provided a python package for interfacing with the data (called ablit). You can use this code to extract and save the data for different passage sizes to file.

The raw data is in the dataset/ folder.

To install ablit as a package and load it from anywhere, do "pip install ." in the top level of the repo.


```python
from ablit import AblitDataset
```


```python
'''Specify loading a specific partition of the data (train, dev, or test).
If no partition given (partition=None), the full dataset will be loaded.'''

dataset = AblitDataset(dirpath='./dataset', partition='dev')
dataset
```




    AblitDataset(
            dirpath=./dataset,
            partition=dev,
            books=[{'bleak-house': {'book_title': 'Bleak House', 'author': 'Charles Dickens'}}, {'can-you-forgive-her': {'book_title': 'Can You Forgive Her?', 'author': 'Anthony Trollope'}}, {'daniel-deronda': {'book_title': 'Daniel Deronda', 'author': 'George Eliot'}}, {'mansfield-park': {'book_title': 'Mansfield Park', 'author': 'Jane Austen'}}, {'north-and-south': {'book_title': 'North and South', 'author': 'Elizabeth Gaskell'}}, {'shirley': {'book_title': 'Shirley', 'author': 'Charlotte Bronte'}}, {'the-way-we-live-now': {'book_title': 'The Way We Live Now', 'author': 'Anthony Trollope'}}, {'tristram-shandy': {'book_title': 'Tristram Shandy', 'author': 'Laurence Sterne'}}, {'vanity-fair': {'book_title': 'Vanity Fair', 'author': 'W. M. Thackeray'}}, {'wuthering-heights': {'book_title': 'Wuthering Heights', 'author': 'Emily Bronte'}}],
            )




```python
dataset.books[0]
```




    Book(
            dirpath=./dataset/bleak-house,
            id=bleak-house,
            book_title=Bleak House,
            author=Charles Dickens,
            chapters={0: 'Preface'}
            )




```python
'''You can also retrieve a book by its id.'''

book = dataset.get_book_by_id("wuthering-heights")
book
```




    Book(
            dirpath=./dataset/wuthering-heights,
            id=wuthering-heights,
            book_title=Wuthering Heights,
            author=Emily Bronte,
            chapters={0: 'Chapter 1'}
            )




```python
chapters = list(book.chapters)
chapter = chapters[0]
chapter
```




    Chapter(
            book_id=wuthering-heights,
            book_title=Wuthering Heights,
            chapter_idx=0,
            chapter_title=Chapter 1,
            original="1801-I have just returned from a visit to my landlord-the solitary neighbour that I shall be troubled with. This is certainly a beautiful country! In all England, I do not believe that I could have fixed on a situation so completely removed from the stir of society. A perfect misanthropist's Heaven-and Mr. Heathcliff and I are such a suitable pair to divide the desolation between us. A capital fellow! He little imagined how my heart warmed towards him when I beheld his black eyes withdraw so suspiciously under their brows, as I rode up, and when his fingers sheltered themselves, with a jealous resolution, still further in his waistcoat, as I announced my name.
    "Mr. Heathcliff?" I said.
    A nod was the answer.
    "Mr. Lockwood, your new tenant, sir. I do myself the honour of calling as soon as possible after my arrival, to express the hope that I have not inconvenienced you by my perseverance in soliciting the occupation of Thrushcross Grange: I heard yesterday you had had some thoughts-"
    "T..."
            abridged="1801. I have just returned from a visit to my landlord - my solitary neighbour. This is certainly a beautiful country! In all England, I do not believe that I could have found a place so completely removed from society. A perfect misanthropist's heaven: for which Mr. Heathcliff and I are equally suited. A capital fellow! He little imagined how my heart warmed towards him when I saw his black eyes withdraw so suspiciously under their brows, and his fingers shelter themselves jealously in his waistcoat, as I rode up.
    'Mr. Heathcliff?' I said.
    He nodded.
    'Mr. Lockwood, your new tenant, sir. I called to express the hope that I have not inconvenienced you by renting Thrushcross Grange.'
    'Thrushcross Grange is my own, sir,' he answered, wincing. 'I should not allow anyone to inconvenience me, if I could hinder it. Walk in!'
    The 'walk in' was uttered with closed teeth, meaning, 'Go to the Devil.' However, I decided to accept the invitation: I felt interested in a man even more exaggeratedly r..."
            )




```python
'''All text within a chapter (paragraphs, chunks, sentences) is represented with 
the Passage superclass, which specifies character indices corresponding to the chapter text, 
and the abridged text that is aligned with it'''

paragraphs = list(chapter.paragraphs)
paragraphs[0]
```




    Paragraph(
            idx=0,
            n_original_sentences=6,
            original_start_char=0,
            original_end_char=669,
            original="1801-I have just returned from a visit to my landlord-the solitary neighbour that I shall be troubled with. This is certainly a beautiful country! In all England, I do not believe that I could have fixed on a situation so completely removed from the stir of society. A perfect misanthropist's Heaven-and Mr. Heathcliff and I are such a suitable pair to divide the desolation between us. A capital fellow! He little imagined how my heart warmed towards him when I beheld his black eyes withdraw so suspiciously under their brows, as I rode up, and when his fingers sheltered themselves, with a jealous resolution, still further in his waistcoat, as I announced my name.
    ",
            abridged_start_char=0,
            abridged_end_char=521,
            abridged="1801. I have just returned from a visit to my landlord - my solitary neighbour. This is certainly a beautiful country! In all England, I do not believe that I could have found a place so completely removed from society. A perfect misanthropist's heaven: for which Mr. Heathcliff and I are equally suited. A capital fellow! He little imagined how my heart warmed towards him when I saw his black eyes withdraw so suspiciously under their brows, and his fingers shelter themselves jealously in his waistcoat, as I rode up.
    "
            )




```python
sentences = list(chapter.sentences)
sentences[0]
```




    Sentence(
            idx=0,
            original_start_char=0,
            original_end_char=108,
            original="1801-I have just returned from a visit to my landlord-the solitary neighbour that I shall be troubled with. ",
            abridged_start_char=0,
            abridged_end_char=80,
            abridged="1801. I have just returned from a visit to my landlord - my solitary neighbour. "
            )




```python
'''Get passages ("chunks") of a minimum length in terms of number of sentences (merge_n_sentences). 
A chunk consists of one or more paragraphs whose total
number of sentences is equal to or greater than merge_n_sentences. 
No splitting within paragraphs is done; paragraphs are merged only. 
This means that a chunk will exceed merge_n_sentences when a paragraph exceeds merge_n_sentences. 
Moreover, trailing sentences left over after all preceding chunks in the text have been gathered
will be added to the final chunk in the text; this may also cause the chunk to exceed
merge_n_sentences.'''

chunks = list(chapter.chunks(merge_n_sentences=9))
chunks[0]
```




    Chunk(
            idx=0,
            original_paragraph_idxs=[0, 1, 2],
            n_original_sentences=9,
            original_start_char=0,
            original_end_char=717,
            original="1801-I have just returned from a visit to my landlord-the solitary neighbour that I shall be troubled with. This is certainly a beautiful country! In all England, I do not believe that I could have fixed on a situation so completely removed from the stir of society. A perfect misanthropist's Heaven-and Mr. Heathcliff and I are such a suitable pair to divide the desolation between us. A capital fellow! He little imagined how my heart warmed towards him when I beheld his black eyes withdraw so suspiciously under their brows, as I rode up, and when his fingers sheltered themselves, with a jealous resolution, still further in his waistcoat, as I announced my name.
    "Mr. Heathcliff?" I said.
    A nod was the answer.
    ",
            abridged_start_char=0,
            abridged_end_char=558,
            abridged="1801. I have just returned from a visit to my landlord - my solitary neighbour. This is certainly a beautiful country! In all England, I do not believe that I could have found a place so completely removed from society. A perfect misanthropist's heaven: for which Mr. Heathcliff and I are equally suited. A capital fellow! He little imagined how my heart warmed towards him when I saw his black eyes withdraw so suspiciously under their brows, and his fingers shelter themselves jealously in his waistcoat, as I rode up.
    'Mr. Heathcliff?' I said.
    He nodded.
    "
            )




```python
'''Alternatively, specify max_n_sentences if you want to merge paragraphs
into chunks of max_n_sentences, and you additionally want to split within paragraphs 
so that no passage exceeds max_n_sentences. Trailing sentences within a paragraph that 
are less than max_n_sentences will be treated as their own chunk. In the paper, the results
for chunks were obtained using max_n_sentences=10.'''

chunks = list(chapter.chunks(max_n_sentences=3))
chunks[0]
```




    Chunk(
            idx=0,
            original_paragraph_idxs=[0],
            n_original_sentences=3,
            original_start_char=0,
            original_end_char=267,
            original="1801-I have just returned from a visit to my landlord-the solitary neighbour that I shall be troubled with. This is certainly a beautiful country! In all England, I do not believe that I could have fixed on a situation so completely removed from the stir of society. ",
            abridged_start_char=0,
            abridged_end_char=220,
            abridged="1801. I have just returned from a visit to my landlord - my solitary neighbour. This is certainly a beautiful country! In all England, I do not believe that I could have found a place so completely removed from society. "
            )




```python
'''Get the rows of aligned spans that are the direct output of the validation task.
These passages can be thought of as "oracle spans" because the size of the passages 
participating in the alignment are not immediately known for any arbitrary text, 
they are only known because they are defined by the result of the validation task. 
For example, if a row aligns two original sentences with one abridged sentence,
the fact that these two original sentences should be grouped together
in the same alignment isn't automatically known just by looking at the text itself. 
In contrast, the boundaries for sentences, paragraphs, and chunks 
can be inferred directly from just the original text.'''

rows = list(chapter.rows)
rows[12]
```




    Row(
            idx=12,
            n_original_sentences=1,
            original_start_char=1060,
            original_end_char=1140,
            original=""I should not allow any one to inconvenience me, if I could hinder it-walk in!"
    ",
            n_abridged_sentences=2,
            abridged_start_char=750,
            abridged_end_char=830,
            abridged="'I should not allow anyone to inconvenience me, if I could hinder it. Walk in!'
    ",
            )




```python
''' For a given original passage, get the slices within it that 
overlap with text in the corresponding abridged passage. 
This is available for all passages.
This information was used for token sequence labeling experiments reported in the paper, 
to predict which text in the original also shows up in the abridgement.'''

paragraphs[4].overlaps
```




    [Passage(
             version=original,
             start_char=1005,
             end_char=1032,
             text="cross Grange is my own, sir"
             ),
     Passage(
             version=original,
             start_char=1035,
             end_char=1037,
             text="he"
             ),
     Passage(
             version=original,
             start_char=1049,
             end_char=1060,
             text=", wincing. "
             ),
     Passage(
             version=original,
             start_char=1061,
             end_char=1079,
             text="I should not allow"
             ),
     Passage(
             version=original,
             start_char=1088,
             end_char=1129,
             text="to inconvenience me, if I could hinder it"
             ),
     Passage(
             version=original,
             start_char=1130,
             end_char=1137,
             text="walk in"
             )]



#### Extract a full dataset of aligned passages

It's easy to produce a list with the aligned original and abridged passages of whatever size you want, and save it as a file that you can load in your experiments. Just select the data partition you want to extract, then specify the attribute for your desired passage type while looping through all the books/chapters, as shown below.


```python
import pandas
pandas.options.display.max_colwidth = None
```


```python
from ablit import AblitDataset

dataset = AblitDataset(dirpath='./dataset',
                       partition="dev")
```


```python
dataframe = []
for book in dataset.books:
    for chapter in book.chapters:
        ''' Pick your passage type, e.g.:'''
#         for passage in chapter.rows:
#         for passage in chapter.paragraphs:
#         for passage in chapter.chunks(merge_n_sentences=None, max_n_sentences=10):
        for passage in chapter.sentences:
            dataframe.append({'original': passage.original,
                              'abridged': passage.abridged,
                              'book': book.book_title,
                              'chapter': chapter.chapter_title})
dataframe = pandas.DataFrame(dataframe)
dataframe
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original</th>
      <th>abridged</th>
      <th>book</th>
      <th>chapter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A Chancery judge once had the kindness to inform me, as one of a company of some hundred and fifty men and women not labouring under any suspicions of lunacy, that the Court of Chancery, though the shining subject of much popular prejudice (at which point I thought the judge's eye had a cast in my direction), was almost immaculate.</td>
      <td></td>
      <td>Bleak House</td>
      <td>Preface</td>
    </tr>
    <tr>
      <th>1</th>
      <td>There had been, he admitted, a trivial blemish or so in its rate of progress, but this was exaggerated and had been entirely owing to the "parsimony of the public," which guilty public, it appeared, had been until lately bent in the most determined manner on by no means enlarging the number of Chancery judges appointed--I believe by Richard the Second, but any other king will do as well.\n</td>
      <td></td>
      <td>Bleak House</td>
      <td>Preface</td>
    </tr>
    <tr>
      <th>2</th>
      <td>This seemed to me too profound a joke to be inserted in the body of this book or I should have restored it to Conversation Kenge or to Mr. Vholes, with one or other of whom I think it must have originated.</td>
      <td></td>
      <td>Bleak House</td>
      <td>Preface</td>
    </tr>
    <tr>
      <th>3</th>
      <td>In such mouths I might have coupled it with an apt quotation from one of Shakespeare's sonnets:\n</td>
      <td></td>
      <td>Bleak House</td>
      <td>Preface</td>
    </tr>
    <tr>
      <th>4</th>
      <td>"My nature is subdued To what it works in, like the dyer's hand: Pity me, then, and wish I were renewed!"\n</td>
      <td></td>
      <td>Bleak House</td>
      <td>Preface</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1138</th>
      <td>He-probably swayed by prudential consideration of the folly of offending a good tenant-relaxed a little in the laconic style of chipping off his pronouns and auxiliary verbs, and introduced what he supposed would be a subject of interest to me,-a discourse on the advantages and disadvantages of my present place of retirement.</td>
      <td>Probably not wishing to offend a good tenant, he began to talk less curtly, discussing the advantages and disadvantages of my new house.</td>
      <td>Wuthering Heights</td>
      <td>Chapter 1</td>
    </tr>
    <tr>
      <th>1139</th>
      <td>I found him very intelligent on the topics we touched; and before I went home, I was encouraged so far as to volunteer another visit to-morrow.</td>
      <td>I found him very intelligent on these topics; and before I went home, I offered to visit him tomorrow.</td>
      <td>Wuthering Heights</td>
      <td>Chapter 1</td>
    </tr>
    <tr>
      <th>1140</th>
      <td>He evidently wished no repetition of my intrusion.</td>
      <td>He did not seem to wish for it.</td>
      <td>Wuthering Heights</td>
      <td>Chapter 1</td>
    </tr>
    <tr>
      <th>1141</th>
      <td>I shall go, notwithstanding.</td>
      <td>I shall go, all the same.</td>
      <td>Wuthering Heights</td>
      <td>Chapter 1</td>
    </tr>
    <tr>
      <th>1142</th>
      <td>It is astonishing how sociable I feel myself compared with him.</td>
      <td>It is astonishing how sociable I feel compared with him.</td>
      <td>Wuthering Heights</td>
      <td>Chapter 1</td>
    </tr>
  </tbody>
</table>
<p>1143 rows × 4 columns</p>
</div>




```python
# Save the dataset, e.g.:
# dataframe.to_json("my_dataset.jsonl", orient='records', lines=True)
```


```python

```
