This directory contains code used to create the dataset and conduct the experiments and analyses presented in the paper.

## Creating the aligned dataset

Below are the steps we took for automatically producing aligned rows of original and abridged spans.

### Install the library dependencies

```pip install -r requirements.txt```

### Download the original and abridged book chapters [here](https://drive.google.com/file/d/1AV_DQfMvTw0TCpcSHWdc0riP6WiPOaKS/view?usp=sharing)

### Split the chapters into sentence segments
```
python dataset_creation/split_segments.py -in_dir chapters/ -out_dir sentences/ -meta_data_file chapters/meta_data.json
```

### Run the alignment algorithm on the segmented data

##### Alternative 1 (our approach):

```
python dataset_creation/align.py -scoring_method string -segments_in_dir sentences/ -meta_data_file chapters/meta_data.json -out_dir rows/ -size_penalty 0.175
```

This will use the ROUGE (string overlap) scoring method for alignment, which is the method that obtained the best result as reported in the paper. See the paper for a description of the size_penalty parameter.

align.py creates two directories in out_dir/: "json" contains the .json version of the aligned rows and "sheets" are .csv versions of the rows that we uploaded to Google Sheets for the human validation procedure.

##### Alternative 2 (lower alignment accuracy than Alternative 1 for this dataset):

You can also apply a vector-based scoring method, which applies a vector embedding model (e.g. BERT) to encode segments, then aligns them based on vector similarity:

First transform the segments into vectors (you can use any HuggingFace model that produces embeddings, e.g. bert-based-uncased):

```
python dataset_creation/transform.py -in_dir sentences/ -out_dir vectors/ -meta_data_file chapters/meta_data.json -vectorizer bert-base-uncased
```

Then run alignment:
```
python dataset_creation/align.py -scoring_method vector -segments_in_dir sentences/ -vectors_in_dir vectors/ -meta_data_file chapters/meta_data.json -out_dir rows/ -size_penalty 0.21
```

### Validation

The paper describes the procedure we used for estimating the alignment accuracy and engaging human effort to correct erroneously aligned rows. Annotators used Google Sheets to edit the spreadsheet (.csv) files produced by the align.py script. The resulting files were then downloaded and converted back into .json files via the process_validation_sheets.py script.

##### Evaluating the alignments

We evaluated the accuracy of the automatically produced aligned rows by comparing them with human-validated rows. Given predicted and validated (gold) rows in the same format as rows/json, you can run the alignment accuracy metric described in the paper:

```python dataset_creation/evaluate_alignment.py -pred_alignment_dir rows/json/ -gold_alignment_dir [a directory with files corresponding to rows/json/ that have been human-validated] -meta_data_file chapters/meta_data.json```

### Package the dataset

Finally, we converted the dataset to a format readable by the ablit/ package. Assuming that "rows/json" contains the alignments you want in the dataset, you can run:

```
python dataset_creation/pack_final_dataset.py -in_dir rows/json -out_dir dataset/ -meta_data_file chapters/meta_data.json -assign_partition
```

The -assign_partition parameter creates the train/dev/test split for the data.

### Get dataset stats

This computes the description statistics for the dataset reported in the paper (e.g. number of passages, passage alignment scores, lexical analysis)

```
python dataset_creation/dataset_stats.py -ablit_dir dataset/ -output_file dataset/dataset_stats.json
```

## Experiments

We used the script [here](http://github.com/huggingface/transformers/blob/master/examples/pytorch/summarization/run_summarization.py) for fine-tuning the sequence-to-sequence models (T5 and BART) on the dataset. The best-performing of these fine-tuned models (BART) is available as **ablit-bart-base** on [HuggingFace](https://huggingface.co/LanguageWeaver/ablit-bart-base). 


