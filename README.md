# Hybrid Gusum

This is an implementation of the [GUSUM](https://aclanthology.org/2022.textgraphs-1.5
) ( as shorthand for Graph-Based Unsupervised Summarization) combination with a model-based summarizatiob.

## Installation

You need to install python3 and following libraries

```
pip install sentence-transformers
pip install nltk
pip install numpy
pip install py-rouge
pip install transformers
pip install datasets

```
Otherwise install in a Conda environment with following command:
```bash
conda env create -f environment.yml
```
## Results of Evaluation 
The evaluation is done by this [notebook](https://colab.research.google.com/drive/1w2uCvvzJSvck_fuMaoyPnEOBqxnN78ID?usp=drive_link) with GPU T4.   
For training data, it took 10 hours and for test data around 1.5 hours.
The final result in work is a combination of both training and test data.

## Data used in the paper

DialogSum, you can download it from : https://huggingface.co/datasets/knkarthick/dialogsum

## Files Structure

The files are organized as follows:

```
.
├── README.md
├── environment.yml
├── evaluator
│   ├── evaluator_test_all.pkl
│   └── evaluator_train_all.pkl
├── evaluator.py
├── graph.py
├── main_dialogsum.py
├── sentenceRanker.py
├── summarizer.py
└── utils.py

```
## Evaluation

Automated evaluation used the py-rouge package. https://pypi.org/project/py-rouge/ Our evaluation metrics are set as shown below.

```
evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                           max_n=4,
                           limit_length=False,
                           length_limit=1000,
                           length_limit_type='words',
                           apply_avg=apply_avg,
                           apply_best=apply_best,
                           alpha=0.2, 
                           weight_factor=1.2,
                           stemming=True)
                           
 ```

## Citation
```
@InProceedings{gokhan-smith-lee:2022:textgraphs,
  author    = {Gokhan, Tuba  and  Smith, Phillip  and  Lee, Mark},
  title     = {GUSUM: Graph-based Unsupervised Summarization Using Sentence Features Scoring and Sentence-BERT},
  booktitle      = {Proceedings of TextGraphs-16: Graph-based Methods for Natural Language Processing},
  month          = {October},
  year           = {2022},
  address        = {Gyeongju, Republic of Korea},
  publisher      = {Association for Computational Linguistics},
  pages     = {44--53},
  abstract  = {Unsupervised extractive document summarization aims to extract salient sentences from a document without requiring a labelled corpus. In existing graph-based methods, vertex and edge weights are usually created by calculating sentence similarities. In this paper, we develop a Graph-Based Unsupervised Summarization(GUSUM) method for extractive text summarization based on the principle of including the most important sentences while excluding sentences with similar meanings in the summary. We modify traditional graph ranking algorithms with recent sentence embedding models and sentence features and modify how sentence centrality is computed. We first define the sentence feature scores represented at the vertices, indicating the importance of each sentence in the document. After this stage, we use Sentence-BERT for obtaining sentence embeddings to better capture the sentence meaning. In this way, we define the edges of a graph where semantic similarities are represented. Next we create an undirected graph that includes sentence significance and similarities between sentences. In the last stage, we determine the most important sentences in the document with the ranking method we suggested on the graph created. Experiments on CNN/Daily Mail, New York Times, arXiv, and PubMed datasets show our approach achieves high performance on unsupervised graph-based summarization when evaluated both automatically and by humans.},
  url       = {https://aclanthology.org/2022.textgraphs-1.5}
}

```


