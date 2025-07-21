I’m building a GPU-accelerated active learning pipeline for Arabic text classification using the following technologies:

Data Context:

* I have labeled and preprocessed Arabic text stored in `.csv` files.
* The dataset contains a "text" column and a "label" column.
* The task is binary classification: security vs non-security.

Project Goal:
Create two active learning pipelines for this classification task:

1. Uncertainty sampling pipeline using SBERT embeddings and a classifier.
2. Committee-based sampling pipeline using SBERT embeddings and multiple classifiers.

I want both pipelines implemented in a clean, modular way using Jupyter Notebook, each in a separate cell.

Environment:

* I am using Python 3.11 in a Conda environment called `al_env`.
* My environment includes:

  * PyTorch 2.6+ with CUDA 12.6 or 12.8
  * sentence-transformers
  * scikit-learn
  * GPU-enabled lightgbm

Technical Requirements:

*you can use small-text or you can do it from scratch
* Use SBERT (SentenceTransformer from sentence-transformers) as the embedding model.
* For classifiers:

  * Use LogisticRegression, SVM, and GPU-enabled LightGBM.
* Implement two complete pipelines:

  1. Uncertainty Sampling:

     * SBERT embeddings
     * Any single classifier (e.g., LogisticRegression or SVM)
     * Use uncertainty sampling based on least confidence or margin sampling
  2. Committee-Based Sampling:

     * SBERT embeddings
     * Use a committee of at least 3 classifiers (e.g., SVM, LogisticRegression, LightGBM)
     * Use disagreement-based strategy such as vote entropy

Each pipeline should include:

* Loading and preprocessing the data
* Generating SBERT embeddings
* Initializing the labeled and unlabeled pools
* Running 1-2 active learning iterations
* Selecting new samples for labeling
* Returning sampled instances (texts and their predicted probabilities or disagreement scores)

The final result should be implemented in a clean python directory, and consider pipelines in the directory structure. It should be fully runnable using the specified setup and compatible with GPU acceleration.