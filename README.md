## My progress and workflow:
- I have worked further with the hybrid model. Uses TF-IDF (cosine similarity) and SVD; this doesnt capture semantics (for instance, synonyms). For further developement, I plan to try Word Embeddings.
- Started on a baseline for the content-based recommender. I plan to use Sentence-BERT (word embedding) for this; it captures semantics, and has pre-trained models.
- I need to do some error handling (windows) to speed up the programming. Normalization script has been hard-coded.


### Discussed:
- Try regression.
- Sentence_BERT: should maybe not use pre-trained models, check with prof potentially.
- For hybrid, make sure TF-IDF + cosine similarity is an actual hybrid model (check presentations).
- Word embedding up for testing, may not be necessary.
- Train model on which articles have been clicked - furthermore, recommend articles clicked by similar users. We are recommending articles, and it's being tested on which articles the user actually clicked.

### Files:

hybrid_notebook.ipynb
    This is the one I am primarily using to develop the hybrid model, so this is the file with the most progress and comments. I have written down comments, explainations and runtimes in markdowns, to make it more understandable than the .py file. Uses TF-IDF (cosine) and SVD.

hybrid.py
    I will transfer all the code into this one, as this will be the final product.

content_based.py
    I have started to code the content-based in this, but I will be making a notebook to develop it further - a personal preference. I will be testing word embeddings, to verify if this will be more accurate than cosine.