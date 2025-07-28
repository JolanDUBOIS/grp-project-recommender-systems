# News Recommendation System (MIND Dataset)

This project explores different recommender system approaches using Microsoft's [MIND dataset](https://msnews.github.io/). The system is structured around a clean data pipeline, multiple recommendation models, and evaluation workflows.

It was developed as a two-month group assignment for the **Recommender Systems** course at **NTNU** (Norwegian University of Science and Technology).  
The goal was to implement, evaluate, and compare recommender systems through modular and extensible Python code.

## Dataset

This project uses the [MIND dataset](https://msnews.github.io/), which includes:
- **User behavior logs**: Clicks, timestamps, impressions
- **News metadata**: Titles, abstracts, categories
- **Knowledge embeddings**: Entity and relation vectors

To run the project, set the dataset path via an environment variable:

```bash
export MIND_DATA_FOLDER_PATH="path/to/mind/data"
```

Or set it in a `.env` file and run `source .env`.

## Project Structure

The code is organized into three main components:

### 1. `data_normalization/`
Prepares raw MIND data for modeling:
- Parses and normalizes `impressions.tsv`, `behaviors.tsv`, and `news.tsv`
- Structures user/news interactions for downstream processing
- Loads optional embeddings for content-based models
- Saves outputs for reuse in `data_normalized/` (if enabled)

This step is required before any model training or evaluation.

→ See [`data_normalization/README.md`](./src/data_normalization/README.md) for full documentation.

---

### 2. `recommender_systems/`
Implements multiple models, all following a shared interface:
- **BaselineMostPopular** — recommends globally popular items within a recent time window
- **ItemItemCollaborativeFiltering**
- **ALSMatrixFactorization**
- **ContentBasedFiltering** — leverages news metadata and embeddings
- **TrueHybrid** — combines collaborative and content-based predictions

Models are evaluated uniformly and can be swapped easily.

---

### 3. `evaluation/`
Handles evaluation logic:
- **Sliding window** evaluation using temporal splits
- **Validation set** workflows
- Computes **multiple metrics** (Precision, Recall, MRR, Accuracy, Coverage, Diversity)

Evaluation outputs are logged cleanly and stored for comparison.

## Running the Code

Run from the project root using the CLI:

```bash
python -m src -dn           # Normalize training data
python -m src -dn -val      # Normalize validation data
python -m src -t            # Run evaluations or experiments
```

**CLI arguments:**
- `--data-normalization` (`-dn`): Run the data processing pipeline
- `--validation` (`-val`): Apply to validation set
- `--test` (`-t`): Launch testing/evaluation workflow

All logic is defined in [`src/__main__.py`](./src/__main__.py).

## Evaluation Metrics

Models are assessed using both static and time-aware strategies. Computed metrics include:

- Precision@10
- Recall@10
- Mean Reciprocal Rank (MRR)
- Accuracy
- Coverage
- Diversity

Results are logged using Python's `logging` module for clarity and reproducibility.

## Project Status

This project is **academic and exploratory**. The models are functional but not optimized for production deployment. Some components are still experimental or simplified.

That said, the structure is clean and extensible for testing new approaches or integrating external models.

## Acknowledgments

This work was done as part of the “**Recommender Systems**” course at **NTNU**, over a 2-month period.

## License

This code is for educational purposes only. No formal license is attached.