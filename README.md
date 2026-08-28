# Predicting Sentiment From Hotel Reviews

A natural-language-processing pipeline that classifies hotel review text into
**five star-rating classes** using a **Bidirectional LSTM** trained on
**pre-trained Word2Vec (GoogleNews) embeddings**.

The project is a Python / TensorFlow deep-learning exercise. The current
default branch, `dev-migration-tf2`, contains a port of the original
standalone-Keras (TF 1.x) implementation to **TensorFlow 2.18 / Keras 3**
while preserving the original model behaviour and results. See
[MIGRATION.md](MIGRATION.md) for the full porting checklist.

---

## 1. Project goal

Given the free-text `review_text` of a hotel guest review, predict the
guest's star rating (one of five classes, encoded `0`–`4`).

- **Input:** raw review text
- **Output:** predicted rating class (`0`–`4`) and a 5-way probability vector
- **Model:** `Embedding (Word2Vec, trainable) → Bidirectional LSTM →
  GlobalMaxPool → Dense(256, ReLU) → BatchNorm → Dropout → Dense(5, softmax)`

---

## 2. Features

- Pre-trained **Word2Vec (GoogleNews, 300-dim)** embeddings loaded into a
  trainable embedding layer.
- **Bidirectional LSTM** (512 units) with dropout for sequence modelling.
- **GlobalMaxPooling1D** to summarise the sequence before classification.
- **Regularisation** via L2 weight decay, batch normalisation, and dropout.
- **Log-smoothed class weights** (`create_class_weight`) to handle the
  imbalanced star distribution.
- **PAD / EOS / UNK** token handling:
    - `PAD` → `0`
    - `UNK` (out-of-vocabulary / infrequent words) mapped to a fixed index
    - `EOS` inserted at the first padding position of each sequence
- **Parallelised text pre-processing** across CPU cores
    (`parallelproc.applyParallel`).
- **Training callbacks:** `EarlyStopping`, `ModelCheckpoint`, and
    `ReduceLROnPlateau`.
- **Evaluation** with confusion matrix + per-class accuracy plots
    (`sklearn.metrics`), saved to PNG.
- **Migration test suite** (`test_migration.py`) verifying TF 2.18 / Keras 3
  compatibility.

---

## 3. Repository layout

```
.
├── hotel_review_sentiment_classifier_lstm.py     # Main training / evaluation pipeline
├── Hotel Review Sentiment Classifier with LSTM.ipynb    # Notebook version of the workflow
├── test_migration.py                            # TF 2.18 / Keras 3 compatibility tests
│
├── utils.py                                     # Text cleaning / tokenisation helpers
├── wordnetutils.py                              # WordNet / stopword lists
├── global_settings.py                           # Project-wide constants & shared state
├── parallelproc.py                              # Parallel DataFrame apply utility
├── LossLearningRateScheduler.py                 # Loss-driven learning-rate scheduler
│
├── MIGRATION.md                                 # TF 1.x → TF 2.18 / Keras 3 port checklist
├── CLAUDE.md                                    # Migration project notes
│
├── data/
│     ├── Train/hotel_sentiment_v01.xlsx          # Training dataset (Excel)
│     └── Test/Other_Hotels_Test.xlsx             # Out-of-sample test dataset (Excel)
│
├── SplitPairs.csv                               # Word-splitting pairs for restructured text
├── AdditionalStopwords.csv                      # Extra stopword list
├── AllowedStopwords.csv                         # "Allowed" stopwords (excluded from removal)
└── AllowedNumbers.csv                           # Numeric tokens kept during cleaning
```

**Not tracked by Git** (generated artifacts / large binaries — see `.gitignore`):
`best_model.keras`, `hotel-sentiment-model.keras`, `training_history.png`,
`confusion_matrix.png`, `GoogleNews-vectors-negative300.bin`,
`__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`.

---

## 4. Requirements

- **Python** ≥ 3.10
- **TensorFlow** `>= 2.18`
- **Keras** `>= 3`
- **NumPy** `>= 2.0`
- `pandas`, `scikit-learn`, `matplotlib`, `gensim`, `openpyxl`
- **Word2Vec GoogleNews** binary vectors (300-dim) at the repository root:
  `GoogleNews-vectors-negative300.bin`

> Note: the `numpy >= 2.0` / TensorFlow 2.18 / Keras 3 combination may produce
> a harmless `tf.compat.v1` deprecation warning (e.g.
> `tf.compat.v1.train.add_to_collection`). It does not affect training.

---

## 5. Installation

```bash
# 1. Create / activate the environment
conda create -n tfrl-latest-metal python=3.12
conda activate tfrl-latest-metal

# 2. Install dependencies
pip install "tensorflow>=2.18" "keras>=3" "numpy>=2.0" pandas scikit-learn matplotlib gensim openpyxl

# 3. Download the Word2Vec vectors to the repo root (one-time)
#    https://nlp.stanford.edu/projects/wordvect/     →  GoogleNews-vectors-negative300
```

---

## 6. Usage

Run the full train → evaluate pipeline from the repository root:

```bash
python hotel_review_sentiment_classifier_lstm.py
```

The pipeline executes the following stages in order:

1. **Load & clean** the training spreadsheet
    (`data/Train/hotel_sentiment_v01.xlsx`) and pre-process the text.
2. **Tokenise & encode** into fixed-length sequences
    (length `400`, `PAD`/`EOS`/`UNK` mapping).
3. **Load Word2Vec** embeddings from `GoogleNews-vectors-negative300.bin`.
4. **Build & compile** the Bidirectional LSTM model
    (categorical cross-entropy, Adam).
5. **Train** with early stopping, checkpointing, and LR reduction
    → writes `best_model.keras`.
6. **Plot** training history → `training_history.png`.
7. **Evaluate** on `data/Test/Other_Hotels_Test.xlsx`
    → `confusion_matrix.png` and a per-class accuracy table.

Key hyper-parameters (see `global_settings.py` / the script header):

| Parameter            | Value |
|--------------------|-------|
| `MAX_SEQUENCE_LENGTH`| 400   |
| `EMBEDDING_DIM`     | 300   |
| `LSTM units`        | 512   |
| `BATCH_SIZE`        | 64    |
| `EPOCHS`            | 20    |
| `VALID_RATIO`       | 0.2   |
| Classes             | 5 (`0`–`4`) |

Run the migration compatibility tests separately:

```bash
python test_migration.py
```

---

## 7. Outputs

| File                          | Description                            |
|------------------------------|----------------------------------------|
| `best_model.keras`            | Best saved model (`ModelCheckpoint`)   |
| `hotel-sentiment-model.keras` | Full saved model                       |
| `training_history.png`        | Train/val accuracy & loss curves       |
| `confusion_matrix.png`        | Confusion matrix + per-class accuracy  |

All of these are **git-ignored** — they are generated on each run.

---

## 8. Migration status (TF 2.18 / Keras 3)

The `dev-migration-tf2` branch ports the original standalone-Keras code to
TF 2.18 / Keras 3. The full status is tracked in
[MIGRATION.md](MIGRATION.md); highlights:

- `import tensorflow as tf`, `from tensorflow import keras`
- Embedding built from a NumPy matrix (`weights=[...]`, `input_length=...`)
- Optimizer moved to `model.compile()` via `keras.optimizers.Adam(...)`
- Callbacks from `tensorflow.keras.callbacks`; checkpoint format `.keras`
- `on_epoch_begin(epoch, logs=None)` for the custom LR scheduler

Run `python test_migration.py` to confirm the environment passes the checks.

---

## 9. Notes

- Pre-processing, tokenisation, and data handling live in `utils.py`,
  `wordnetutils.py`, and `global_settings.py`; shared constants (Word2Vec
  model path, validation split, random seed, `cpu`, etc.) are initialised in
  `global_settings.py`.
- This is an educational / research project; it has no declared open-source
  licence. Do not redistribute without permission from the author.