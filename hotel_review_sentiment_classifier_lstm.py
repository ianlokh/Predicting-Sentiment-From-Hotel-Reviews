#!/usr/bin/env python
"""
Hotel Review Sentiment Classifier with LSTM

Trains a Bidirectional LSTM model with pre-trained Word2Vec embeddings
to classify hotel review sentiment into 5 star-rating categories.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for script usage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tensorflow.keras.layers import (
    Dense, Embedding, LSTM, Bidirectional,
    BatchNormalization, GlobalMaxPooling1D, Dropout,
)
from tensorflow.keras.regularizers import l2
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model

try:
    from keras.src.legacy.preprocessing.text import Tokenizer
    from keras.src.legacy.preprocessing.sequence import pad_sequences
except ImportError:
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import gensim
from gensim.utils import simple_preprocess
from multiprocessing import cpu_count

import utils as utils
import global_settings as gs
from parallelproc import applyParallel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PATH = os.path.dirname(os.path.abspath(__file__))
os.chdir(PATH)

MAX_SEQUENCE_LENGTH = 300
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 20

tokenize = lambda x: simple_preprocess(x)
_number_of_groups = int(cpu_count() * 0.8)
_cpu = int(cpu_count() * 0.8)
gs.init()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def create_class_weight(labels_dict, mu=0.5):
    """Compute class weights using log-smoothing."""
    total = np.sum(list(labels_dict.values()))
    class_weight = dict()
    for key, value in labels_dict.items():
        score = math.log(mu * total / float(value))
        class_weight[key] = score if score > 1.0 else 1.0
    return class_weight


def set_eos(row, eos_index):
    """Set the first zero-padding position to the EOS token index."""
    values = list(row)
    idx = values.index(0) if 0 in values else -1
    row.iloc[idx] = eos_index
    return row


# ---------------------------------------------------------------------------
# 1. Load & clean data
# ---------------------------------------------------------------------------
def load_data():
    """Read the training spreadsheet and apply text pre-processing."""
    i = 2
    rows = pd.DataFrame()
    while i <= 2:
        rw = utils.readXlsx("./data/Train/hotel_sentiment_v01.xlsx", sheet=i, header=True)
        rows = pd.concat([rows, pd.DataFrame(rw)], ignore_index=True)
        i += 1
    df = pd.DataFrame(rows)
    del rw, rows
    df = df.dropna().drop_duplicates()
    df.insert(0, "grpId", df.index % _number_of_groups)

    print("Starting pre-processing")
    df = applyParallel(
        df.groupby(df.grpId), utils.clean_text,
        {"dest_col_ind": df.shape[1] - 1, "dest_col": "processed_text", "src_col": "review_text"},
        _cpu,
    )
    df = applyParallel(
        df.groupby(df.grpId), utils.lower_case,
        {"dest_col_ind": df.shape[1] - 1, "dest_col": "processed_text", "src_col": "processed_text"},
        _cpu,
    )
    df = applyParallel(
        df.groupby(df.grpId), utils.restructureText,
        {"dest_col_ind": df.shape[1] - 1, "dest_col": "processed_text", "src_col": "processed_text"},
        _cpu,
    )
    df = applyParallel(
        df.groupby(df.grpId), utils.remove_stopwords,
        {"dest_col_ind": df.shape[1] - 1, "dest_col": "processed_text", "src_col": "processed_text"},
        _cpu,
    )
    return df


# ---------------------------------------------------------------------------
# 2. Tokenisation & sequence encoding
# ---------------------------------------------------------------------------
def tokenise_and_encode(df):
    """Build vocabulary, tokenise reviews, and apply PAD / EOS / UNK mapping."""
    wordlist = utils.get_words_by_freq(df["processed_text"], 10)
    wordlist.extend(["PAD", "EOS", "UNK"])

    max_features = 23413
    tokenizer = Tokenizer(num_words=max_features, split=" ")
    tokenizer.fit_on_texts(df["processed_text"].values)
    X = tokenizer.texts_to_sequences(df["processed_text"].values)
    X = pd.DataFrame(pad_sequences(X, padding="post", truncating="post", maxlen=MAX_SEQUENCE_LENGTH))

    word_index = tokenizer.word_index
    word_index["PAD"] = 0
    word_index["EOS"] = len(word_index)
    word_index["UNK"] = len(word_index)

    # Map infrequent words to UNK
    temp = {v: v if k in wordlist else word_index["UNK"] for k, v in word_index.items()}
    temp[0] = word_index["PAD"]
    temp[len(word_index)] = word_index["EOS"]
    X = X.apply(lambda y: y.map(lambda x: temp.get(x, x)), axis=1)

    # Insert EOS token at first padding position
    X = X.apply(lambda y: set_eos(y, word_index["EOS"]), axis=1)

    return X, word_index, wordlist


# ---------------------------------------------------------------------------
# 3. Train / validation split & label encoding
# ---------------------------------------------------------------------------
def split_data(X, df):
    """One-hot encode labels and split into train / validation sets."""
    enc1 = OneHotEncoder()
    Y = pd.DataFrame(enc1.fit_transform(pd.DataFrame(df["score"])).toarray())
    x_train, x_valid, y_train, y_valid = train_test_split(
        X, Y, test_size=0.15, random_state=gs.seedvalue,
    )
    return x_train, x_valid, y_train, y_valid


# ---------------------------------------------------------------------------
# 4. Build embedding matrix from Word2Vec
# ---------------------------------------------------------------------------
def build_embedding_matrix(word_index):
    """Load Google News Word2Vec and construct the embedding matrix."""
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        "./GoogleNews-vectors-negative300.bin", binary=True,
    )
    vocab = word2vec_model.key_to_index

    embeddings_index = {}
    for word in word_index.keys():
        if vocab.get(word) is not None:
            embeddings_index[word] = word2vec_model.vectors[vocab.get(word)]
        else:
            embeddings_index[word] = [0] * 300

    pad_embed = np.empty(300, dtype=float); pad_embed.fill(0.001)
    eos_embed = np.empty(300, dtype=float); eos_embed.fill(0.999)
    unk_embed = np.empty(300, dtype=float); unk_embed.fill(0.555)
    embeddings_index["PAD"] = pad_embed
    embeddings_index["EOS"] = eos_embed
    embeddings_index["UNK"] = unk_embed

    embedding_dimension = 300
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector[:embedding_dimension]

    return embedding_matrix, embedding_dimension


# ---------------------------------------------------------------------------
# 5. Model definition
# ---------------------------------------------------------------------------
def build_model(embedding_matrix, embedding_dimension):
    """Construct and compile a Bidirectional LSTM classifier."""
    lstm_out = 512

    model = Sequential([
        Embedding(
            embedding_matrix.shape[0], embedding_dimension,
            weights=[embedding_matrix], trainable=True,
        ),
        Bidirectional(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.0,
                           return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(256, activation="relu", kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(5, activation="softmax"),
    ])

    adam = keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
    )
    model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
    print(model.summary())
    return model


# ---------------------------------------------------------------------------
# 6. Training
# ---------------------------------------------------------------------------
def train_model(model, x_train, y_train, x_valid, y_valid, classweights):
    """Train the model with early stopping, checkpointing, and LR reduction."""
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5,
        restore_best_weights=True, verbose=1,
    )
    checkpoint = keras.callbacks.ModelCheckpoint(
        "best_model.keras",
        monitor="val_accuracy", save_best_only=True, verbose=1,
    )
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=3,
        factor=0.5, min_lr=1e-6, verbose=1,
    )

    hist = model.fit(
        np.array(x_train), np.array(y_train),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1,
        validation_data=(np.array(x_valid), np.array(y_valid)),
        shuffle=True,
        class_weight=classweights,
        callbacks=[early_stop, checkpoint, reduce_lr],
    )
    return hist


# ---------------------------------------------------------------------------
# 7. Plotting helpers
# ---------------------------------------------------------------------------
def plot_training_history(hist):
    """Plot accuracy and loss curves and save to a file."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(hist.history["accuracy"], label="Train Acc")
    ax1.plot(hist.history["val_accuracy"], label="Val Acc")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(hist.history["loss"], label="Train Loss")
    ax2.plot(hist.history["val_loss"], label="Val Loss")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=150)
    plt.close()
    print("Saved training_history.png")


def plot_confusion_matrices(y_valid_classes, preds_classes, class_names):
    """Plot raw and normalised confusion matrices and save to a file."""
    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        HAS_SEABORN = False

    cnf_matrix = confusion_matrix(y_valid_classes, preds_classes)
    cnf_matrix_norm = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    if HAS_SEABORN:
        sns.heatmap(cnf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.subplot(1, 2, 2)
    if HAS_SEABORN:
        sns.heatmap(cnf_matrix_norm, annot=True, fmt=".2%", cmap="Greens",
                    xticklabels=class_names, yticklabels=class_names)
    else:
        plt.imshow(cnf_matrix_norm, interpolation="nearest", cmap=plt.cm.Greens)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=90)
        plt.yticks(tick_marks, class_names)
    plt.title("Normalized Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved confusion_matrix.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # 1. Load & pre-process
    df = load_data()

    # 2. Tokenise & encode
    X, word_index, wordlist = tokenise_and_encode(df)

    # 3. Train / validation split
    x_train, x_valid, y_train, y_valid = split_data(X, df)

    # 4. Embedding matrix
    embedding_matrix, embedding_dimension = build_embedding_matrix(word_index)

    # 5. Class weights
    _y_train_classes = np.argmax(np.array(y_train), axis=1)
    labels_dict = {
        int(cls): int(cnt)
        for cls, cnt in zip(*np.unique(_y_train_classes, return_counts=True))
    }
    classweights = create_class_weight(labels_dict, mu=0.5)
    print("Class weights:", classweights)

    # 6. Build & train
    model = build_model(embedding_matrix, embedding_dimension)
    hist = train_model(model, x_train, y_train, x_valid, y_valid, classweights)

    # 7. Plot training history
    plot_training_history(hist)

    # 8. Save final model
    model.save("hotel-sentiment-model.keras")
    print("Saved hotel-sentiment-model.keras")

    # 9. Evaluate
    model = load_model("hotel-sentiment-model.keras")
    preds = model.predict(np.array(x_valid), batch_size=None, verbose=1)

    preds_classes = np.argmax(preds, axis=1)
    y_valid_classes = np.argmax(np.array(y_valid), axis=1)
    class_names = np.unique(y_valid_classes) + 1

    target_names = [f"Star {i}" for i in range(1, 6)]
    print("Classification Report:")
    print(classification_report(y_valid_classes, preds_classes,
                                target_names=target_names, digits=3))
    print(f"Overall Accuracy: {accuracy_score(y_valid_classes, preds_classes):.3f}")

    # 10. Confusion matrix
    plot_confusion_matrices(y_valid_classes, preds_classes, class_names)


if __name__ == "__main__":
    main()
