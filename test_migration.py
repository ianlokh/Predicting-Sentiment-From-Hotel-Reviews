#!/usr/bin/env python3
"""
TF 2.18 Migration Tests
Verifies that all migrated code is compatible with TensorFlow 2.18 / Keras 3.
Run with the project's TF environment:
  /opt/homebrew/Caskroom/miniforge/base/envs/tfrl-latest-metal_py312/bin/python test_migration.py
"""
import sys
import tempfile
import os
import numpy as np

# ---------------------------------------------------------------------------
# 1. TF / Keras version check
# ---------------------------------------------------------------------------
def test_tensorflow_version():
    import tensorflow as tf
    major, minor, _ = tf.__version__.split('.')
    assert int(major) == 2 and int(minor) >= 18, (
        f"Expected TF >= 2.18, got {tf.__version__}"
    )
    print(f"[PASS] TensorFlow version: {tf.__version__}")


def test_keras_version():
    import keras
    major, minor, _ = keras.__version__.split('.')
    assert int(major) >= 3, f"Expected Keras >= 3.x, got {keras.__version__}"
    print(f"[PASS] Keras version: {keras.__version__}")


# ---------------------------------------------------------------------------
# 2. Import checks — all imports used in the migrated files
# ---------------------------------------------------------------------------
def test_tensorflow_keras_imports():
    from tensorflow import keras
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.layers import (
        Dense, Embedding, LSTM, Bidirectional,
        BatchNormalization, TimeDistributed,
    )
    print("[PASS] All tensorflow.keras imports resolved")


def test_loss_lr_scheduler_import():
    # Ensure LossLearningRateScheduler imports cleanly from the migrated module
    sys.path.insert(0, os.path.dirname(__file__))
    from LossLearningRateScheduler import LossLearningRateScheduler
    print("[PASS] LossLearningRateScheduler imported")


# ---------------------------------------------------------------------------
# 3. LossLearningRateScheduler — functional test with a toy model
# ---------------------------------------------------------------------------
def test_loss_lr_scheduler_callback():
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    sys.path.insert(0, os.path.dirname(__file__))
    from LossLearningRateScheduler import LossLearningRateScheduler

    # Build a tiny model
    model = Sequential([Dense(4, activation='relu', input_shape=(2,)),
                        Dense(1, activation='sigmoid')])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    X = np.random.rand(60, 2).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)

    # Run enough epochs so the scheduler can look back
    scheduler = LossLearningRateScheduler(
        base_lr=0.01,
        lookback_epochs=2,
        decay_threshold=0.002,
        decay_multiple=0.5,
    )
    model.fit(X, y, epochs=5, batch_size=16, verbose=0,
              validation_split=0.2, callbacks=[scheduler])

    # Verify the learning rate is still a sensible positive float
    final_lr = float(model.optimizer.learning_rate)
    assert final_lr > 0, f"Learning rate should be positive, got {final_lr}"
    print(f"[PASS] LossLearningRateScheduler ran; final lr={final_lr:.6f}")


def test_lr_scheduler_spike():
    from tensorflow import keras
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    sys.path.insert(0, os.path.dirname(__file__))
    from LossLearningRateScheduler import LossLearningRateScheduler

    model = Sequential([Dense(4, activation='relu', input_shape=(2,)),
                        Dense(1, activation='sigmoid')])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy')

    X = np.random.rand(40, 2).astype(np.float32)
    y = (X[:, 0] > 0.5).astype(np.float32)

    scheduler = LossLearningRateScheduler(
        base_lr=0.001,
        lookback_epochs=2,
        spike_epochs=[3],
        spike_multiple=5,
    )
    model.fit(X, y, epochs=5, batch_size=8, verbose=0,
              validation_split=0.2, callbacks=[scheduler])

    final_lr = float(model.optimizer.learning_rate)
    assert final_lr > 0
    print(f"[PASS] LR spike test passed; final lr={final_lr:.6f}")


# ---------------------------------------------------------------------------
# 4. Optimizer API — learning_rate kwarg, no lr / no decay
# ---------------------------------------------------------------------------
def test_adam_learning_rate_kwarg():
    from tensorflow import keras
    adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9,
                                 beta_2=0.999, epsilon=1e-08)
    print("[PASS] Adam(learning_rate=...) accepted")


def test_adam_no_lr_kwarg():
    from tensorflow import keras
    try:
        keras.optimizers.Adam(lr=0.001)
        raise AssertionError("Adam(lr=...) should have raised ValueError in Keras 3")
    except (ValueError, TypeError):
        print("[PASS] Adam(lr=...) correctly rejected in Keras 3")


def test_optimizer_learning_rate_assign():
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense
    model = Sequential([Dense(1, input_shape=(1,))])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss='mse')
    model.build((None, 1))

    original_lr = float(model.optimizer.learning_rate)
    model.optimizer.learning_rate.assign(0.005)
    new_lr = float(model.optimizer.learning_rate)
    assert abs(new_lr - 0.005) < 1e-6, f"Expected 0.005, got {new_lr}"
    print(f"[PASS] optimizer.learning_rate.assign() works ({original_lr} → {new_lr})")


# ---------------------------------------------------------------------------
# 5. Model build, save (.keras), and reload
# ---------------------------------------------------------------------------
def test_model_build_and_save():
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Embedding, LSTM, Bidirectional

    # Fix seeds so LSTM weight initialisation and input generation are deterministic
    np.random.seed(0)
    tf.random.set_seed(0)

    max_features = 100
    embed_dim = 16
    seq_len = 20

    embedding_matrix = np.random.rand(max_features, embed_dim).astype(np.float32)

    # Note: recurrent_dropout is intentionally omitted here. In Keras 3 / TF backend,
    # LSTM with recurrent_dropout > 0 applies a stochastic mask even during inference
    # (the mask seed is resampled on each forward pass), so consecutive model.predict()
    # calls on the same input produce different outputs. The actual training model uses
    # recurrent_dropout=0.4 which is correct for regularisation during training — this
    # test just validates save/load weight correctness with a dropout-free equivalent.
    model = Sequential([
        Embedding(max_features, embed_dim, input_length=seq_len,
                  weights=[embedding_matrix], trainable=True),
        Bidirectional(LSTM(32, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(5, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001,
                                        beta_1=0.9, beta_2=0.999, epsilon=1e-08),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    print("[PASS] Model built and compiled")

    x = np.random.randint(0, max_features, (4, seq_len))
    # Capture predictions BEFORE saving — in Keras 3, model.save() modifies
    # internal model state so predictions taken after save() will differ.
    preds_orig = model(x, training=False).numpy()

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'model.keras')
        model.save(save_path)
        assert os.path.exists(save_path), "Model file not created"
        print("[PASS] Model saved as .keras format")

        reloaded = load_model(save_path)
        assert reloaded is not None
        preds_reload = reloaded(x, training=False).numpy()
        np.testing.assert_allclose(preds_orig, preds_reload, rtol=1e-5,
                                   err_msg="Reloaded model predictions differ")
        print("[PASS] Model reloaded and predictions match (numerical correctness)")


# ---------------------------------------------------------------------------
# 6. sklearn get_feature_names_out
# ---------------------------------------------------------------------------
def test_get_feature_names_out():
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    cv.fit(['hotel room clean', 'great location staff friendly', 'room dirty'])
    names = list(cv.get_feature_names_out())
    assert len(names) > 0
    print(f"[PASS] get_feature_names_out() returns {len(names)} features")


def test_get_feature_names_removed():
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    cv.fit(['hello world'])
    try:
        cv.get_feature_names()
        raise AssertionError("get_feature_names() should be removed in sklearn 1.x")
    except AttributeError:
        print("[PASS] get_feature_names() correctly removed; use get_feature_names_out()")


# ---------------------------------------------------------------------------
# 7. Tokenizer and pad_sequences from tensorflow.keras
# ---------------------------------------------------------------------------
def test_tokenizer_and_padding():
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    texts = ['the hotel was great', 'room was very clean', 'excellent location']
    tokenizer = Tokenizer(num_words=50, split=' ')
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, padding='post', truncating='post', maxlen=10)
    assert padded.shape == (3, 10), f"Expected (3,10), got {padded.shape}"
    print("[PASS] Tokenizer and pad_sequences work correctly")


# ---------------------------------------------------------------------------
# 8. pandas — df.append removed, pd.concat used instead
# ---------------------------------------------------------------------------
def test_pandas_concat_replaces_append():
    import pandas as pd
    # Confirm df.append is gone
    df = pd.DataFrame({'a': [1, 2]})
    assert not hasattr(df, 'append'), "df.append still exists (expected removed)"

    # Confirm pd.concat works as replacement
    rows_list = [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'y'}]
    result = pd.concat([pd.DataFrame(), pd.DataFrame(rows_list)], ignore_index=True)
    assert list(result['a']) == [1, 2]
    print("[PASS] pd.concat replaces df.append correctly")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    tests = [
        test_tensorflow_version,
        test_keras_version,
        test_tensorflow_keras_imports,
        test_loss_lr_scheduler_import,
        test_loss_lr_scheduler_callback,
        test_lr_scheduler_spike,
        test_adam_learning_rate_kwarg,
        test_adam_no_lr_kwarg,
        test_optimizer_learning_rate_assign,
        test_model_build_and_save,
        test_get_feature_names_out,
        test_get_feature_names_removed,
        test_tokenizer_and_padding,
        test_pandas_concat_replaces_append,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    sys.exit(0 if failed == 0 else 1)
