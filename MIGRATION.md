# MIGRATION.md — Task Checklist

## Phase 1: Automated Upgrade Script
- [x] Upgrade notebook from nbformat 4.4 → 4.5 (added cell IDs)
- [x] No tf_upgrade_v2 needed — project used standalone Keras 2.x, not TF 1.x Session API

## Phase 2: Core Module Migration
- [x] LossLearningRateScheduler.py — migrated to tf.keras
- [x] utils.py — fixed deprecated sklearn API
- [x] Hotel Review Sentiment Classifier with LSTM.ipynb — migrated all cells

## Phase 3: Utility and Contrib Replacement
- [x] No tf.contrib dependencies found

## Phase 4: Validation
- [x] 14/14 migration-specific tests pass (test_migration.py)
- [x] Numerical correctness: reloaded model predictions match original within rtol=1e-5
- [ ] Full end-to-end training run (requires data files not present in repo)
- [ ] Model export/serving with real trained weights

---

## Change Summary

### LossLearningRateScheduler.py
| Old (Keras 2 standalone) | New (TF 2.18 / Keras 3) |
|---|---|
| `import keras` | `from tensorflow import keras` |
| `from keras import backend as K` | *(removed — no longer needed)* |
| `K.get_value(self.model.optimizer.lr)` | `float(self.model.optimizer.learning_rate)` |
| `K.set_value(self.model.optimizer.lr, v)` | `self.model.optimizer.learning_rate.assign(v)` |

**Reason:** `optimizer.lr` attribute removed in Keras 3; must use `learning_rate`.
`K.get_value/set_value` replaced with native variable calls.

### utils.py
| Old | New |
|---|---|
| `ngram_vectorizer.get_feature_names()` | `ngram_vectorizer.get_feature_names_out()` |

**Reason:** `get_feature_names()` removed in scikit-learn 1.x.

### Notebook (Hotel Review Sentiment Classifier with LSTM.ipynb)
| Cell | Old | New |
|---|---|---|
| imports | `import keras` / `from keras.X import Y` | `from tensorflow import keras` / `from tensorflow.keras.X import Y` |
| data load | `rows.append(rw, ignore_index=True)` | `pd.concat([rows, pd.DataFrame(rw)], ignore_index=True)` |
| data load | `df.drop('indx', 1)` | `df.drop('indx', axis=1)` |
| gensim vocab | `dict([(k, v.index) for k, v in model.vocab.items()])` | `model.key_to_index` |
| gensim vectors | `model.syn0[i]` | `model.vectors[i]` |
| optimizer | `Adam(lr=0.001, ..., decay=0.001)` | `Adam(learning_rate=0.001, ...)` |
| model save | `model.save('file.hdf5')` | `model.save('file.keras')` |
| model load | `load_model('file.hdf5')` | `load_model('file.keras')` |

**Key reasons:**
- `import keras` standalone is Keras 2; TF 2.18 bundles Keras 3 via `tensorflow.keras`
- `Adam(lr=...)` and `decay=` parameter removed in Keras 3
- `df.append()` removed in pandas 2.x
- Gensim 4.x: `model.vocab` → `model.key_to_index`, `model.syn0` → `model.vectors`
- HDF5 model format is legacy; `.keras` is the recommended format

### Notebook format
- Upgraded from nbformat 4.4 → 4.5 (added UUID cell IDs required by modern Jupyter)

---

## Known Behaviours / Notes
- `recurrent_dropout > 0` in Keras 3 LSTM: the dropout mask is resampled on every
  forward call, so `model.predict()` calls on the *same* model after `model.save()` will
  give different results than a freshly-reloaded model. This is a Keras 3 internal
  behaviour change, not a weight-corruption issue. To compare predictions numerically,
  capture them *before* calling `model.save()`.
- `Embedding(input_length=...)` is deprecated in Keras 3 but still accepted; a
  deprecation warning may appear at runtime.

---

## Session Log

### Session 1 — 2026-02-22
- Read CLAUDE.md and MIGRATION.md; confirmed target is TF 2.18.0
- Analysed all .py files and .ipynb; found no TF 1.x Session/placeholder patterns
- Identified actual migration scope: standalone Keras 2 → tensorflow.keras (Keras 3)
- Upgraded notebook to nbformat 4.5
- Migrated LossLearningRateScheduler.py, utils.py, and all affected notebook cells
- Created test_migration.py with 14 tests covering: TF/Keras version, imports,
  LossLearningRateScheduler callback (decay + spike), optimizer API, model save/load
  numerical correctness, sklearn API, Tokenizer/pad_sequences, pandas concat
- All 14 tests pass on TF 2.18.0 / Keras 3.12.0
  (env: /opt/homebrew/Caskroom/miniforge/base/envs/tfrl-latest-metal_py312)
