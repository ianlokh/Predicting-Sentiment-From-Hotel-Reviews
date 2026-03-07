# CLAUDE.md — TensorFlow Migration Project

## Project Overview
This project migrates [project name] from TensorFlow [source version] to TensorFlow 2.18.
The codebase is a [brief description: e.g., image classification pipeline with custom training loops].

## Migration Goals
- Remove all tf.compat.v1 usage and Session-based execution
- Convert all models to tf.keras API
- Replace tf.contrib dependencies with tf.keras, tf-addons, or tf-slim
- Maintain numerical correctness (results must match within tolerance)
- Preserve all existing test coverage and add migration-specific tests

## Current TF Version & Target
- Source: TensorFlow [1.x or older 2.x version]
- Target: TensorFlow 2.18.0

## Translation Rules

| TF 1.x Pattern | TF 2.x Equivalent | Notes |
|---|---|---|
| `tf.Session()` / `sess.run()` | Eager execution (remove sessions) | Use `tf.function` for graph mode |
| `tf.placeholder()` | Function arguments | Pass tensors directly |
| `tf.Variable` + `tf.global_variables_initializer()` | `tf.Variable` (auto-initialized) | No manual init needed |
| `tf.contrib.layers.*` | `tf.keras.layers.*` or `tf_slim` | Check tf-addons for missing ops |
| `tf.flags` | `absl.flags` | External package |
| `tf.train.Saver` | `tf.train.Checkpoint` or `model.save()` | Prefer SavedModel format |
| `tf.estimator.*` | `tf.keras` with custom training | Estimator API is deprecated |
| `feed_dict` | Direct tensor passing | Use tf.data pipelines |
| `tf.layers.*` | `tf.keras.layers.*` | Direct replacement |
| `tf.losses.*` | `tf.keras.losses.*` | API is similar |

## Project Structure
