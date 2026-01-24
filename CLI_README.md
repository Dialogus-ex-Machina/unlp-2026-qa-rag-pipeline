# `unlp`

**Usage**:

```console
$ unlp [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `eval`: Run evaluation for a given metric
* `invoke`: Run workflow for random sampled question...
* `kb`: Knowledge base management commands

## `unlp eval`

Run evaluation for a given metric

**Usage**:

```console
$ unlp eval [OPTIONS] [METRIC]:[answer-accuracy|answer-faithfulness|composite-accuracy|doc-source-accuracy|doc-source-page-accuracy]
```

**Arguments**:

* `[METRIC]:[answer-accuracy|answer-faithfulness|composite-accuracy|doc-source-accuracy|doc-source-page-accuracy]`: [default: answer-accuracy]

**Options**:

* `-ds, --dataset [full-qa|sport-qa|medicine-qa|full-qa-with-context|sport-qa-with-context|medicine-qa-with-context]`: [default: full-qa]
* `-m, --model TEXT`
* `-key, --api-key TEXT`
* `--help`: Show this message and exit.

## `unlp invoke`

Run workflow for random sampled question from dataset.

**Usage**:

```console
$ unlp invoke [OPTIONS]
```

**Options**:

* `-ds, --dataset [full-qa|sport-qa|medicine-qa|full-qa-with-context|sport-qa-with-context|medicine-qa-with-context]`: [default: full-qa]
* `-m, --model TEXT`
* `-key, --api-key TEXT`
* `--help`: Show this message and exit.

## `unlp kb`

Knowledge base management commands

**Usage**:

```console
$ unlp kb [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `create`: Creates knowledge base.

### `unlp kb create`

Creates knowledge base.
Uploads embeddings to vector store and builds index for retrieval of documents.

**Usage**:

```console
$ unlp kb create [OPTIONS]
```

**Options**:

* `-m, --model TEXT`
* `-key, --api-key TEXT`
* `--help`: Show this message and exit.
