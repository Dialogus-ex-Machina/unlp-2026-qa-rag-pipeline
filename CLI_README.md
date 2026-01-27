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

* `evaluate`
* `invoke`: Run workflow for random sampled question...
* `kb`: Knowledge base management commands

## `unlp evaluate`

**Usage**:

```console
$ unlp evaluate [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `accuracy`: Run evaluation for a given metric
* `mteb`: Run mteb for a given tasks
* `faithfulness`: Evaluate faithfulness of answers.

### `unlp evaluate accuracy`

Run evaluation for a given metric

**Usage**:

```console
$ unlp evaluate accuracy [OPTIONS] [METRIC]:[answers|composite|doc-sources|doc-source-pages]
```

**Arguments**:

* `[METRIC]:[answers|composite|doc-sources|doc-source-pages]`: [default: answers]

**Options**:

* `-ds, --dataset [full|sport|medicine]`: [default: full]
* `-m, --model TEXT`
* `-key, --api-key TEXT`
* `--help`: Show this message and exit.

### `unlp evaluate mteb`

Run mteb for a given tasks

**Usage**:

```console
$ unlp evaluate mteb [OPTIONS]
```

**Options**:

* `-em, --embedding-model TEXT`: [default: bflhc/Octen-Embedding-0.6B]
* `-t, --task [all|belebele|web_faq|qa]`: [default: qa]
* `--help`: Show this message and exit.

### `unlp evaluate faithfulness`

Evaluate faithfulness of answers.

**Usage**:

```console
$ unlp evaluate faithfulness [OPTIONS]
```

**Options**:

* `-ds, --dataset [full|sport|medicine]`: [default: full]
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

* `-ds, --dataset [full|sport|medicine]`: [default: full]
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
