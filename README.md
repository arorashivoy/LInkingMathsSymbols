# Linking mathematical symbols to their descriptions

SemEval-2022 Task 12 (**SYMLINK**): given a scientific paper, find the mathematical
symbols and the natural-language descriptions that define them, and work out which
description belongs to which symbol.

A paper says *"where \( \alpha \) is the learning rate"*. A reader links `α` to
"learning rate" without thinking about it. The task is to do that automatically,
across papers where the same symbol is reused with different meanings and where one
description can cover several symbols at once.

Course project at IIIT-Delhi, April 2024, built jointly by
**Chehak Malhotra** and **Shivoy Arora**.

We worked on it together from one machine, so the whole commit history sits under
a single account. The contributor graph is not a record of who did what here.

## Approach

Two stages, trained separately.

**1. Span tagging** — a token-classification model marks every token as the beginning
or inside of a `SYMBOL` span, the beginning or inside of a `PRIMARY` description span,
or `OTHER`. Fine-tunes [`KISTI-AI/scideberta`](https://huggingface.co/KISTI-AI/scideberta),
a DeBERTa variant pretrained on scientific text, with a hand-written PyTorch training
loop: 10 epochs, AdamW at `lr=1e-5`, sequences truncated to 512 tokens.

**2. Relation classification** — every plausible pair of spans from stage 1 is
classified into one of five relations:

| Relation | Meaning |
|---|---|
| `Direct` | this description defines this symbol |
| `Corefer-Symbol` | the two spans are the same symbol |
| `Corefer-Description` | the two spans describe the same thing |
| `Count` | one span states how many of the other there are |
| `None` | unrelated |

This stage uses [`studio-ousia/luke-base`](https://huggingface.co/studio-ousia/luke-base)
via `LukeForEntityPairClassification`. LUKE is the right choice here because it takes
entity spans as first-class inputs rather than requiring the pair to be marked up in
the text — which is exactly the shape of this problem. Trained with the Hugging Face
`Trainer` for 4 epochs at batch size 8.

## Results

**Stage 1 — span tagging**, over 24,064 test tokens:

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `B_PRIMARY` | 0.95 | 0.93 | 0.94 | 132 |
| `B_SYMBOL` | 0.79 | 0.68 | 0.73 | 219 |
| `I_PRIMARY` | 0.75 | 0.90 | 0.82 | 420 |
| `I_SYMBOL` | 0.61 | 0.88 | 0.72 | 930 |
| `OTHER` | 0.99 | 0.97 | 0.98 | 22,363 |
| **accuracy** | | | **0.96** | 24,064 |
| **macro avg** | 0.82 | 0.87 | **0.84** | 24,064 |

The 0.96 accuracy is not the interesting number. `OTHER` is 93% of the tokens, so
almost any model scores highly on it. **Macro-F1 of 0.84 is the honest summary**, and
the weakest classes are the ones the task is about: `I_SYMBOL` at 0.72 and `B_SYMBOL`
at 0.73. Symbol boundaries are where this model loses.

**Stage 2 — relation classification**:

| | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|
| Validation | 0.8214 | 0.7347 | | |
| **Test** | **0.7114** | **0.6647** | 0.6581 | 0.7124 |

The eleven-point drop from validation to test is the result worth explaining rather
than hiding: the model is fitting something about the validation split that does not
generalise.

## Running the notebooks

Both were written for [Kaggle](https://www.kaggle.com) and will not run unchanged
anywhere else. They read from hardcoded `/kaggle/input/...` paths, expect the
SemEval-2022 Task 12 data to be attached as a Kaggle dataset, and assume a **T4 GPU**.
To run them elsewhere, download the task data from the
[SYMLINK competition page](https://competitions.codalab.org/competitions/34011) and
change those paths.

```sh
pip install -r requirements.txt
```

| Notebook | Stage |
|---|---|
| `nlp-project-ner-2.ipynb` | span tagging |
| `nlp-project-re.ipynb` | relation classification |

`NLP_project.pdf` is the write-up.
