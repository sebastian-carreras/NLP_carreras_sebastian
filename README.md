# Natural Language Processing — from TF-IDF to seq2seq

Four assignments from the NLP course of the **AI Specialization at FIUBA** (Universidad de Buenos
Aires), working up from classical text vectorization to encoder-decoder architectures. Each one is
a self-contained notebook with its results reported as they came out — including the parts that
didn't work.

> **What this is:** coursework, kept honest. The models are small, the corpora are small, and
> several results are negative. They're reported that way on purpose — a language model trained on
> one book is *supposed* to have a high perplexity, and saying so is more useful than hiding it.

| # | Task | Approach | Headline result |
|---|---|---|---|
| 1 | Text classification | TF-IDF + Naive Bayes | **F1-macro 0.693** (ComplementNB) on 20 Newsgroups |
| 2 | Word embeddings | Word2Vec (gensim) | One analogy solved cleanly: `francia − paris + inglaterra = londres` |
| 3 | Language modelling | LSTM vs GRU | Best perplexity **1925** (LSTM), and a clear overfitting story |
| 4 | Machine translation & QA | Seq2seq encoder-decoder | From-scratch LSTM vs a pretrained transformer — the gap is the point |

---

## Challenge 1 — Text classification with TF-IDF and Naive Bayes

**Dataset:** 20 Newsgroups — 11,314 training documents across 20 classes, headers/footers/quotes
removed. The raw TF-IDF vocabulary is 101,631 terms.

Document similarity was explored first (take 5 documents at random, inspect their 5 nearest
neighbours, check whether the similarity agrees with the class label), then the matrix was
transposed to study term–term similarity in the same space.

For classification, a `Pipeline` of `TfidfVectorizer` + Naive Bayes was tuned with `GridSearchCV`
optimizing `f1_macro`.

**Results**

| Model | F1-macro |
|---|---|
| **ComplementNB** | **0.693** |
| MultinomialNB | 0.656 |

Best configuration: `ComplementNB(alpha=1.0)`, `max_features=30000`, `ngram_range=(1,1)`,
`stop_words='english'`.

ComplementNB wins by ~4 points, which is the expected result on 20 Newsgroups: the classes are
uneven, and ComplementNB is specifically designed to correct the bias MultinomialNB develops on
imbalanced data. Unigrams beat bigrams here, and capping the vocabulary at 30k terms — under a
third of the raw 101k — didn't hurt.

**Topics:** TF-IDF, n-grams, cosine similarity, `GridSearchCV`, scikit-learn pipelines.
📓 [Notebook](Challenge%201/exercises/Challenge_1.ipynb) · 📄 [Assignment](Challenge%201/exercises/Assignment.md)

---

## Challenge 2 — Word embeddings with Word2Vec

**Corpus:** *Los Tres Mosqueteros* (Alexandre Dumas), in Spanish — a single novel, which is a
deliberately small corpus.

Trained with gensim: `min_count=10`, `window=3`, `vector_size=100`, `negative=19`.

**Nearest neighbours** — the model clearly picks up the book's social structure:

```
rey        → luis, escudero, xiii, richelieu, bassompierre
cardenal   → suizo, escudero, cojin, duque, abate
mosqueteros→ soldados, companeros, valientes, gascuna
milady     → burgues, infame, dartagnan, prisionera, felton
```

`rey` retrieving *luis*, *xiii* and *richelieu* is a real result: the embedding space has placed
Louis XIII and Cardinal Richelieu next to the concept of the king, which is exactly the political
structure of the novel. Nothing in the training signal told it who these people were.

**Analogy tests** — mostly failures, with one clean success:

```
francia − paris + inglaterra = londres          ✅ country/capital, solved
buckingham − inglaterra + richelieu = reino     ~ close: right domain, wrong word
dartagnan − mosquetero + porthos = uniforme     ❌
milady − venganza + winter = lord               ❌
```

This is the honest outcome for one book. Analogy arithmetic needs the kind of coverage you get
from a billion-token corpus; with a single novel you get strong *local* neighbourhoods (the
similarity results above) but not reliable *global* geometry. The country–capital pair works
because it's the one relation the text repeats often enough.

**Topics:** Word2Vec (skip-gram / CBOW), negative sampling, semantic similarity, analogy
arithmetic, embedding visualization.
📓 [Notebook](Challenge%202/exercises/Challenge_2.ipynb) · 📄 [Assignment](Challenge%202/exercises/Assignment.md)

---

## Challenge 3 — Language modelling with LSTM and GRU

**Corpus:** *El Arte de la Guerra* (Sun Tzu), Spanish translation.

Two recurrent architectures were trained for next-word prediction and compared under identical
preprocessing, with a per-epoch perplexity callback on the validation set.

**Results**

| Model | Best perplexity | At epoch | Perplexity at end of training |
|---|---|---|---|
| **LSTM** | **1925** | 11 of 31 | 7147 |
| GRU | 2355 | 2 of 22 | 10580 |

The LSTM wins on best perplexity. But the more interesting column is the last one: **both models
were trained well past their optimum**, ending 3.7× and 4.5× worse than their own best epoch. The
perplexity callback was measuring the right thing the whole time — the training loop just wasn't
stopping on it. Early stopping on validation perplexity is the fix, and the numbers above are what
its absence costs.

**Sequence generation** — the classic decoding trade-off, visible in the outputs:

```
beam search + stochastic sampling:  "la guerra es ventaja operaciones gran los"
greedy search:                      "la guerra es la que de de"
```

Greedy collapses onto the highest-frequency function words (`la`, `que`, `de`) and repeats them.
Sampling escapes the loop and produces content words, but without coherent syntax. Neither is
good — with a corpus this size neither can be — but they fail in opposite and instructive ways.

**Topics:** language modelling, RNN/LSTM/GRU, perplexity, tokenization, greedy search, beam
search (deterministic and stochastic), temperature.
📓 [Notebook](Challenge%203/exercises/Challenge_3.ipynb) · 📄 [Assignment](Challenge%203/exercises/Assignment.md)

---

## Challenge 4 — Seq2seq: translation and QA

### 4a — Neural machine translation (English → Spanish)

An encoder-decoder LSTM built from scratch with the Keras functional API — three models in total:
one for training, plus separate encoder-only and decoder-only models for step-by-step inference.

Then the same sentences were run through a **pretrained transformer** (Helsinki-NLP, via Hugging
Face) for comparison:

| Input | From-scratch LSTM | Pretrained transformer |
|---|---|---|
| *Tom is naked.* | `tom es la noche` | **`Tom está desnudo.`** |
| *My mother say hi.* | `él no no es` | **`Mi madre me manda saludos.`** |
| *Mary wants to buy a dress.* | `él no no no se dijo a mary` | — |

The from-scratch model produces Spanish-shaped word sequences with no semantic fidelity; the
pretrained model is simply correct — it even renders *"say hi"* idiomatically as *"me manda
saludos"* rather than literally.

That gap is the actual deliverable here. Building the seq2seq by hand is what makes the
comparison meaningful: it shows concretely what a few thousand training pairs and a 128-unit LSTM
can and cannot buy you, against a model trained on orders of magnitude more data.

**Topics:** seq2seq, encoder-decoder, teacher forcing, separate inference models, transfer
learning with Hugging Face.
📓 [Notebook](Challenge%204/exercises/Challenge_4_translator.ipynb)

### 4b — QA bot (a debugging story)

A conversational QA bot on the same encoder-decoder pattern. The **first version was broken** — it
returned the same degenerate output for every question:

```
Q: How can I reset my password?    A: <start> <start> ... western western ... india global
Q: What payment methods...?        A: <start> <start> ... western western ... india global   ← identical
```

The notebook now opens with the full diagnosis. There were five bugs, but two mattered, and both
were in the **data, not the architecture**:

1. **Missing target shift (the root cause).** The decoder's input and target were the same array,
   so teacher forcing trained the model to predict token *t* from token *t* — the identity
   function. Seeded with `<start>` at inference, it predicts `<start>`, feeds it back, and loops.
   The garbage wasn't noise; it was the model doing exactly what it was trained to do.
2. **94.9% of the training pairs were wrong answers.** `wiki_qa` is a sentence-*selection*
   dataset — each row is (question, candidate, `label`) — and the notebook ignored `label`, so
   19,320 of 20,360 rows taught the model to answer with a sentence that doesn't answer the
   question. The fix switches to **PersonaChat**, the conversational dataset the assignment's own
   example questions come from.

Also fixed: an 11 GB dense one-hot replaced by `sparse_categorical_crossentropy`; the output
vocabulary (was truncated to 3k of 32k words); and a model that was redefined *after* training.

**After the fix**, the same model answers coherently:

```
Q: Do you have any pet?            A: i do not have any pets
Q: Where are you from?             A: i am from the midwest how about you
Q: What do you do for a living?    A: i work at a local hospital
Q: Hi, how are you doing?          A: i am good how are you
Q: Do you read?                    A: i do not have any pets          ← still generic on some inputs
```

The remaining failure is honest and expected: a small seq2seq without attention, trained with
cross-entropy, collapses toward frequent generic answers. Early stopping on validation loss
(best epoch 14 of 17) keeps it from overtraining — the discipline that challenge 3 lacked.

**The lesson:** both real bugs were silent. Nothing threw an exception, the accuracy curve went
up, and the output was garbage — the exact failure mode that matters in an ML pipeline, where the
dangerous errors are the ones that return a plausible number instead of crashing.

📓 [Notebook](Challenge%204/exercises/Challenge_4_QA_bot.ipynb) — runs end to end, cells execute in order.

---

## Running the notebooks

```bash
conda activate ia_ceia_18co     # or any env with the dependencies below
jupyter lab
```

**Dependencies:** `scikit-learn`, `gensim`, `tensorflow` / `keras`, `pandas`, `numpy`,
`matplotlib`, `seaborn`, `transformers`, `datasets`.

The **translator (4a)** additionally needs the **FastText** `cc.en.300.vec` vectors
(~4.5 GB uncompressed); the notebook downloads them on first run and caches a pickle alongside.
The **QA bot (4b)** trains its embeddings from scratch and needs no external vectors — it pulls
**PersonaChat** through the `datasets` library on first run.

The book corpora used by challenges 2 and 3 are included under each challenge's
`exercises/book_dataset/`.

---

*NLP coursework — AI Specialization, FIUBA (Universidad de Buenos Aires). Sebastian Carreras.*
