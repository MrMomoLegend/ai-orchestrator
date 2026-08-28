# The document corpus

This folder is where the assistant's knowledge base lives. Anything you drop in here
(`.txt`, `.md`, `.pdf`) becomes answerable material once you run `python ingest.py`.

## What is in the repository

Four short plain-text documents, written by the author, are included. They are the
controlled corpus used for the feature prototype in the preliminary report:

- `Artificial Intelligence.txt`
- `Machine Learning.txt`
- `Space Exploration.txt`
- `University of London Final Project.txt`

They are enough to run the system end to end and see grounded answering and refusal
behaviour working, which is what a first clone needs.

## What is not, and why

The corpus the evaluation in Chapter 5 was run against is **not** in this repository.
It was 28 files from the CM3060 Natural Language Processing reading list — extracts from
*Speech and Language Processing* (course textbook), *Natural Language Processing with
Python* (course textbook), *Data Science for Business* (course textbook) and
*Introduction to Information Retrieval* (course textbook), supplied through the module.

Those are published works under third-party copyright. Using them locally for private
study and research is fair dealing; republishing them on a public repository is not. They
are therefore excluded from the working tree **and** from the git history.

This is why the reported numbers cannot be reproduced byte-for-byte from a clone alone.
The scripts, the 30-question set with its labels, the per-question outputs and the
distance measurements are all committed under `eval/` and `results/`, so every claim
remains auditable — but rebuilding the exact 3,215-chunk vector store requires the same
source PDFs, which anyone enrolled on the module already has.

## Using your own corpus

Nothing about the system is specific to that material. Drop your own documents in:

```bash
cp ~/my-notes/*.pdf docs/
python ingest.py
```

`ingest.py` builds two collections from a single pass over the same files —
`documents_sentence` (sentence-aware chunking, the shipped default) and `documents_fixed`
(fixed-size, the prototype behaviour) — so the Section 5.4 ablation stays runnable on
whatever you ingest.

Two things worth knowing before you judge the results on your own material:

**Corpus size matters more than it looks.** Section 4.8 records an early threshold sweep
that produced badly overlapping distance distributions and a best balanced accuracy of
0.847. The cause was a vector store holding five chunks. With too little to match against,
retrieval distance reflects the sparsity of the corpus rather than the relevance of the
query, and any threshold derived from it is fitted to noise. `ingest.py` reports the chunk
count on completion — if it is in the tens rather than the thousands, treat the refusal
threshold as untuned.

**The 0.53 threshold is not universal.** It is the midpoint of the separating margin
measured on the evaluation corpus, where answerable and out-of-corpus questions separated
cleanly with a gap of 0.112. A corpus with different structure or density may not separate
as well. Retune with:

```bash
python eval/exp_rag.py --sweep
```

Scanned PDFs with no text layer will ingest as empty. The upload endpoint reports this as
a specific error rather than silently adding nothing.
