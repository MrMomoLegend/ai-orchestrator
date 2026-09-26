# The document corpus

This folder is the assistant's knowledge base. Anything you drop in here (`.txt`, `.md`,
`.pdf`) becomes answerable material once you run:

    python ingest.py --reset

## What is in the repository

22 openly licensed PDFs on natural language processing, so a fresh clone runs end to end:

- **Wikipedia articles (13)**: CYK algorithm, GPT-4, Hallucination (artificial
  intelligence), Hidden Markov model, Large language model, Named-entity recognition,
  Part-of-speech tagging, Retrieval-augmented generation, Speech recognition, Viterbi
  algorithm, Word2Vec, n-gram, tf–idf. Licensed CC BY-SA 4.0; © Wikipedia contributors.
  Source: https://en.wikipedia.org
- **Dive into Deep Learning (d2l.ai) sections**: approx-training, bert, glove,
  language-model, sentiment-analysis-and-dataset, similarity-analogy, subword-embedding,
  transformer, word2vec. Zhang, Lipton, Li & Smola, *Dive into Deep Learning*. Text
  licensed CC BY-SA 4.0. Source: https://d2l.ai

Both are redistributed unmodified under CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/).

## Relation to the report

Chapter 5's project-corpus results were measured against exactly these files, with the
questions in `eval/questions_rag.csv`. The retrieval threshold (0.53) and k (10) were fixed
earlier on a non-redistributable course-reading corpus and were not re-tuned here.
