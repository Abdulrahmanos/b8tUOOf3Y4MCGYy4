# Potential Talent: Candidate Ranking with NLP

An educational natural language processing project that ranks potential candidates by comparing their job titles with a recruiter's target role.

The notebook begins with transparent lexical baselines, progresses to dense and contextual embeddings, and then demonstrates how recruiter feedback can personalize the ranking.

## Project overview

The workflow:

1. Loads and inspects the candidate dataset.
2. removes duplicate job titles.
3. Cleans and audits the text.
4. Converts job titles into numerical representations.
5. Ranks candidates using cosine similarity.
6. Re-ranks candidates when a recruiter stars or rejects profiles.
7. Compares the different representation methods.

## Methods explored

| Method | Purpose |
| --- | --- |
| Bag of Words | Raw word-count baseline |
| TF-IDF | Weights distinctive terms more strongly than common terms |
| Word2Vec | Learns local word-context relationships |
| GloVe | Learns from global word co-occurrence |
| FastText | Adds character-level subword information |
| BERT | Produces contextual token embeddings |
| SBERT | Produces sentence embeddings designed for semantic similarity |

The notebook uses cosine similarity to compare a query embedding with each candidate embedding:

```text
cos(q, c) = (q · c) / (||q|| ||c||)
```

## Recruiter feedback

A recruiter can star relevant candidates or reject irrelevant candidates. The SBERT query is adjusted toward the average starred-candidate embedding and away from the average rejected-candidate embedding:

```text
q' = normalize(0.60q + 0.40mean(stars) - 0.15mean(rejects))
```

The candidates are then ranked again using the updated query. A small bonus keeps directly starred candidates visible while allowing similar, unstarred candidates to move upward too.

## Repository contents

- `potential talent.ipynb` — data exploration, text representations, similarity ranking, model comparison, and recruiter-feedback demonstration.
- `potential-talents.xlsx` — candidate data with ID, job title, location, connection count, and fit fields.

## Getting started

### 1. Clone the repository

```powershell
git clone https://github.com/Abdulrahmanos/b8tUOOf3Y4MCGYy4.git
cd b8tUOOf3Y4MCGYy4
```

### 2. Create and activate a virtual environment

Python 3.11 or 3.12 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install the dependencies

```powershell
python -m pip install jupyter pandas numpy matplotlib scikit-learn openpyxl torch transformers sentence-transformers
```

### 4. Run the notebook

```powershell
jupyter lab "potential talent.ipynb"
```

Run the cells from top to bottom. The pretrained BERT and SBERT models are downloaded the first time their sections run, so those cells require an internet connection.

## Key conclusion

Bag of Words and TF-IDF provide understandable lexical baselines. The locally trained Word2Vec, GloVe, and FastText examples are useful for learning, but the dataset is too small for them to become strong production models. Pretrained SBERT is the most suitable semantic baseline in this notebook because it is designed for sentence-level similarity.

A production model should be evaluated with real recruiter relevance labels and ranking metrics such as Precision@K, Recall@K, MAP, or NDCG.

## Status

This repository is a learning project and experimental prototype, not a production hiring system.
