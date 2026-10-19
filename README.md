# Movie Recommender — Content-Based Filtering

Recommends movies similar to a given title using the **MovieLens 32M** dataset.
Each film is described by its genres plus the tags real users applied to it;
those descriptions are vectorised with TF-IDF and compared using cosine
similarity. An interactive widget lets you type a title and get recommendations.

**Dataset:** [MovieLens 32M](https://files.grouplens.org/datasets/movielens/ml-32m.zip)
— 87,585 movies · 32,000,204 ratings · 2,000,072 tags.

## Example

```python
get_recommendations("Toy Story")
```

| title | genres | rating | similarity |
|---|---|---|---|
| Toy Story 2 (1999) | Adventure, Animation, Children, Comedy, Fantasy | 3.81 | 0.859 |
| Bug's Life, A (1998) | Adventure, Animation, Children, Comedy | 3.56 | 0.744 |
| Toy Story 3 (2010) | Adventure, Animation, Children, Comedy, Fantasy | 3.83 | 0.636 |
| Finding Dory (2016) | Adventure, Animation, Comedy | 3.54 | 0.621 |
| Monsters, Inc. (2001) | Adventure, Animation, Children, Comedy, Fantasy | 3.84 | 0.606 |

Two more that exercise the title handling:

```python
get_recommendations("The Matrix")
# -> Matrix Reloaded, Matrix Revolutions, Tron, Dark City, Avalon

get_recommendations("Remember the Titans")
# -> Rudy, Friday Night Lights, Gridiron Gang, The Express, Invincible
```

## Running it

```bash
pip install -r requirements.txt
jupyter notebook MoviePrediction.ipynb
```

The notebook downloads and unzips the 228 MB dataset on first run (skipped
afterwards), then executes end to end in roughly 75 seconds.

## The important detail: similarity is computed on demand

A full pairwise matrix would be:

```python
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)   # don't
```

With 87,585 movies that is an 87,585 × 87,585 matrix — **7.7 billion entries**,
about **61 GB** as `float64`, which will not fit in memory on an ordinary
machine. So the notebook does not build it.

You almost never need the full matrix. A query only needs the single row for the
movie asked about:

```python
scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).ravel()   # 1 x 87,585
```

That is a few hundred kilobytes and milliseconds of work, and it gives identical
results for the query. This is what makes the notebook run on a normal laptop.

## Method

1. **Prepare** — average rating and rating count per movie; collapse each
   movie's tags into one string. 84,432 movies have at least one rating;
   51,323 have tags.
2. **Describe** — `combined_features` = genres + tags, per movie.
3. **Vectorise** — TF-IDF, capped at 20,000 features, with bigrams and
   `min_df=2`. Genres are a handful of repeated tokens; tags are free text, so
   TF-IDF correctly weights a rare tag far above `Drama`.
4. **Rank** — cosine similarity for the queried movie against every other,
   computed on demand.
5. **Look up titles** — `normalise_title()` strips the year, lowercases, and
   folds MovieLens' trailing article back to the front, so `The Matrix` finds
   `Matrix, The (1999)`. Exact match first, then a guarded substring fallback
   that prefers the most-rated candidate.

## Layout

```
MoviePrediction.ipynb   the analysis, executed with outputs saved
requirements.txt
```

## Limitations

This is **content-based only**. It uses no collaborative signal — what users who
liked a film went on to like — so it can only ever recommend things that *look*
similar, never "people who liked this also liked…". It also treats every tag as
equally trustworthy, with no weighting by the user who applied it, and the
popularity-unweighted similarity can surface an obscure title above an obvious
one.

A **rating-prediction** model (matrix factorisation, or ALS) answers a different
and harder question — predicting what rating a user would give — and would be
the natural next step. Note this repo is a recommender, not a "prediction" model
in that sense.
