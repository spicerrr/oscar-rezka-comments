# Methodology

## Research question

How do evaluative valence, expressed emotions, and targets of praise and
criticism differ in HdRezka comments to selected films connected with the
2026 Academy Awards season?

## Unit and variables

The unit of analysis is one comment.

Main variables:

- relevance to the film;
- valence: positive, negative, mixed, neutral, unclear;
- explicitly expressed emotions;
- praise targets and criticism targets;
- comparison and intertextuality;
- Oscar stance;
- rhetorical modes;
- exploratory interpretive frames.

The analysis describes expressed textual reactions. It does not infer a
commenter's internal psychological state.

## Sampling

The full collected corpus contains 20,660 comments for 28 films. A
film-stratified sample was created:

- all comments were retained if a film had at most 120 comments;
- otherwise 120 comments were selected with a fixed random seed;
- the resulting sample contains 2,555 comments.

For corpus-level proportions, film weights can be calculated as:

`full number of comments for the film / sampled comments for the film`.

## Annotation

The final annotation pipeline uses `qwen3:8b` locally through Ollama, a compact
structured-output schema, deterministic post-processing rules, checkpointed
saving and missing-ID retries.

## Statistical analysis

Core outputs include frequencies, shares, film-by-category contingency
tables, chi-square tests, Cramer's V, and descriptive comparisons of likes.
Detailed frames are treated as exploratory because their coding stability is
lower than the stability of valence.
