# Postdoc Market Simulation

How many postdoc applications should you send? I built this to try to answer that question quantitatively, after getting frustrated by the standard advice of "it's just very stochastic".

The full write-up is in [this blog post](https://astrockragh.github.io/post/postdoc_experience2/). The short version: I model the postdoc job market as a multi-round matching process where committees rank applicants with noise (via the [Mallows model](https://en.wikipedia.org/wiki/Mallows_model)), calibrate the noise against real Princeton cohort data, and then ask how many applications a candidate at a given skill percentile needs to send to have an X% chance of getting at least one offer.

## Try it yourself

The easiest way to play with this is the **[Google Colab notebook](https://colab.research.google.com/github/astrockragh/postdoc_market_simulation/blob/main/postdoc_market_colab.ipynb)**. You put in your estimated percentile range and how many applications you'd consider sending, and it runs the market simulation and gives you back your odds.

## Repo structure

- `postdoc_market.py` — the simulation module. Contains the `ApplicantCohort` class, grid sweep functions, and plotting utilities.
- `postdoc_market_colab.ipynb` — interactive notebook for Colab. This is what you want if you just want to run it.
- `postdoc_selection_simulations_public_clean.ipynb` — the analysis notebook that produces all the figures in the blog post.
- `top-k-mallows/` — submodule from [ekhiru/top-k-mallows](https://github.com/ekhiru/top-k-mallows), used for sampling from the Mallows distribution.
- `figs/` — all figures used in the blog post.

## Quick start (local)

```bash
git clone --recurse-submodules https://github.com/astrockragh/postdoc_market_simulation.git
cd postdoc_market_simulation
pip install numpy scipy matplotlib
```

```python
from postdoc_market import ApplicantCohort

cohort = ApplicantCohort(n_students=1000, n_postdocs=300, stochasticity=0.74)
cohort.rank_applicants()
cohort.inject_candidate(student_percentile=15, n_applications=20)
cohort.run_market()
print(cohort.get_injected_result())
```

Note: `n_students` and `n_postdocs` are total cohort sizes. The module applies application fractions internally (75% of students, 1/3 of postdocs), giving ~850 actual market participants.

## The main result

| Percentile | P=50% | P=75% | P=90% | P=95% | P=99% |
| --- | --- | --- | --- | --- | --- |
| Top 5% | 2 | 5 | 8 | 8 | 15 |
| Top 10% | 5 | 8 | 10 | 20 | 25 |
| Top 15% | 8 | 15 | 25 | 30 | 40 |
| Top 20% | 20 | 40 | 60 | 60 | 80 |
| Top 25% | 60 | 80 | 150 | 150 | 200 |
| Top 35% | >200 | >200 | >200 | >200 | >200 |

Read: "if you're roughly top 15%, send ~25 applications for a 90% chance of at least one offer."

## Citation

If you use this in your own work, a link to the [blog post](https://astrockragh.github.io/post/postdoc_experience2/) or this repo would be appreciated.
