# Postdoc Selection Simulations - Public Clean Notebook

## Overview
Created a comprehensive Jupyter notebook (`postdoc_selection_simulations_public_clean.ipynb`) that produces all figures and tables for the blog post on postdoc market stochasticity.

## Notebook Structure

### Setup Section
- Imports all required libraries
- Adds `top-k-mallows/` to sys.path for Mallows distribution sampling
- Imports all functions from `postdoc_market.py` module
- Sets random seed for reproducibility

### Analysis Sections (14 markdown + 15 code cells)

1. **Skill Distribution Visualization**
   - Histogram comparing student (N(0,1)) vs postdoc (N(1.3, 0.8)) skills
   - Output: `skill_distribution.png`

2. **Prestige Distribution**
   - Bar chart of three-tier position structure
   - Society of Fellows (6), Prize Fellowships (50), Standard Postdocs (200)
   - Output: `prestige_histogram.png`

3. **Single Market Simulation**
   - Runs one full market with verbose=1 output
   - Shows round-by-round summaries
   - Uses: n_students=750, n_postdocs=100, stochasticity=0.74

4. **Rank Matching Scatter Plot**
   - Position rank vs applicant rank for hired candidates
   - Students and postdocs marked differently
   - Perfect match line shows diagonal
   - Output: `scatter_rank_vs_rank.png`

5. **Acceptance Rounds Scatter Plot**
   - Position rank vs applicant rank, colored by acceptance round
   - Shows which round applicants got hired
   - Output: `scatter_by_round.png`

6. **Offers by Round Visualization**
   - Violin plots of applicant rank distribution by round (1, 2, 3)
   - Shows median and mean
   - Output: `offers_by_round.png`

7. **Grid Sweep Analysis** (Core Analysis)
   - Percentiles: [5, 10, 15, 20, 25, 35]
   - Applications: [2, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
   - N=100 runs per condition
   - Outputs success rates and standard deviations
   - Creates heatmap with annotations
   - Output: `heatmap_offers.png`

8. **Calibration Analysis**
   - Tests phi values from 0.1 to 1.0 (19 values)
   - Compares simulated offer distributions vs Princeton cohort data (y_obs)
   - Uses chi-squared error metric
   - Identifies best fit: phi ≈ 0.74
   - Output: `calibration_phi.png`

9. **Injection Table**
   - Shows applications needed for 50%, 75%, 90% success rates
   - For each percentile in grid
   - Text output (printed to notebook)

10. **What-If Scenarios (2×2 grid)**
    - Baseline: 750 students, 100 postdocs, splits=[6,50,200]
    - Half Positions: splits=[3,25,100]
    - Double Students: 1500 students
    - Double Positions: splits=[12,100,400]
    - Output: `whatif_scenarios.png`

11. **Stochasticity Sensitivity (1×2 grid)**
    - Low stochasticity: phi=0.05 (near-perfect selection)
    - High stochasticity: phi=0.95 (high randomness)
    - Output: `whatif_stochasticity.png`

12. **Final Verification**
    - Checks that all 9 output files exist
    - Prints summary statistics

## Key Configuration Parameters

All simulations use:
- **n_students**: 750 (NOT the default 1000)
- **n_postdocs**: 100 (NOT the default 300)
- **splits**: np.array([6, 50, 200]) (NOT the default [6, 50, 200])
- **stochasticity**: 0.74 (calibrated value)
- **matplotlib backend**: Agg (non-interactive)
- **DPI**: 150 (for blog publication)

## Output Files

All saved to `/sessions/peaceful-modest-bell/mnt/postdoc_market_simulation/`:

1. `skill_distribution.png` - Skill distributions
2. `prestige_histogram.png` - Position tier structure
3. `scatter_rank_vs_rank.png` - Rank matching
4. `scatter_by_round.png` - By round coloring
5. `offers_by_round.png` - Violin/box plots
6. `heatmap_offers.png` - Grid sweep results (core analysis)
7. `calibration_phi.png` - Phi fitting curve
8. `whatif_scenarios.png` - Policy change effects
9. `whatif_stochasticity.png` - Randomness comparison

## Usage

```python
# Run in Jupyter:
jupyter notebook postdoc_selection_simulations_public_clean.ipynb

# Or convert to HTML/PDF:
jupyter nbconvert --to html postdoc_selection_simulations_public_clean.ipynb
```

## Dependencies

- numpy
- matplotlib
- pandas
- postdoc_market.py (module in same directory)
- top-k-mallows/ (subdirectory with Mallows sampling)

## Notes

- The notebook is self-contained and can be run end-to-end
- Grid sweep (section 7) is the most computationally expensive part (~10-20 min)
- Calibration (section 8) tests 19 phi values (~5-10 min)
- All other sections complete in < 1 minute
- Total runtime: approximately 20-40 minutes on standard hardware
