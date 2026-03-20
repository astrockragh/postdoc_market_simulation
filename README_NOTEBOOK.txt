================================================================================
POSTDOC SELECTION SIMULATIONS - PUBLIC CLEAN NOTEBOOK
================================================================================

CREATED: 2026-03-16
FILE: postdoc_selection_simulations_public_clean.ipynb
LOCATION: /sessions/peaceful-modest-bell/mnt/postdoc_market_simulation/

================================================================================
OVERVIEW
================================================================================

A comprehensive Jupyter notebook that produces all figures and tables for the
blog post on postdoc market stochasticity. The notebook simulates a realistic
postdoctoral job market with two-phase selection (Mallows-distributed ranking
followed by applicant choice) and explores how randomness affects outcomes.

================================================================================
STRUCTURE
================================================================================

The notebook has 29 cells total:
  - 14 markdown cells (explanatory text)
  - 15 code cells (analysis and visualization)

Sections:
  1. Setup and Imports
  2. Skill Distribution Visualization
  3. Prestige Distribution
  4. Single Market Simulation (verbose output)
  5. Rank Matching Scatter Plot
  6. Acceptance Rounds by Round
  7. Offers Distribution by Round (violin plots)
  8. Grid Sweep Analysis (CORE: percentiles × applications)
  9. Calibration (fit phi parameter)
  10. Injection Table (applications needed for target rates)
  11. What-If Scenarios (policy changes)
  12. Stochasticity Sensitivity (low vs high phi)
  13. Final Verification

================================================================================
KEY PARAMETERS
================================================================================

Standard Cohort Configuration:
  - n_students: 750
  - n_postdocs: 100
  - Position splits: [6, 50, 200] (Society of Fellows, Prize, Standard)
  - Stochasticity (phi): 0.74 (calibrated value)
  - Offer rounds: 15

Grid Sweep Configuration:
  - Skill percentiles: [5, 10, 15, 20, 25, 35]
  - Applications tested: [2, 5, 10, 15, 20, 25, 30, 40, 60, 80, 100, 150, 200]
  - Runs per condition: 100
  - Total simulations: 6 × 13 × 100 = 7,800 markets

Calibration Configuration:
  - Phi values tested: 0.1 to 1.0 (19 values)
  - Metric: chi-squared against Princeton cohort data
  - Best fit: phi ≈ 0.74

Output Settings:
  - Backend: Agg (non-interactive)
  - Resolution: 150 DPI
  - Format: PNG (suitable for blog publication)

================================================================================
OUTPUT FILES
================================================================================

The notebook saves 9 PNG figures to the working directory:

1. skill_distribution.png
   - Histogram: Student vs Postdoc skill distributions
   - Shows skill advantage of postdocs entering market

2. prestige_histogram.png
   - Bar chart: Three-tier position structure
   - Society of Fellows (6), Prize (50), Standard (200) = 256 total

3. scatter_rank_vs_rank.png
   - Scatter plot: Applicant rank vs Position rank
   - Students and postdocs marked differently
   - Diagonal line shows perfect matching

4. scatter_by_round.png
   - Scatter plot: Colored by acceptance round
   - Shows temporal progression of hiring

5. offers_by_round.png
   - Violin plots: Applicant rank distribution by round (Rounds 1, 2, 3)
   - Shows how early rounds get better-ranked applicants

6. heatmap_offers.png
   - Heatmap: Success rates across percentiles × applications
   - CORE ANALYSIS: Shows how many apps needed for each skill level
   - Color-coded with values (0.00-1.00)

7. calibration_phi.png
   - Line plot: Chi-squared error vs phi values
   - Shows phi=0.74 is optimal fit to real data

8. whatif_scenarios.png
   - 2×2 scatter plots: Four policy scenarios
   - Baseline, Half Positions, Double Students, Double Positions
   - Shows impact of market structure changes

9. whatif_stochasticity.png
   - 1×2 scatter plots: Low vs High randomness
   - phi=0.05 (nearly perfect selection) vs phi=0.95 (random)
   - Demonstrates importance of imperfect information

================================================================================
DEPENDENCIES
================================================================================

Required Python Packages:
  - numpy
  - matplotlib
  - pandas

Required Local Files:
  - postdoc_market.py (main simulation module)
  - top-k-mallows/ (directory with Mallows sampling library)

The notebook automatically:
  - Adds top-k-mallows/ to sys.path
  - Imports all necessary functions from postdoc_market.py
  - Sets matplotlib backend to Agg
  - Sets random seed for reproducibility

================================================================================
RUNTIME
================================================================================

Estimated execution time on standard hardware:

  Section                    Time
  ------                     ----
  Setup & Imports            < 1 sec
  Skill Distribution         < 1 sec
  Prestige Distribution      < 1 sec
  Single Market              < 1 sec
  Rank Matching Plots        < 1 sec
  Round Analysis             < 1 sec
  Grid Sweep Analysis        10-20 min (7,800 simulations)
  Calibration Analysis       5-10 min (19 phi values × markets)
  Injection Table            < 1 sec
  What-If Scenarios          2-5 min (4 single markets)
  Stochasticity Analysis     1-2 min (2 single markets)
  Verification               < 1 sec
  
  TOTAL RUNTIME:             20-40 minutes

The grid sweep and calibration sections dominate runtime. Other sections
are quick and suitable for interactive use.

================================================================================
USAGE
================================================================================

Run in Jupyter:
  $ jupyter notebook postdoc_selection_simulations_public_clean.ipynb

Convert to HTML:
  $ jupyter nbconvert --to html postdoc_selection_simulations_public_clean.ipynb

Convert to PDF:
  $ jupyter nbconvert --to pdf postdoc_selection_simulations_public_clean.ipynb

Run in VSCode:
  - Open the .ipynb file
  - Click "Run All Cells" in the Jupyter extension

Run programmatically:
  $ jupyter nbconvert --to notebook --execute --inplace \
    postdoc_selection_simulations_public_clean.ipynb

================================================================================
NOTEBOOK QUALITY CHECKS
================================================================================

✓ Valid JSON format (verified with json.tool)
✓ All required cells present (29 cells)
✓ Proper markdown formatting with headers
✓ Code imports verified against postdoc_market.py API
✓ Output file paths consistent with working directory
✓ DPI and backend settings explicitly configured
✓ Random seed set for reproducibility
✓ All numpy/matplotlib operations tested in main module

================================================================================
NOTES FOR BLOG POST
================================================================================

The notebook is designed as a standalone, reproducible analysis that:

1. Explains each analysis section with markdown cells
2. Produces publication-quality figures (150 DPI)
3. Shows all intermediate calculations
4. Includes summary statistics and interpretation
5. Demonstrates both single-run and multi-run analysis
6. Validates findings through calibration

All figures use consistent color schemes and are ready for blog publication.
The notebook can be shared publicly with reproducible results.

================================================================================
