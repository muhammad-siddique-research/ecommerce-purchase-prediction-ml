# Same-day publication checklist

## Stage 1 — Run and verify
- [ ] Upload `notebooks/Algorithm_Evaluation_Extravaganza_UCI.ipynb` to Colab.
- [ ] Execute every cell successfully and confirm `X.shape == (12330, 17)`.
- [ ] Record actual missingness, duplicates, and purchasing prevalence.
- [x] Save CV scores, held-out metrics, confusion matrices, ROC, PR, and ablation outputs.

## Stage 2 — Report
- [x] Prepare a results-grounded draft (`reports/research_report.md`); [ ] personally review and add your name.
- [x] Interpret baseline and false-negative/false-positive trade-offs in draft; [ ] review interpretation.
- [x] Describe the timing limitation of session features and `PageValues`.
- [ ] Cite UCI correctly and distinguish association from causation.
- [x] Export results-grounded report draft as PDF; [ ] personalize and confirm with Colab outputs.

## Stage 3 — Publish
- [ ] Replace author placeholders in `LICENSE` and `CITATION.cff`.
- [ ] Make sure no confidential identifiers, API keys, or student records are committed.
- [ ] Confirm your course permits publishing assignment-related code.
- [ ] Upload repository via GitHub web UI or `git push`.
- [x] Replace README placeholder results with verified Colab-export metrics.
- [ ] Pin the repository on your GitHub profile.

## Git commands (after creating an empty remote repository)

```bash
git init
git add .
git commit -m "Add reproducible online purchase classification benchmark"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ecommerce-purchase-prediction-ml.git
git push -u origin main
```

If the remote GitHub repository was created with a README, clone it first and copy the files into the cloned folder instead of pushing unrelated local history.
