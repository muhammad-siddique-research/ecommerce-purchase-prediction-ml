# What to do after obtaining `experiment_outputs.zip`

1. **Inspect and confirm findings.** Decision Tree F1=0.643 is highest; compare with the majority baseline accuracy=0.845 and 0 recall. Consider the large PageValues sensitivity result (0.643 -> 0.161).
2. **Verify the executed notebook.** The notebook in `notebooks/` has been restored from the user's earlier executed Colab upload, preserving its saved outputs. Review its figures, metrics, and code cells before submitting it as coursework.
3. **Verify EDA outputs.** Confirm dataset dimensions (12,330 x 17), missing-value check, duplicate-like feature rows, and the browsing-duration distribution. Your ZIP contains figures and metrics, not the text output of every notebook cell.
4. **Review `reports/research_report.md` or `.pdf`.** Enter your author name, confirm all factual claims against your notebook, edit the reflections in your own voice, and comply with your course's AI-assistance guidelines.
5. **Review `README.md` and attribution.** Replace the author placeholder in CITATION.cff and LICENSE; verify whether your institution permits publication.
6. **Review before publishing.** The project is organized in the existing GitHub repository. Keep it private until your author placeholders are completed, the report is reviewed, and your course permits public release. When ready, set the repository description and link it from your profile.
7. **Before submitting:** rerun the notebook from a fresh Colab runtime, check that figures render, and download the final executed notebook and reviewed PDF.

No measured model results have been fabricated. The CSVs and PNGs are copies of the file you uploaded.
