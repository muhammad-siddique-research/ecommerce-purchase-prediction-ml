# E-Commerce Purchase Intention Prediction: Comparative Classification Evaluation

**Researcher:** [YOUR PUBLICATION NAME]  
**Date:** [RUN DATE]  
**Code repository:** [URL]  
**Dataset:** Sakar and Kastro (2018), UCI DOI 10.24432/C5F88Q

> **Reproducibility disclosure:** This is a writing template. Replace all `[PLACEHOLDER]` fields with actual experimental outputs. Do not submit these prompts as measured findings.

## Abstract (150–200 words)

This study compares three supervised learning classifiers for predicting completed purchasing sessions in an e-commerce setting. Using 12,330 session records and a binary `Revenue` outcome from the UCI Online Shoppers Purchasing Intention Dataset, the study evaluates Decision Tree, K-Nearest Neighbors, and Gaussian Naive Bayes models against a majority-class baseline. Models are tuned using stratified cross-validation on training records and assessed using an untouched holdout set. Key criteria include precision, recall, F1-score, ROC-AUC, and average precision. **[INSERT: verified comparative results]**. The analysis discusses the implications of class imbalance, session-level feature availability, and the limitations of interpreting predictive associations as causes of purchasing behavior.

## 1. Introduction and research question

E-commerce clickstream indicators may help identify shopping sessions associated with purchases. However, a model that reports high accuracy may be uninformative in the presence of substantial class imbalance. This research asks: **Which of three conventional classification algorithms most effectively identifies purchasing sessions when evaluated on an identical holdout set?** Performance is defined using decision-relevant metrics rather than classification accuracy alone.

## 2. Dataset and exploratory analysis

The UCI Online Shoppers Purchasing Intention Dataset contains 12,330 sessions, 17 features, and the binary target `Revenue` (Sakar & Kastro, 2018). **[INSERT: observed missingness, extreme values, class counts, duplicate-pattern discussion, plots]**. Session-summary fields should not be misrepresented as pre-session predictors.

## 3. Preprocessing and experimental design

Categorical features were one-hot encoded; numerical features were median-imputed where needed and standardized. All estimated preprocessing parameters were fit exclusively within each training fold via scikit-learn pipelines. A stratified 80/20 split with a fixed random seed separated training/tuning from final evaluation. A three-fold stratified cross-validation grid search selected each algorithm's hyperparameters using F1-score. **[INSERT: runtime versions, selected hyperparameters, train/test counts]**.

A Decision Tree supports nonlinear partitions and transparent rules; KNN provides a distance-based alternative that requires appropriate scaling; Gaussian Naive Bayes offers a probabilistic benchmark with strong conditional-distribution assumptions. A majority-class classifier anchors the comparison.

## 4. Results

**Replace this table with real outputs from `results/test_model_metrics.csv`.**

| Algorithm | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Decision Tree | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| KNN | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |
| Gaussian Naive Bayes | [ ] | [ ] | [ ] | [ ] | [ ] | [ ] |

**Figure 1.** [INSERT: class distribution]  
**Figure 2.** [INSERT: confusion matrices]  
**Figure 3.** [INSERT: ROC comparison]  
**Figure 4.** [INSERT: precision–recall comparison]

**Training CV selection:** [INSERT best cross-validated F1 and parameter selection; explain which algorithm performed best according to the pre-specified training criterion.]  
**Holdout interpretation:** [INSERT careful performance comparison; note that different performance metrics may favor different models.]

## 5. Robustness and limitations

**PageValues ablation:** [INSERT results from `results/pagevalues_ablation.csv`; discuss changes without asserting a leakage mechanism is proven.]  
**Limitations:** Single observational dataset; incomplete metadata on session time ordering; imbalance; no causal identification; potential distribution shift; uncertainty not quantified by a single holdout split. Feature availability at a prospective decision point should be verified before claiming deployment readiness.

## 6. Reflection and lessons learned

[INSERT: a specific preprocessing decision you changed; an unexpected failure or result; how the baseline shaped interpretation; what improvement you would prioritize next.]

## 7. Conclusion

[INSERT: direct research-question answer grounded in observed results. Avoid claiming business profit or causal impact without intervention data.]

## References (APA 7)

Sakar, C., & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset* [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q

Sakar, C. O., Polat, S. O., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications, 31*, 6893–6908. https://doi.org/10.1007/s00521-018-3523-0

scikit-learn developers. (n.d.). *Common pitfalls and recommended practices*. https://scikit-learn.org/stable/common_pitfalls.html
