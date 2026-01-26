---
zoom: 0.75
---

# Lecture Summary

<v-clicks depth="2">

* **Classification** predicts categorical (qualitative) responses
  * Unlike regression which predicts quantitative values

* **Logistic Regression**
  * Models probability via sigmoid function: $p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$
  * For $K > 2$ classes: OVA, OVO, or softmax (multinomial) approaches

* **Linear Discriminant Analysis (LDA)**
  * Bayesian approach: models $P(X|Y)$ assuming Gaussian distribution
  * Assumes shared covariance $\Sigma$ across all classes → linear decision boundary

* **Quadratic Discriminant Analysis (QDA)**
  * Allows class-specific covariance $\Sigma_k$ → quadratic decision boundary
  * More flexible but requires more parameters

* **Classification Metrics**
  * Accuracy alone is insufficient (especially for imbalanced classes)
  * Use confusion matrix, precision, recall, F1-score, ROC-AUC

</v-clicks>

---
layout: end
hideInToc: true
---