---
layout: center
---
# Classification Metrics

---

# Confusion Matrix

#### Consider 4 class classification problem:
<br>
<div class="grid grid-cols-[3fr_4fr] gap-15">
<div>
  <figure>
    <img src="/confusion_matrix_1.png" style="width: 290px !important;">
  </figure>
</div>
<div>

#### In the matrix:
* Value - number of objects of $j$-th class that were predicted as $i$-th class

* Diagonal elements are <span style="color:#82B366">**true**</span> classifications
* Off-diagonal elements are <span style="color:#B85450">**false**</span> classifications
</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/confusion_matrix_2.png" style="width: 390px !important;">
  </figure>
</div>
<div>


</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/accuracy.png" style="width: 390px !important;">
  </figure>
</div>
<div>

#### Quality metrics:
  <figure>
    <img src="/accuracy_f.png" style="height: 60px !important;">
  </figure>
</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/tpr.png" style="width: 390px !important;">
  </figure>
</div>
<div>

#### Quality metrics:
  <figure>
    <img src="/accuracy_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/tpr_f.png" style="height: 60px !important;">
  </figure>
</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/fpr.png" style="width: 390px !important;">
  </figure>
</div>
<div>

#### Quality metrics:
  <figure>
    <img src="/accuracy_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/tpr_f_clean.png" style="height: 60px !important;">
  </figure>
    <figure>
    <img src="/fpr_f.png" style="height: 60px !important;">
  </figure>
</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/pr.png" style="width: 390px !important;">
  </figure>
</div>
<div>

#### Quality metrics:
  <figure>
    <img src="/accuracy_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/tpr_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/fpr_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/pr_f.png" style="height: 60px !important;">
  </figure>
</div>
</div>

---

# Binary Classification

#### Consider a binary classification problem with classes **+** and **-**
<br>
<div class="grid grid-cols-[3fr_4fr] gap-30">
<div>
  <figure>
    <img src="/confusion_matrix_2.png" style="width: 390px !important;">
  </figure>
</div>
<div>

#### Quality metrics:
  <figure>
    <img src="/accuracy_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/tpr_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/fpr_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/pr_f_clean.png" style="height: 60px !important;">
  </figure>
  <figure>
    <img src="/F1_f.png" style="height: 60px !important;">
  </figure>
</div>
</div>

---

# Predicting Continuous Values

#### Many classification algorithms deal with continuous evaluation function
<br>
<div class="grid grid-cols-[2fr_2fr] gap-20">
<div>
    <figure>
    <img src="/continuous_classification_1.png" style="height: 250px !important">
    <br>
    <figcaption style="font-size: 16px;"><center>Prediction of the classification model</center>
    </figcaption>
  </figure>
</div>
<div>
    <figure>
    <img src="/continuous_classification_2.png" style="height: 250px !important">
    <br>
    <figcaption style="font-size: 16px;"><center>The value of the evaluation function in the data</center>
    </figcaption>
  </figure>
</div>
</div>

---

# ROC Curve

#### Receiver Operating Characteristic = **TPR** as a function of **FPR**
<div class="grid grid-cols-[5fr_4fr]">
<div>
<br>
  <figure>
    <img src="/ROC_curve.png" style="width: 500px !important">
  </figure>
</div>
<div>

* All thresholds are used to generate ROC
* Useful for comparing different models without focusing on any threshold
* Area Under Curve (AUC) summarized the ROC into a single value
  * Model comparison can be automated
* $\mathrm{AUC} \in [0, 1]$ with max value when ROC "hugs" top left corner
* If ROC follows "no information" line then predictors are unrelated to probability of interest
</div>
</div>

[Interactive demo](http://arogozhnikov.github.io/2015/10/05/roc-curve.html)

---

# Precision-Recall Curve

  <figure>
    <img src="/Precision_Recall_curve.png" style="width: 500px !important">
  </figure>

#### The Precision-Recall curve shows the trade-off between Precision and Recall for different thresholds. A large area under the curve indicates both a high Precision value (low false positive rate) and a high Recall value (low false negative rate)

---

# Confusion Matrix. Eg. Default Dataset

* **Diagonal**: correct predictions
* **Off-diagonal**: misclassified & needs attention
* **Accuracy rate** (score) is $(9644+81)/10000 = 97.25\%$ with error rate of $2.75\%$
  * A **null classifier** tags everyone as non-defaulters produces $333/10000 = 3.33\%$ error rate. Impressive?!?
  * Accuracy is a poor metric for **imbalanced classes**
    * Imbalanced because one class is much larger than another

<div class="grid grid-cols-[5fr_4fr]">
<div>

* "<span style="color:#82B366">**Positives**</span>" = <span style="color:#82B366">defaulters</span>, the class of interest to us
* “<span style="color:#B85450">**Negatives**</span>” = the other class (<span style="color:#B85450">non-defaulters</span>)
* We need class-specific metrics of performance

</div>
<div>
<br>
<figure>
  <img src="/Default_confusion_matrix.png" style="width: 655px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Table. 4.4</a>
  </figcaption>
</figure>
</div>
</div>

---

# Specificity & Sensitivity (Recall). Eg. Default Dataset

* **Specificity** or $\%$ of correctly identified true negatives
  * Out of $9667$ actual <span style="color:#B85450">non-defaults</span> $9644$ were correctly predicted, $99.8\%$
* **Sensitivity** (or recall) of $\%$ of correctly identified true positives
  * Out of $333$ actual defaults $81$ were correctly predicted, $24.3\%$
  * Recall is poor because Bayes LDA is maximizing accuracy
* A bank may have greater concern with <span style="color:#82B366">defaulters</span>
  * <span style="color:#82B366">Defaulters</span> are costlier than <span style="color:#B85450">non-defaulters</span>
  * Bank may want to identify more <span style="color:#82B366">defaulters</span> at a cost of misclassifying <span style="color:#B85450">non-defaulters</span>
* So, we can lower the **threshold** of identifying positives

<div class="grid grid-cols-[5fr_4fr]">
<div>

* Eg. $\mathbb{P} [\color{#82B366}\mathrm{default}\color{#006} | X] > 0.2$ are classified as <span style="color:#82B366">defaulters</span>
  * Then, we predict $430$ <span style="color:#82B366">defaulters</span>
    * includes $138$ of true positives
    * recall $↑41.4\%$ and accuracy $↓96.3\%$

</div>
<div>
<figure>
  <img src="/Default_confusion_matrix.png" style="width: 655px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Table. 4.4</a>
  </figcaption>
</figure>
</div>
</div>

---

# Type I & II Error Rates

* False positive (FP): a <span style="color:#B85450">non-defaulter</span> labeled as <span style="color:#82B366">defaulter</span>
* False negative (FN): a <span style="color:#82B366">defaulter</span> is labeled as <span style="color:#B85450">non-defaulter</span>
  * $\mathrm{FP~rate} := \mathbb{P}[\mathrm{Type~I~error}]$
  * $\mathrm{FN~rate} := \mathbb{P}[\mathrm{Type~II~error}]$
  * $\mathrm{TPR}= 1 - \mathrm{FNR}$
* Lower threshold "flags" more <span style="color:#82B366">defaulters</span> (lower FNR) and <span style="color:#B85450">non-defaulters</span> (higher FPR)

<br>
<figure>
  <img src="/ISLP/ISLP_figure_4.7.png" style="width: 455px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px; position: relative; left: 400px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.7</a>
  </figcaption>
</figure>