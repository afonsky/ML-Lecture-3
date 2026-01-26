---
zoom: 0.85
---

# Current Homework Assignments

<script setup>
const timelineSource = `
@ [2026-01-19~2026-02-01T18:00] #green {🏆Your first Kaggle Competition} 💎Diamonds-26
* [2026-01-27T23:59] #green {🏆Your first Kaggle Competition} Interim deadline (Soft)
* [2026-01-28T18:00] #green {🏆Your first Kaggle Competition} Interim deadline (Hard)
* [2026-01-31T23:59] #green {🏆Your first Kaggle Competition} Final deadline (Soft)
* [2026-02-01T18:00] #green {🏆Your first Kaggle Competition} Final deadline (Hard)

@ [2026-01-27~2026-02-04T18:00] #red {🤖STACK+Maxima Tutorial} 🤖STACK+Maxima Tutorial
* [2026-02-03T23:59] #yellow {🤖STACK+Maxima Tutorial} Soft deadline
* [2026-02-04T18:00] #red {🤖STACK+Maxima Tutorial} Hard deadline
`
</script>

<ChronosTimeline :source="timelineSource" />
    
---

# Map of Estimators (models) in Sklearn
<style>
.slidev-layout {
  font-size: 1.1em
}
</style>

<div class="grid grid-cols-[5fr_2fr] gap-15]">
<div>
<figure>
  <img src="/ml_map.png" style="width: 680px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source: <a href="https://scikit-learn.org/1.4/tutorial/machine_learning_map/index.html">https://scikit-learn.org/1.4/tutorial/machine_learning_map/index.html</a>
  </figcaption>
</figure>
</div>
<div>


* **Supervised**:
  * Regression
  * **Classification**
* Unsupervised:
  * Clustering
  * Dimensionality reduction
* Other:
  * Reinforcement learning
  * Semi-supervised
</div>
</div>

---

# Classification
<v-clicks depth="3" every="1">

* Linear regression handles **quantitative** response
* What if response is **qualitative**: male/female, cat/dog/rat/ant, bad/neutral/good?
  * These are **classes**, **categories**, **levels**, **factors**
  * The appropriate models in supervised-learning are **classifiers**
* Examples of classifiers:
  * **Logistic regression** ← Today's lecture
  * **Linear Discriminant Analysis** (LDA) ← Today's lecture
  * **Quadratic Discriminant Analysis** (QDA) ← Today's lecture
  * Suitable for both regression and classification:
    * **Classification And Regression Tree** (CART) ← Future lectures
    * **K-Nearest Neighbor** (KNN) ← Future lectures
    * **Support Vector Machine** (SVM) and **Classifier** (SVC) ← Future lectures
</v-clicks>

---


# Classifier Comparison
* “[No free lunch](https://en.wikipedia.org/wiki/No_free_lunch_theorem)” (NFL) Theorem by [David Wolpert](https://en.wikipedia.org/wiki/David_Wolpert): “Any two optimization algorithms are equivalent when their performance is averaged across all possible problems”
  * In a world of uncertainty, “[All models are wrong, but some are useful](https://en.wikipedia.org/wiki/All_models_are_wrong)”, [George Box](https://en.wikipedia.org/wiki/George_E._P._Box)
    * Linear models: great for linear decision boundaries, but underperform in non-linear
    * Neural networks are general purpose models, but can underperform too

<figure>
  <img src="/sphx_glr_plot_classifier_comparison_001.png" style="width: 780px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 9px; position: relative; left: 350px;">Image source: <a href="https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html">https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html</a>
  </figcaption>
</figure>

---

# Note 1: Linear Separability

<br>
<br>

<center>
<figure>
  <img src="/02_03.png" style="width: 700px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 9px; position: relative; left: 350px;">Image source: <a href="https://github.com/rasbt/machine-learning-book">https://github.com/rasbt/machine-learning-book/</a>
  </figcaption>
</figure>
</center>

---

# Note 2: Under/Overfitting in Classification

<br>
<br>

<center>
<figure>
  <img src="/03_07.png" style="width: 700px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 9px; position: relative; left: 350px;">Image source: <a href="https://github.com/rasbt/machine-learning-book">https://github.com/rasbt/machine-learning-book/</a>
  </figcaption>
</figure>
</center>

---

# Eg. Credit Card Debt *Default* Dataset

* **Goal**: classify individuals into **default** and **non-default** categories of $Y$
  * **Features**: Income, Balance (owed to credit card issuer), Student (Yes/No)
  * **Response levels**: <span style="color:#5EA4D7">**default**</span>, <span style="color:#CA6320">**non-default**</span>

<br>
<center>
<figure>
  <img src="/ISLP/ISLP_figure_4.1.png" style="width: 585px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html#page=146">ISLP Fig. 4.1</a>
  </figcaption>
</figure>
</center>

<!--
* Do you see associacions in Y vs Balance, Y vs Income?
* What can you tell about distributions of the plotted variables?
* Notice that plots carry all needed descriptive information to “read” them.
  *Typically, titles are omitted for plots that have captions.
-->
