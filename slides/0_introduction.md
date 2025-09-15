---
zoom: 0.9
---

# Changes in ML Course

* Hackathon
    * In addition to the 15% (7.5% for ICEF) weight assigned to the grading formula, successfully completing the hackathon gives you the following bonuses:
        * 🥇place: you will receive 10 points for any two <span v-mark="{ at: 1, color: 'yellow', type: 'highlight' }">categories of</span> control elements by choice (except midterm and exam)
        * 🥈and 🥉places: you will receive 10 points for one <span v-mark="{ at: 2, color: 'yellow', type: 'highlight' }">category of</span> control element by choice (except midterm and exam)
        * Trophy prizes or merch are also expected
* Quizzes
    * The results of two (of 12) worst in-seminar quizzes will be discarded
    * Retakes of missing quizzes are available for valid reason
        * Once in December
    
---

# Examples of Classification

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
  * **Logistic regression**
  * **Linear Discriminant Analysis** (LDA)
  * Suitable for both regression and classification:
    * **Classification And Regression Tree** (CART)
    * **K-Nearest Neighbor** (KNN)
    * **Support Vector Machine** (SVM) and **Classifier** (SVC)
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
<figure>
  <img src="/ISLP/ISLP_figure_4.1.png" style="width: 585px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html#page=146">ISLP Fig. 4.1</a>
  </figcaption>
</figure>

<!--
* Do you see associacions in Y vs Balance, Y vs Income?
* What can you tell about distributions of the plotted variables?
* Notice that plots carry all needed descriptive information to “read” them.
  *Typically, titles are omitted for plots that have captions.
-->
