---
layout: center
---
# Quadratic Discrimintant Analysis

---

#  Quadratic Discrimintant Analysis (QDA)

* LDA assumes $X |_Y \sim N (\mu_Y, \Sigma)$
* QDA assumes $X |_Y \sim N (\mu_Y, \Sigma_{\color{red}Y}\color{#006})$, where $\Sigma_Y \in \R^{p\times p}$ matrix, $Y = 1:K$
* The QDA discriminant function is then
$\delta_k(x) := - \frac{1}{2} \cdot
\color{grey}\underbrace{\color{#006} \tilde{x}_k^T \Sigma_k^{-1} \tilde{x}_k}_{\mathrm{quadratic~} x \mathrm{~term}}
\color{#006} -
\color{grey}\underbrace{\color{#006} \frac{1}{2}\log\lvert\Sigma_k\rvert + \log\pi_k}_{\mathrm{constant~term}}
$
* where $\tilde{x}_k := x - \mu_k$
* Compare this to LDA's: $\delta_k(x) := \color{grey}\underbrace{\color{#006} x^T \Sigma_k^{-1} \mu_k}_{\mathrm{linear~} x \mathrm{~term}} \color{#006} - \color{grey}\underbrace{\color{#006} \frac{1}{2}\mu_k\Sigma_k^{-1}\mu_k + \log\pi_k}_{\mathrm{constant~term}}$
  * Here we only compute a single inverse
* Classification is the same: the class $k$ with highest $\hat\delta_k(x)$ wins $x$
* Estimation: $\Sigma_k$ is symmetric requiring $\frac{1}{2}p(p+1)$ estimates $k$ times
  * $50$ predictors requires $1275$ covariance estimates in $\Sigma_1, ..., \Sigma_k$

---

# LDA vs QDA

* LDA is less flexible and can only fit linear decision boundaries
* LDA avoids overfitting, tends to have lower variance
  * Works better for fewer observations
  * Computational performance is linear with the number of features
* QDA works better for larger $n$, when risk of overfitting is low
  * Note $\Sigma_k$ does not increase with $n$ (computing $\Sigma_k^{-1}$ is $\mathcal{O}(n^3)$, i.e. expensive)
  * QDA has quadratic decision boundary

<div class="grid grid-cols-[5fr_5fr]">
<div>

* LDA outperforms, if Bayes is linear
* QDA outperforms, if Bayes is quadratic

</div>
<div>
<figure>
  <img src="/ISLP/ISLP_figure_4.9.png" style="width: 695px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.9</a>
  </figcaption>
</figure>
</div>
</div>

---

# LDA vs Logistic Regression

* Both have log odds as linear functions of $x$
* LDA for $k = 2$: $~\log \big( \frac{p_1(x)}{p_2(x)}\big) = c_0 + c_1 x$
  * $c_i$ are estimated MLEs of parameter of Gaussian PDF of $X|Y$
* Logistic regression for $k = 2$: $~\log \big( \frac{p_1(x)}{p_2(x)}\big) = \beta_0 + beta_1 x$
  * $\beta_i$ are estimated via MLE of regression parameters
* If Gaussian assumption is correct, LDA outperforms

---

# LDA vs KNN

* LDA is parametric
  * Requires estimation of $N_p (\mu_k, \Sigma_k)$
* KNN is non-parametric
  * No parameters to fit
  * No assumptions about the shape of the decision boundary
    * KNN outperforms with non-linear Bayes decision boundary
  * Cannot identify important variables