---
layout: center
---
# Linear Discriminant Analysis

---

# Bayes Theorem
<div class="grid grid-cols-[2fr_1fr]">
<div>

* 250 years old formula

$$
\color{grey}\underbrace{\color{#006} f(\theta | x)}_{\mathrm{posterior}} \color{#006} =
\color{grey}\underbrace{\color{#006} f(\theta)}_{\mathrm{prior}} \color{#006} \cdot
\color{grey}\underbrace{\color{#006} f(x | \theta)}_{\mathrm{likelihood}} \color{#006} \bigg/
\color{grey}\underbrace{\color{#006} f(x)}_{\mathrm{marginal}} \color{red} \propto
\color{grey}\underbrace{\color{#006} f(\theta) \cdot f_\theta(x)}_{\mathrm{unnormalized~ posterior}}
\color{#006},
$$
where the marginal density is a constant w.r.t. $\theta$
</div>
<div>
<figure>
  <img src="/Thomas_Bayes.gif" style="width: 135px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Thomas Bayes (1702—1761)
  </figcaption>
</figure>
</div>
</div>

* **Goal**: infer the posterior distribution of $\theta | x$ from the assumed prior $f$ and the likelihood $f_\theta$
  * It combines **prior** knowledge about $\theta$ with the current  **evidence** (data) to improve the posterior
  * Here $x$ is fixed and $\theta$ and $\theta | x$ are RVs
* If we know the density $f(\theta) \cdot f_\theta(x)$, which must integrate to $1$, then:
  * The constant of integration, $f(x)$, can be determined numerically
  * Likewise, we can drop all constants (w.r.t. $\theta$) in the likelihood and prior to simplify the expression

---

# Effect of Observed Sample
<div></div>

$$
\color{grey}\underbrace{\color{#006} f(\theta | x)}_{\mathrm{posterior}} \color{red} \propto
\color{grey}\underbrace{\color{#006} f(\theta) \cdot f_\theta(x)}_{\mathrm{prior} \cdot \mathrm{likelihood}}
$$

<div class="grid grid-cols-[1fr_1fr]">
<div>
<v-clicks depth="3">

* A prior is our guess about the posterior
  * We can make pedictions with no observation
    * In frequentist world, this is not possible
* With observed sample we update the posterior and improve predictions
* Posterior is a pointwise product
</v-clicks>
</div>
<div>
<figure>
  <img src="/prior2post_1-1.svg" style="width: 450px !important">
  <figcaption style="color:#b3b3b3ff; font-size: 11px">Image source:
    <a href="https://m-clark.github.io/bayesian-basics/example.html">https://m-clark.github.io/bayesian-basics/example.html</a>
  </figcaption>
</figure>
</div>
</div>

---

# Estimation of Posterior Distribution

* Goal: model **conditional distribution**, $Y | X$:<br>
$p_k(X) := \mathbb{P} [Y = k | X]$
  * Note: this probability is some function of $X$
* In logistic regression, we model it directly as $~p_k(X) = \frac{e^{X^\prime \beta}}{1+e^{X^\prime \beta}}$, where $X^\prime$ is a transpose of $X$
* Alternative method:
$$
\color{grey}\underbrace{\color{#006} p_k(X)}_{\mathrm{posterior}} \color{#006} =
\frac{
\color{grey}\overbrace{\color{#006} \mathbb{P}[Y = k]}^{\pi_k, \mathrm{prior}}
\color{#006} \cdot
\color{grey}\overbrace{\color{#006} \mathbb{P}[X | Y = k]}^{f_k(X)}
}{\mathbb{P}X}
= \frac{\pi_k \cdot f_k(X)}{\sum\limits_i \pi_i f_i(X)}
$$
* we estimate the class-specific distribution of $X$, $f_k(X)$

* Now, we have to estimate $\pi_k$, $f_k(X)$ for each class $k = 1:K$
  * $\pi_k$ is a **function of response value counts**, not the constant $\pi \approx 3.14159$

---

# Estimating a Prior Probability Function, $\pi_k$

* Assuming a **random sample of observations** from a population
  * $\pi_k$ - proportion of observations in class $k$
* Eg 1. Our sample has the following response values: $\{👽,👽,👽,\color{red}{🕷,🕷,🕷},\color{green}{🕸,🕸,🕸,🕸}\}$
$$\vec{\pi} := [\pi_👽, {\pi_{\color{red}{🕷}}}, \pi_{\color{green}{🕸}}] = \bigg[\frac{3}{10}, \frac{3}{10}, 1 - \pi_👽 - \pi_{\color{red}{🕷}}\bigg] = \bigg[\frac{3}{10}, \frac{3}{10}, \frac{4}{10}\bigg]$$

* Eg 2. We are trying to predict ad clicks with response label $Y \in \{\mathrm{Yes},\mathrm{No}\}$ (user clicked ad or not). The training set has $1000$ observations with clicks and $1\mathrm{M}$ without.

```python {all}{maxHeight:'110px'}
nYes, nNo = 1000, 1000000
print(f'prior for click = Yes class: {nYes/(nYes + nNo)}')
print(f'prior for click = No class: {nNo/(nYes + nNo)}')
prior for click = Yes class: 0.000999000999000999
prior for click = No class: 0.999000999000999
```

  * Note: without any other information we expect $99.9\%$ of users to ignore ads

---

# Estimating $f_k(X)$

<v-clicks depth="3">

* This on is tricky, unless we assume some distribution family
* In **Linear Discriminant Analysis (LDA)**, we assume $f_k \equiv N (\mu_k, \sigma_k^2)$ and need to estimate parameters of the Gaussian<br>
$f_k (x | \mu_k, \sigma_k) := \frac{1}{\sqrt{2 \pi}\sigma_k} \exp \big[ -\frac{(x - \mu_k)^2}{2 \sigma_k^2} \big]$
  * Note: for $K$ classes we assume $K$ Gaussian distributions
* The assumption $\sigma = \sigma_k, \forall k$ simplifies our posterior as:<br>
$p_k(x) = \frac{\pi_k \cdot f_k(X)}{\sum\limits_i \pi_i f_i(X)} = \frac{\pi_k \exp[-0.5 \sigma^{-2}(x - \mu_k)^2]}{\sum\limits_i \pi_i \exp[-0.5 \sigma^{-2}(x - \mu_i)^2]}$
* Try: take $\log$ of both sides and do some algebra to derive **discriminant function**<br>
$\delta_k (x | \mu_k, \sigma) := x \frac{\mu_k}{\sigma^2} - \frac{\mu_k^2}{2 \sigma^2} = \log \pi_k$
* The class with largest $\delta_k$ "wins" the observation $x$
  * i.e. $\delta_k$ determines the LDA class **decision boundaries** for classifying $x$

</v-clicks>

---

# Ex. Bayes Decision Boundary with Gaussian $X|Y$

* Here we have $X|_{Y = 1,2} \sim N (\mu_\gamma = \pm 1.25, \sigma^2 = 1)$
* The classification uncertainty arises from the overlap in densities
* Best dicision is to classify $x$ to the class $k$ with the **highest posterior density**, $p_k(x)$
  * LDA boundary approximates (theoretical) **Bayes decision boundary**

<br>
<figure>
  <img src="/ISLP/ISLP_figure_4.4.png" style="width: 655px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.4</a>
  </figcaption>
</figure>

---

# Examples of non/overlapping densities $X|Y$

* Eg 1. Average pixel brightness is $>100$ for cats and $<100$ for dogs
  * $\mathrm{supp~(X|cat)} \cap \mathrm{supp~(X|dog)} = \empty$
* Eg 2. $X|Y = k \sim \mathrm{Uniform}[k - 1, k]$, $k = 1,2,3$
  * So, we know that $X$ values in $[1,2]$ could come from $k=1$ only
  * What is the class of observation $X = 1.5$?
* Eg 3. However, $X|Y = k \sim \mathrm{Uniform}[k - 1, k + 1]$, $k = 1,2$ have overlapping supports

---

# Estimating Parameters in Discriminant Function, $\delta_k$
* Assume $\sigma = \sigma_k, \forall k$
* Then we need to estimate $2K + 1$ params and plug them into $\delta_k$ to derive
$$\hat{\delta}_k(x) := x \cdot 
\color{grey}\underbrace{\color{#006} \frac{\hat\mu_k}{\hat\sigma^2}}_{c_1}
\color{#006} +
\color{grey}\underbrace{\color{#006} \frac{-\hat\mu_k^2}{2 \hat\sigma^2} + \log \hat\pi_k}_{c_0}
\color{#006} =
c_0 + c_1 x
$$

* $\hat{\delta}_k(x)$ is a linear function of $x$ (i.e. linear in $x$)
* Parameters to estimate: $\mu_1, ..., \mu_k, \pi_1, ..., \pi_k, \sigma$
* $\hat\mu_k := \frac{1}{n_k} \sum_{i:y_i = k} x_i$, sample mean of $x_i$'s in class $k$
* $\hat\sigma^2 := \frac{1}{n-K} \sum_{k=1}^K \sum_{i:y_i = k} (x_i - \hat\mu_k)^2$, a sample variance of $x_i$'s in class $k$
* $n := \sum_k n_k$
* $n_k$ - number of observations in class $k$
* $\hat\pi_k := \frac{n_k}{n}$, a prior probability of class $k$

---

# LDA, $p > 1$

* We now assume RV $X$ has a **multivariate normal distribution** in each class<br>
$\begin{bmatrix} X_1 \\ \vdots \\  X_p \end{bmatrix} |_Y \sim N_p \Bigg( \mu_Y := \begin{bmatrix} \mu_{1Y} \\ \vdots \\  \mu_{pY} \end{bmatrix}, \Sigma_Y := {\begin{bmatrix} \hat\sigma_{11Y} & \ldots & \hat\sigma_{12Y} \\ \vdots & \ddots & \vdots \\ \hat\sigma_{21Y} & \ldots & \hat\sigma_{ppY}\end{bmatrix}} \Bigg)$, for each of $K$ classes
* $\sigma_{ijY} := \mathrm{Cov} (X_i, X_j | Y)$, $\sigma_{iiY} := \mathrm{Cov} (X_i, X_i | Y)$, $Y = 1, ..., K$
  * We need to estimate parameters for $K$ such distributions

<div class="grid grid-cols-[5fr_4fr]">
<div>

* With true parameters we have:
  * $\tilde{x} = x - \mu$
  * $c := (2\pi)^{-\frac{p}{2}} \lvert \Sigma \rvert^{-\frac{1}{2}}, \lvert \Sigma \rvert$ determinant
  * $f_k(x) := c \cdot \exp[-\frac{1}{2} \tilde{x}^T \Sigma^{-1}\tilde{x}]$
  * $\delta_k(x) := x^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k \Sigma^{-1} \mu_k + \log \pi_k$
* We classify x as $\argmax\limits_{\forall k} \delta_k(x)$

</div>
<div>
<br>
<figure>
  <img src="/ISLP/ISLP_figure_4.5.png" style="width: 655px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.5</a>
  </figcaption>
</figure>
</div>
</div>

---

# Eg. $p = 2, k = 3$

* $Y \in \{\color{#0A956C}{GREEN}\color{#006}, \color{#44A7DB}{BLUE}\color{#006}, \color{#E39B1E}{ORANGE} \color{#006}\}$
* $X|_{Y = \color{#0A956C}{GREEN}\color{#006}} \sim N_2 (\cdot, \cdot)$, $X|_{Y = \color{#44A7DB}{BLUE}\color{#006}} \sim N_2 (\cdot, \cdot)$, $X|_{Y = \color{#E39B1E}{ORANGE}\color{#006}} \sim N_2 (\cdot, \cdot)$
* The Bayes decision boundaries are $\{x | \delta_k(x) = \delta_\ell(x), k \neq \ell \}$
* LDA has linear decision boundaries and expresses $\delta_k(x)$ as a linear function of $x$

<figure>
  <img src="/ISLP/ISLP_figure_4.6.png" style="width: 620px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.6</a>
  </figcaption>
</figure>

---

# Classification Methods Demo

[Demo of different classification methods](http://www.ccom.ucsd.edu/~cdeotte/programs/classify.html)
