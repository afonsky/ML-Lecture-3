---
layout: center
---
# Logistic Regression

---

# Can We Classify with Linear Regression?

* Yes, but interpretation and model fit are suboptimal
* **Logistic regression** uses **logistic** function: "better" fit of extremes b/c of its S shape
  * We scale $Y$ to $[0, 1]$ interval and interpret it as probability
  * We can **threshold** the probability to derive class 0 (negative) or class 1 (positive)
  * We compute $p(\mathrm{balance}) := \mathbb{P}[\mathrm{default = Yes | balance}]$
* Linear regression cannot be interpreted as a probability

<figure>
  <img src="/ISLP/ISLP_figure_4.2.png" style="width: 585px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html#page=146">ISLP Fig. 4.2</a>
  </figcaption>
</figure>

---

# Response Variable’s Encodings

* Some models can't use `Cat` and `Dog` levels & require numeric encodings. Why?
* **Binary** response:
  * We can encode `{Cat, Dog}` with **dummy variables** as $\{0, 1\}$ or $\{1, 0\}$
    * The model fit and predictions do not change, but we predict the probability of class $1$
* **Ternary** (with 3 **levels**) or higher response:
  * **One-hot** vectors (or dummy variables) avoid imposing order or metric on $Y$
    * `{Cat, Dog, Rat}` can be encoded as $\{[0,0], [0,1], [1,0]\}$ or similar combination
    * This does not imply any distance or ordering between levels
  * **Numeric** encodings are better suited for ordinal variables (but this imposes metric)
    * `{Cold, Cool, Warm, Hot}` can be encoded with $\{0,1,2,3\}$
      * This tells the model to treat distances between levels as `Hot - Warm` ~ `Warm - Cool` ...
      * However, it may not always hold

---

# Logistic Model

<v-clicks depth="3" every="1">

* We want to model the probability $p(X):= \mathbb{P}[Y = 1 | X]$ as a linear function
  * $p(X) = \beta_0 + \beta_1 X + \epsilon$ is linear regession with aforementioned problems
* A suitable model is via **logistic** function:<br>$p(X) = \frac{e^{\beta_0 + \beta_1 X}}{1 + e^{\beta_0 + \beta_1 X}}$
* We can derive **odds ratio** as $\frac{p(X)}{1-p(X)} = e^{\beta_0 + \beta_1 X}$
* We can further construct a **log-odds** (or logit function):<br>$\log\bigg[\frac{p(X)}{1-p(X)}\bigg] = \beta_0 + \beta_1 X + \epsilon$
  * It is linear function **of** $X$, linear **in** betas
  * Essentially, we are building a linear regression for the log-odds response

</v-clicks>

<!--
* What are domain and range of $p(X)$? Does it include $0$ or $1$?
* What are domain and range of odds ratio? Does it include $0$ or $\infty$? Can be negative?
 -->

---

# Sigmoid Function

<div class="grid grid-cols-[6fr_3fr] gap-10">
<div>

* Sigmoid (logistic) function: $f(x) := \frac{1}{1 + e^{-x}}$
  * Differentiable on $\R$ (just compute a derivative)<br> $\implies$ continuous on $\R$
    * FYI: continuity does not imply differentiability
  * $f(-\infty)=0, f(0)=0.5, f(\infty)=1$
  * Poorly distinguishes values "far from zero", but almost **linear** near zero
    * $f'(x)|_0=f(x)(1-f(x))|_0=\frac{1}{2}\cdot (1-\frac{1}{2})=\frac{1}{4}$ is a rate of change of $f$ at zero
    * $f''(x)|_0=f(x)(1-f(x))(1-2f(x))|_0=$<br>$\frac{1}{4}\cdot 0=0$ is a rate of change of derivative at zero

</div>
<div>
<v-plotly style="width: 300px !important; height: 400px !important"
:data="[
{
mode: 'lines',
name: 'Sigmoid',
type: 'scatter',
x: [-5, -4.814815, -4.62963, -4.444444, -4.259259, -4.074074, -3.888889, -3.703704, -3.518519, -3.333333, -3.148148, -2.962963, -2.777778, -2.592593, -2.407407, -2.222222, -2.037037, -1.851852, -1.666667, -1.481481, -1.296296, -1.111111, -0.925926, -0.740741, -0.555556, -0.37037, -0.185185, 0, 0.185185, 0.37037, 0.555556, 0.740741, 0.925926, 1.111111, 1.296296, 1.481481, 1.666667, 1.851852, 2.037037, 2.222222, 2.407407, 2.592593, 2.777778, 2.962963, 3.148148, 3.333333, 3.518519, 3.703704, 3.888889, 4.074074, 4.259259, 4.444444, 4.62963, 4.814815, 5],
y: [0.006693, 0.008044, 0.009664, 0.011607, 0.013936, 0.016724, 0.020058, 0.02404, 0.02879, 0.034445, 0.041164, 0.049127, 0.058537, 0.069617, 0.08261, 0.097773, 0.115369, 0.135656, 0.158869, 0.185204, 0.214789, 0.247664, 0.283752, 0.322842, 0.364576, 0.408452, 0.453836, 0.5, 0.546164, 0.591548, 0.635424, 0.677158, 0.716248, 0.752336, 0.785211, 0.814796, 0.841131, 0.864344, 0.884631, 0.902227, 0.91739, 0.930383, 0.941463, 0.950873, 0.958836, 0.965555, 0.97121, 0.97596, 0.979942, 0.983276, 0.986064, 0.988393, 0.990336, 0.991956, 0.993307],
line: {color: 'blue'},
visible: true,
showlegend: true
},
{
mode: 'lines',
name: 'Linear',
type: 'scatter',
x: [-5, -4.814815, -4.62963, -4.444444, -4.259259, -4.074074, -3.888889, -3.703704, -3.518519, -3.333333, -3.148148, -2.962963, -2.777778, -2.592593, -2.407407, -2.222222, -2.037037, -1.851852, -1.666667, -1.481481, -1.296296, -1.111111, -0.925926, -0.740741, -0.555556, -0.37037, -0.185185, 0, 0.185185, 0.37037, 0.555556, 0.740741, 0.925926, 1.111111, 1.296296, 1.481481, 1.666667, 1.851852, 2.037037, 2.222222, 2.407407, 2.592593, 2.777778, 2.962963, 3.148148, 3.333333, 3.518519, 3.703704, 3.888889, 4.074074, 4.259259, 4.444444, 4.62963, 4.814815, 5],
y: [-2.0, -1.9074075000000001, -1.8148149999999998, -1.722222, -1.6296295, -1.5370370000000002, -1.4444445, -1.351852, -1.2592595, -1.1666665, -1.074074, -0.9814814999999999, -0.888889, -0.7962965, -0.7037035, -0.611111, -0.5185185000000001, -0.425926, -0.33333349999999995, -0.24074050000000002, -0.14814799999999995, -0.05555549999999998, 0.03703699999999999, 0.1296295, 0.22222199999999998, 0.314815, 0.40740750000000003, 0.5, 0.5925925, 0.6851849999999999, 0.7777780000000001, 0.8703704999999999, 0.962963, 1.0555555, 1.148148, 1.2407405, 1.3333335, 1.425926, 1.5185185, 1.611111, 1.7037035, 1.7962965, 1.888889, 1.9814815, 2.074074, 2.1666665, 2.2592594999999998, 2.351852, 2.4444445, 2.537037, 2.6296295, 2.722222, 2.814815, 2.9074075, 3.0]
,
line: {color: 'black'},
visible: 'legendonly', 
showlegend: true
}]"
:layout="{
xaxis: {title: 'x'},
yaxis: {title: 'f(x)'},
margin: {l: 40, r:0, b:70, t:20, pad: 2},
legend: {x:0.1, y: 0.9}
}"
:config="{displayModeBar: true}"
:options="{}"/>
</div>
</div>

* Sigmoid is a [CDF](https://en.wikipedia.org/wiki/Cumulative_distribution_function) with the corresponding bell-shaped derivative, i.e. [PDF](https://en.wikipedia.org/wiki/Probability_density_function)

---

# Estimation of Coefficients

<v-clicks depth="3" every="1">

* As always, we want the coefficients that are most likely to produce the observed values
  * under the given null hypothesis of model family, hyperparameters, etc.
* A likelihood function:<br>
$\ell(\beta_0, \beta_1 | x) \overset{\mathrm{i.i.d.} X_i}{=} \prod\limits_{i:y_i = 1} p(x_i) \cdot \prod\limits_{j:y_j = 0} \big(1 - p(x_j)\big)$
  * ${i:y_i = 1}$ are indices of all positive cases (a.k.a. class 1)
  * ${j:y_j = 0}$ are indices of all negative cases (a.k.a. class 0)
  * Recall: $p(x_i) = \mathbb{P}[Y | X = x_i]$ and we fixed it as<br> $p(x_i) = \mathbb{P}[Y | X = x_i, \mathrm{linear~model~for~logodds}]$
* Then we maximize the likelihood over the arguments:<br>
$\hat{\beta_0}, \hat{\beta_1} := \argmax\limits_{\beta_0, \beta_1} \ell (\beta_0, \beta_1)$
  * Because of numerical underflow, we use log likelihood $\log \ell ()$

</v-clicks>

---

# Ex. Card Balance

```python {1-3|4-5|6-8|all}
import math 
a = 0.5 # threshold probability 
b, s = 2000, 1 # balance, student (yes|no) 
e = math.exp(-10.6513 + 0.0055 * b) # p.142 ISLP 
p = e / (1 + e) # conditional probability of default 
print(f'P[default = Yes | balance=${b}] = {p:.3f}') 
print(f'P[default = No  | balance=${b}] = {1 - p:.3f}') 
print(f'Decision at {a} = {"" if p>a else "not"} likely to default') 

P[default = Yes | credit card balance=$2000] = 0.586 
P[default = No  | credit card balance=$2000] = 0.414 
Decision at 0.5 = likely to default 
```

---

# Ex. Predicting Default From Student Status

```python {1-3|4-5|6-11|all}
import math 
a = 0.5 # threshold probability 
b, s = 2000, 1 # balance, student (yes|no) 
e = lambda x=1: math.exp(-3.5041 + 0.4049 * x)
p = lambda x=1: e(x) / (1 + e(x)) # conditional probability of default 
print(f'P[default | student] = {p(s):.3f}') 
print(f'P[non-default | student] = {1 - p(s):.3f}') 
print(f'P[default | non student] = {p(1-s):.3f}') 
print(f'P[non-default | non student] = {1 - p(1-s):.3f}') 
print(f'Decision for a student at {a} = {"" if p(s)>a else "not"} likely to default') 
print(f'Decision for a non-student at {a} = {"" if p(1-s)>a else "not"} likely to default')

P[default | student] = 0.043 
P[non-default | student] = 0.957 
P[default | non student] = 0.029 
P[non-default | non student] = 0.971
Decision for a student at 0.5 = not likely to default
Decision for a non-student at 0.5 = not likely to default
```

---

# Making Predictions (i.e. Inferences)

<div class="grid grid-cols-[9fr_3fr] gap-2">
<div>

* Let's explore famous **[Iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)**
  * Goal: classify iris species based on petal and sepal widths and lengths
* 3 classes (species), but let's use just two
  * 50 observations per class
* 4 features, but let's just use sepal with
  * How well does this feature separate classes?
  * Consider sepal width of $x_0 \in \{4, 5, 5.5\}$

<br>
<img src="/iris_boxplots.png" style="width: 585px !important;">
</div>
<div>
  <figure>
<img src="/iris_setosa.png" style="width: 135px !important;">
<img src="/iris_versicolor.png" style="width: 135px !important;">
<img src="/iris_virginica.png" style="width: 135px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Images source:<br>
    <a href="https://en.wikipedia.org/wiki/Iris_flower_data_set">https://en.wikipedia.org/wiki/Iris_flower_data_set</a>
  </figcaption>
</figure>
</div>
</div>

---

# Ex. Fisher's Iris Data Set

```python {1-2|3-8|10-12|13-20|all}{maxHeight:'380px'}
import pandas as pd, numpy as np; from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
# Iris types: 0 = Setosa, 1 = Versicolour, 2 = Virginica
# Features (in cm): sepal length/width, petal length/width
X, y = load_iris(return_X_y=True)
X, y = X[y!=2], y[y!=2] # drop Virginica iris
X = X[:,[0]]
vX = X[49:50, :] # observations to classify

df = pd.DataFrame(X, columns=['Sepal_Width']);
ax = df.groupby(y).boxplot(vert=False, subplots=False, figsize=(12,3));
tmp = ax.set_title('Boxplot for Sepal width');

clf = LogisticRegression(random_state=1, penalty='none', fit_intercept=True) 
clf.fit(X, y)  # fit coefficients to training observations (gradient descent business)
betas = np.array(list(clf.intercept_) + list(clf.coef_[0]))
Odds = clf.intercept_ + vX * clf.coef_[0]
LogOdds = np.exp(Odds)
pX = LogOdds/(1+LogOdds)
alpha=0.5

print(f'Observations to classify: {vX}') 
print(f'Model coefficients: {betas.round(3)}')
print(f'Odds: {Odds[0]}, Log-odds: {LogOdds[0]}')
print(f'Class probabilities: \n P[Y=0|X]={1-pX[0].round(3)} \n P[Y=1|X]={pX[0].round(3)}')
print(f'Predicted class: {1 if pX > alpha else 0}')

# scikit-learn automatically computes
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True, \
  intercept_scaling=1, l1_ratio=None, max_iter=100, multi_class='auto', n_jobs=None, \
  penalty='none', random_state=1, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)

Observation to classify, X: [[5.]]
Model coefficients: [-27.831 5.14 ]
Odds: [-2.1297569]
Log-odds: [0.11886619]
Class probabilities:
  P[Y=0|X]=[0.894]
  P[Y=1|X]=[0.106]
Predicted class: 0

print(f'In-sample (training) accuracy: {clf.score(X, y)}')
print(f'Predicted probabilities (neg & pos classes): {clf.predict_proba(vX).round(3)}')
print(f'Predicted classes (thresholded probabilities): {clf.predict(vX)}')

In-sample (training) accuracy: 0.89
Predicted probabilities (neg & pos classes): [[0.894 0.106]]
Predicted classes (with thresholded probabilities): [0]
```

---

# Multiple Logistic Regression

* An extension to simple linear regression with log-odds:
$$\log\bigg(\frac{p(\bf{X})}{1-p(\bf{X})}\bigg) = \beta_0 + \beta_1 X_1 + ... + \beta_p X_p$$

* Probability of class 1 (positive) given $\bf{X} := X_{1:p}$:
$$p(\bf{X}) = \frac{\exp[\beta_0 + \beta_1 X_1 + ... + \beta_p X_p]}{1 + \exp[\beta_0 + \beta_1 X_1 + ... + \beta_p X_p]}$$


---

# Confounded (Correlated) Predictors

* Overall default rates are higher for students
* If balance is a predictor, then students have lower default rates
  * This is because students tend to have higher credit card balances and higher balances are associated with higher defaults
<br>
<br>

<figure>
  <img src="/ISLP/ISLP_figure_4.3.png" style="width: 585px !important;">
  <figcaption style="color:#b3b3b3ff; font-size: 9px;">Image source:
    <a href="https://hastie.su.domains/ISLP/ISLP_website.pdf.download.html">ISLP Fig. 4.3</a>
  </figcaption>
</figure>

---

# Logistic Regression for >2 Response Classes

* Consider $Y \in \{\mathrm{cat, dog, rat}\}$ with $K = 3$ classes
* Logistic regression handles binary response, i.e. $p(X)$ or $1 - p(X)$, but we happy to apply it to $K$ classes in **one-versus-all** (OVA or one-vs-rest) fashion:<br>
$p_k(X): = \mathbb{P} [Y = k | X]$<br>
$\log \frac{p_k(X)}{p_{\neg k}(X)}: = \beta_{k0} + \beta_{k1}X = [1 X] \bf{\beta}_k$
  * we only need to do this $K-1$ times, since $\sum\limits_k p_k (X) = 1$
  * So, we have $K$ linear functions:
    * $X \rightarrow X^{\prime}\beta_{\mathrm{cat}}, X \rightarrow X^{\prime}\beta_{\mathrm{dog}}, X \rightarrow X^{\prime}\beta_{\mathrm{rat}}$
* **Multinomial logistic regression** (a.k.a. softmax regression) with **multilogit** function:
  * It uses a generalized logistic function, called **softmax**:  $p_k(\bf{X}) := \frac{e^{\bf{X}^\prime \beta_k}}{\sum\limits_j e^{\bf{X}^\prime \beta_j}}$
    * So, if $p_{\mathrm{cat}}(\bf{X}) = 0.2$, $p_{\mathrm{dog}}(\bf{X}) = 0.3$, then: $p_{\mathrm{cat}}(\bf{X}) = 1 - p_{\mathrm{cat}}(\bf{X}) - p_{\mathrm{dog}}(\bf{X}) = 0.5$

---

# One-versus-one (OVO) approach

* We can also apply logistic regression for any pairs of $K$ response levels
  * This yields $\frac{K(K-1)}{2}$ pairs of inputs
  * Advantage: if you start with equal number of observations in each class (say, $n_k = 10$), then each logistic regression deals with **balanced classes**
    * In **one-versus-all**, we will have $10$ observations in class $k$ and $20$ obsevations in the rest

---

# Fitting Logistic Regression for 2 Classes

* We assume i.i.d. $X_i \sim \mathrm{Binomial} \big(\theta_{2 \times 1} := (n, p)\big)$
  * For $K > 2$, we assume i.i.d. $X_i \sim \mathrm{Multinomial} \big(\theta_1, ..., \theta_{K-1}\big)$
* Then maximize log likelihood estimator (MLE) of $\theta$:
$$\ell (\theta) = \log \mathrm{Bin}(\bf{X} | \theta) = \log \prod \mathrm{Bin} (X_i | \theta) = \sum \log \mathrm{Bin} (X_i | \theta)$$
* In terms of $\beta := [\beta_0, \beta_1]$, we have:
$$\ell(\beta) = \sum\limits_{i=1}^N \bigg\{ y_i \log p (x_i; \beta_ + (1 - y_i) \log \big(1 - p (x_i; \beta)\big) \bigg\} = \sum\limits_{i=1}^N \bigg\{ y_i \beta^T x_i - \log (1 + e^{\beta^T x_i}) \bigg\}$$
  * Take derivative, set it to zero to find critical values (min, max, saddle), second derivatives to identify local max. Then determine global max
  * See [ESL textbook](https://hastie.su.domains/ElemStatLearn/), p. 120-122
