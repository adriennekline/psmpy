`PsmPy`
=====

Matching techniques for epidemiological observational studies as carried out in Python. Propensity score matching is a statistical matching technique   used   with   observational   data   that   attempts   to ascertain the validity of concluding there is a potential causal link between a treatment or intervention and an outcome(s) of interest. It does so by accounting for a set of covariates between a binary treatment state (as would occur in a randomized control trial, either received the intervention or not), and control for potential confounding (covariates) in outcome measures between the treatment and control groups such as death, or length of stay etc. It is using this technique on observational data that we gain an insight into the effects or lack thereof of an interventional state.

---

## Citing this work:

A. Kline, Y. Luo, PsmPy:  *A  Package  for  Retrospective  Cohort Matching  in  Python*, (under review at EMBC 2022)

---

* Integration with Jupyter Notebooks
* Additional plotting functionality to assess balance before and after
* A more modular, user-specified matching process
* Ability to define 1:1 or 1:many matching

---

# Installation

Install the package through pip:

```bash
$ pip install psmpy
```

* [Installation](#installation)
* [Data Preparation](#data-prep)
* [Predict Scores](#predict-scores)
* [Matching algorithm](#matching-algorithm)
* [Graphical Outputs](#graphical-outputs)
* [Extra Attributes](#extra-attributes)
* [Conclusion](#conclusion)

----

# Data Prep


```python
# import relevant libraries
sns.set(rc={'figure.figsize':(10,8)}, font_scale = 1.3)
```

```python
# read in your data
data = pd.read_csv(path)
```
----

# Import psmpy class and functions
```python
from psmpy import PsmPy
from psmpy.functions import cohenD
from psmpy.plotting import *
```

# Initialize PsmPy Class

Initialize the `PsmPy` class:

```python
psm = PsmPy(df, treatment='treatment', indx='pat_id', exclude = [])
```

**Note:**

* `PsmPy` - The class. It will use all covariates in the dataset unless formally excluded in the `exclude` argument.
* `df` - the dataframe being passed to the class
* `exclude` - (optional) parameter and will ignore any covariates (columns) passed to the it during the model fitting process. This will be a list of strings. Note, it is not necessary to pass the unique index column here. That process will be taken care of within the code after specifying your index column.
* `indx` - required parameter that references a unique ID number for each case in the dataset.

# Predict Scores
Calculate logistic propensity scores/logits:

```python
psm.logistic_ps(balance = True)
```

**Note:**

* `balance` - Whether the logistic regression will run in a balanced fashion, default = True.

There often exists a significant **Class Imbalance** in the data. This will be detected automatically in the software where the majority group has more records than the minority group. We account for this by setting `balance=True` when calling `psm.logistic_ps()`. This tells `PsmPy` to sample from the majority group when fitting the logistic regression model so that the groups are of equal size. This process is repeated until all the entries of the major class have been regressed on the minor class in equal paritions. This calculates both the logistic propensity scores and logits for each entry.

Review values in dataframe:

```
psm.predicted_data
```

---

# Matching algorithm - version 1

Perform KNN matching

```python
psm.knn_matched(matcher='propensity_logit', replacement=False, caliper=None)
```

**Note:**

* `matcher` - `propensity_logit` (default) and generated inprevious  step  alternative  option  is  `propensity_score`, specifies the argument on which matching will proceed
* `replacement` -  `False`   (default),   determines   whethermacthing  will  happen  with  or  without  replacement,when replacement is false matching happens 1:1
* `caliper` - `None` (default), user can specify caliper size relative  to  std.  dev  of  the  control  sample,  restricting neighbors eligible to match within a certain distance. 

---

# Matching algorithm - version 2

Perform KNN matching 1:many 

```python
psm.knn_matched_12n(matcher='propensity_logit', how_many=1)
```

**Note:**

* `matcher` - `propensity_logit` (default) and generated inprevious  step  alternative  option  is  `propensity_score`, specifies the argument on which matching will proceed
* `how_many` - `1` (default) performs 1:n matching, where 'n' is specified by the user and matched the minor class 'n' times to the major class 

---

# Graphical Outputs

## Plot the propensity score or propensity logits
Plot the distribution of the propensity scores (or logits) for the two groups side by side.

```python
psm.plot_match(Title='Side by side matched controls', Ylabel='Number ofpatients', Xlabel= 'Propensity logit',names = ['treatment', 'control'],save=True)
```

**Note:**

* `title` -    'Side   by   side   matched   controls' (default),creates plot title
* `Ylabel` -  'Number  of  patients'  (default),  string,  labelfor y-axis
* `Xlabel` -  'Propensity logit' (default), string, label for x-axis 
* `names` - ['treatment', 'control'] (default), list of strings for legend
* `save` -  False  (default),  saves  the  figure  generated  to current working directory if True

## Plot the effect sizes 

```python
psm.effect_size_plot(save=False)
```

**Note:**
* `save` -  False  (default),  saves  the  figure  generated  tocurrent working directory if True

---

# Extra Attributes
Other attributes available to user:
## Matched IDs

```python
psm.matched_ids
```

* `matched_ids` - returns  a  dataframe  of  indicies from  the  minor  class  and  their  associated  matched indice from the major class psm.

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Major_ID</th>
      <th>Minor_ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6781</td>
      <td>9432</td>
    </tr>
    <tr>
      <td>3264</td>
      <td>7624</td>
    </tr>
    <tr>
    </tr>
  </tbody>
</table>


**Note:**
That not all matches will be unique if `replacement=False`

## Matched Dataframe 

```python
psm.df_matched
```

* `df_matched` - returns a subset of the original dataframe using indices that were matched. This works regardless of which matching protocol is used. 

## Effect sizes per variable

```python
psm.effect_size
```

* `effect_size` - returns  dataframe  with  columns 'variable', 'matching' (before or after), and 'effect_size'

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>variable</th>
      <th>matching</th>
      <th>effect_size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>hypertension</td>
      <td>before</td>
      <td>0.5</td>
    </tr>
    <tr>
      <td>hypertension</td>
      <td>after</td>
      <td>0.01</td>
    </tr>
    <tr>
      <td>age</td>
      <td>7624</td>
      <td>9432</td>
    </tr>
    <tr>
      <td>age</td>
      <td>7624</td>
      <td>9432</td>
    </tr>
    <tr>
      <td>sex</td>
      <td>7624</td>
      <td>9432</td>
    </tr>
    <tr>
    </tr>
  </tbody>
</table>

**Note:** The thresholds for a small, medium and large effect size were characterizedby Cohen in: J. Cohen, "A Power Primer", Quantitative Methods in Psychology, vol.111, no. 1, pp. 155-159, 1992

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Relative Size</th>
      <th>Effect Size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>small</td>
      <td> &le; 0.2</td>
    </tr>
    <tr>
      <td>medium</td>
      <td> &le; 0.5</td>
    </tr>
    <tr>
      <td>large</td>
      <td> &le;0.8</td>
    </tr>
    <tr>
    </tr>
  </tbody>
</table>

---

# Conclusion
This package offers a user friendly propensity score matching protocol created for a Python environment. In this we have tried to capture automatic figure generation, contextualization of the results and flexibility in the matching and modeling protocol to serve a wide base. 