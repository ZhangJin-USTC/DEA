# DEA: *D*AG *E*stimation with *A*BESS 

This is an `Python` implementation of the following paper:

xxx

If you find this code useful, please consider citing:
```
xxx
```

## Introduction

DEA (also known as ABESS-DAG) is a new approach to combine variable selection methods (especially ABESS) with DAG learning. The biggest advantage of this method over other TOSAs (Topological Ordering Search Algorithms) is that there is no need to estimate the conditional variance to introduce additional error. In addition to the basic "vanilla" version of the method DEA (ABESS-DAG), we propose two different extensions. The first one is layer-by-layer learning algorithm DEAL (L-ABESS-DAG in our code), which is applicable in large-scale problems, and can significantly speed up the computation. The second one is DEAP (P-ABESS-DAG in our code), which is applicable to scenarios with known prior knowledge, and can integrate prior knowledge into the learning results and effectively improve the estimation results of DAG.


## Requirements
- Python
- Library `abess`
- Library `pandas`
- Library `numpy`
- Library `networkx`

## Contents
- `abessdag.py` Main function to run our algorithm, see demo below
- `abessdag-utils.py` Some helper functions to simulate data and evaluate results
- Real datasets
- Simulations

## Demo
First we generate a *ER* graph with 8 nodes and 16 expected edges. The graph weight parameters are uniformly distributed from $[-1,-0.5]\cup[0.5,1]$ with $\sigma=1$

```python
from abessdag-utils import graphgen,simdag
G = graphgen(p=8, q=2, min_w=0.5, max_w=1, graph_type="ER")
data = simdag(n=1000, graph=G, sd=1, error_type="Gaussian")
```

Then, we can apply our algorithm on the data and learn the structure of DAG.
```python
from abessdag import abessdag
result1 = abessdag(data)    # Vanilla ABESS-DAG
result2 = abessdag(data, nu=0.5)    # Layer-by-layer ABESS-DAG
```

We can introduce a priori knowledge sets to improve the estimation.
```python
kset = pd.DataFrame({"Cause":['X0','X3'],"Effect":['X2','X6']})
weak_kset = pd.DataFrame({"Cause":['X5','X6','X7','X0','X1','X5'],"Effect":['X0','X1','X2','X6','X0','X4']})
result3 = abessdag(data, kset=kset)    # ABESS-DAG with knowledge set
result4 = abessdag(data, kset=kset, wkset=weak_kset)    # ABESS-DAG with strong and weak knowledge sets
```

Check outputs
```python
print(result1[0])    # Estimated topological ordering
print(result1[1])    # Estimated edges
```

## Real datasets
We used two real datasets, the marks dataset [1] and the Sachs protein dataset [2].

## References
- [1] Bibby, J. M., Mardia, K. V., & Kent, J. T. (1979). **Multivariate analysis**. *Academic Press*.
- [2] Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., & Nolan, G. P. (2005). **Causal protein-signaling networks derived from multiparameter single-cell data**. *Science*, 308(5721), 523-529.

- We generate data from Gaussian process models through the `RESIT` code, from [here](https://staff.fnwi.uva.nl/j.m.mooij/code/codeANM.zip)
- Original equal variance code for linear models is from [here](https://github.com/WY-Chen/EqVarDAG)
