# ml_utils_ts

**ml_utils_ts** is a Machine learning utils package adapted
 for time-series. It also provides efficient way to compute feature importance

## Installing 

**Require python>=3.6**

Install (editable)
```
pip install -e .
```

## Usage 

Build Sphinx documentation :

    cd docs
    make html
    
The package is composed of several modules including:

 - Module `cross_validation` implements different method of 
 cross-validation parallelized and adapted to time-series (purging, combinatorial, ..)
 - Module `calibration` implement the optimization of hyperparameters
 - Module `feature_importance` for computing feature importances
 - Module `stacking` for model stacking
 
**Notes :**
 
- Support any ML model implementing at least `fit` and `predict`
 methods 


    

