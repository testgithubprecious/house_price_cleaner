#  House Price Cleaner

A powerful preprocessing library built for Kaggle house price advanced regression prediction datasets.  
This tool streamlines the entire data preparation workflow — from cleaning and encoding to feature engineering and selection.

---

## Features

- Fills missing numerical values intelligently
- Encodes ordinal features with custom mappings
- Separates low-cardinality and high-cardinality categorical variables
- One-hot encodes low-cardinality features
- Target encodes high-cardinality features
- Engineers binary flags for sparse features (e.g., pools, porches, miscellaneous)
- Converts skewed continuous variables with log scaling
- Applies MinMax scaling to normalize numeric data
- Automatically selects the most important features using XGBoost

---

##  Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/testgithubprecious/house_price_cleaner.git

