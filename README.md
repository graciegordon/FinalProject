# BMI203-Final Project: Neural Network

[![Build
Status](https://travis-ci.org/graciegordon/FinalProject.svg?branch=master)](https://travis-ci.org/graciegordon/FinalProject)

## What does this repo contain?
1. autoencoder.py: 8x3x8 autoencoder
2. RAP1.py: trains neural network to predict if a sequence will be a RAP1 binding site and predict held out test set
3. utils.py: basic io and encoding functions
4. rocCurve.py: calculates true positive and true negative rates and plots a ROC
## To test
Testing is as simple as running

```
python -m pytest
```

from the root directory of this project.
