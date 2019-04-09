`Allograph` is a library for computer/robot-supported handwriting
activities. It provides algorithms to learn allograph of letters/numbers in
different metrics and different learning strategies from demonstration.

![learning non-cursive demo](doc/learning_demo.png)

*Robot writing in a cursive way learning non-cursive letters.*

Different metrics including : euclidian distance in cartesian space, euclidian
distance in eigenspace with PCA, decomposition in mixture of sigma-log-normal
distributions, Recurrent neural networks.

## Dependencies
- [recordtype](https://pypi.org/project/recordtype/)

## Installation 
Run sudo python setup.py install

## Letter Difficulty Based On A Dissimilarity Metric
The notebook for the computation of the dissimilarity metric can be found [here](https://github.com/LailaHms/allograph/blob/master/notebooks/Letter%20Difficulty%20-%20Dissimilarity%20Measure%20of%20Multivariate%20Time%20Series%20Evolution.ipynb)

The database for the dissimilarity metric computation can be found at https://c4science.ch/diffusion/8574/ but is only accessible to members of the CHILI lab due to ethical constraints pertaining to the nature of the dataset. 
