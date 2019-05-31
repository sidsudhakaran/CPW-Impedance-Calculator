# CPW-Impedance-Calculator
This repo contains the code for an impedance calculator for a coplanar waveguide, implemented with the help of a plain vanilla neural network. This was originally a project idea that I had for a project for my microwaves assignment, but I didn't end up submitting it. I finished up the code a bit later. 

As we know, a neural network can learn the properties of a function during training. 

## Getting Started 

### Coplanar Waveguide Impedance Theory and Formulae
The theory for the coplanar waveguide is given on [Microwaves101](https://www.microwaves101.com/encyclopedias/coplanar-waveguide). They also have an [online calculator](https://www.microwaves101.com/calculators/864-coplanar-waveguide-calculator) to find the parameters of the waveguide. However, [this website](https://chemandy.com/calculators/coplanar-waveguide-with-ground-calculator.htm) gives the formulae to calculate the impedance of the waveguide, alongwith a calculator. 

### Dataset Generation
Since this is a regression problem, where the network needs to learn elliptical functions, it was essential to generate a dataset to train the network. I generated the test set using the CPW in-built functions in the RF Toolbox in Matlab. The code for the same is in the ```Generate.m``` file.

I've also uploaded the ```ZData.csv``` file for the dataset. 

### Prequisites
Following packages are required:
```
python 
TensorFlow
Keras
scikit-learn
numpy
pandas
tabulate
```

### Running 

Code is pretty much self-explanatory. Just change the ```Predictions``` and ```epochs``` values to save your results in different files.  
### Results

In general, results improved as number of epochs was increased. The network performace could be better with a deeper network. Average error is the average of the difference between the predicted and actual values of the impedance.

| Epochs  | Average Error |
| ------------- | ------------- |
| 100  |  7.49 |
| 500  | 1.27  |
| 1000  | 0.57  |

### Acknowledgements
This was inspired by [this paper](https://ieeexplore.ieee.org/document/4763072). However, the source code for the same was not released so I created the network using Keras. 
