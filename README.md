# MNIST-EBM
The project aims at training and generating MNIST images, using the Energy Based Methods (EBM) framework

## About the project
Energy based methods have recently been developed as a framwork for generative ML. The project aims at developing the basic workings of energy based methods. For this the task of generating MNIST images is chosen. However, due to computational as well as algorithmic constraints, various methods and techniques are applied to get this task done.

1. __Binary Data__
  As intractibility of pdf in the likelihood is a major concern in energy based models, I begin with converting the 0-255 grayscale data to 0-1 binary. It is analysed that the images are still clearly recognizable after this conversion. However this reduces the computaional cost

2. __Autoencoder__
  As is with the case for EBMs, the infrence process is not straighforward like the forward pass of a neural net. One of the most import algorithms used in this field in Metropolis-adjusted Langevin algorithm, also called Langevin MCMC. However the algorithm only holds true for real valued variables, Therefore an additional autoencoder is trained that takes the input from 0-1 binary space to $R^{d}$ continous space

3. __Contrastive Divergence__
  Once the setup is done the energy model is trained by minimizing the Negative-Log Likelihood, which is done using Contrastive Divergence technique. Note that even during this technique we need random samples that are in turn generated by MCMC technique mentioned above. 

## Setup and Scripts
### 1. Model training from command line.<br>
```python
python train.py --auto_hidden_dim 1024 --encoded_dim 4 --auto_num_epochs 30 --auto_batch_size 64 --auto_lr 3e-4 --ebm_hidden_dim 8 --ebm_num_epochs 3 --ebm_batch_size 1024 --ebm_lr 3e-4 --mcmc_samples_per_datapoint 8
```

### 2. Model inference from command line.<br>
```python
python infer.py --num_images 10
```

## Model Interface
The Gradio link for model inference can be found [here](URL)

## File Desciption
1. `train.ipynb` file contains a jupyter notebook to train the model from scratch.
2. `inference.ipynb` file contains a jupyter notebook to load the trained model and perform inference
3. `train.py` file contains the python code to train model from scratch.
4. `infer.py` file contains the python code to do model inference.
5. `MCMC.png` file contains the Langevin MCMC used. Please note that this is adopted from Standford CS:236 course.





