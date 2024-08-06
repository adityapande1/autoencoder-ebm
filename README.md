# MNIST-EBM
The project aims at training and generating MNIST images, using the Energy Based Methods (EBM) framework

## About the project
Energy based methods have recently been developed as a framwork for generative ML. The project aims at developing the basic workings of energy based methods. For this the task of generating MNIST images is chosen. However, due to computational as well as algorithmic constraints, various methods and techniques are applied to get this task done. Below is a short discussion on the underlying theoritic basics of the project

Energy Based Models (EBMs) are also called non-normalised probability models. In its most general form, the probablity density function is given as:

$$
\begin{equation}
p_{\theta}(\mathbf{x}) = \frac{\exp(-E_{\theta}(\mathbf{x}))}{Z_{\theta}}
\end{equation}
$$

This is the same very well known Gibb's distribution used in statistical physics. This specific distribution satisfies the maximum entropy property, assuming no specified prior. This gives us a useful precondition to model a lot of processes without any prior assumptions. 
Note the term $Z_{\theta}$ in $Eq \: 1$ denominator is equal to :

$$
\begin{equation}
    Z_{\theta} = \int \exp(-E_{\theta}(\mathbf{x})) \, d\mathbf{x}
\end{equation}
$$

This is because the p.d.f should be normalised. This emerges as the major problem while working with Energy based models. The term $Z_{\theta}$ is in most cases intractable. This is the mojor bottleneck in any kind of EBM modelling. The following discussion elaborates how this $tractibilty$ issue is catered for.

### 1. __Binary Data__<br>
  As intractibility of pdf in the likelihood is a major concern in energy based models, We begin with converting the 0-255 grayscale data to 0-1 binary. It is analysed that the images are still clearly recognizable after this conversion. However this reduces the computaional cost.<br> But still
  Let us for an example take a $32 \times 32$ grayscale image ranged $[0-255]$. The random variable(image in this case) can take any value out of $255^{32*32}$ configurations. Calculation of such scale in not possible even with very large computation capabilities. It thus becomes imperative to for us to navigate around this issue, in order to make progress in EBMs. This leads us to the next step :

### 2. __Autoencoder__<br>
  In order to reduce the input space from wvey high dimensional space, an autoencoder is introduced. The 0-1 images are encoded in a small ${R^{d}}$, This reduced the computaional and training cost. Also,
  as is with the case for EBMs, the infrence process is not straighforward like the forward pass of a neural net. One of the most import algorithms used in this field in Metropolis-adjusted Langevin algorithm, also called Langevin MCMC. However the algorithm only holds true for real valued variables, Therefore an additional autoencoder is trained that takes the input from 0-1 binary space to $R^{d}$ continous space

### 3. __Contrastive Divergence__<br>
Once the setup is done the energy model is trained by minimizing the Negative-Log Likelihood, which is done using Contrastive Divergence technique. Note that even during this technique we need random samples that are in turn generated by MCMC technique mentioned above. The maximum likelihood estimate of the above probablity model is:

$$
\begin{equation}
    \nabla_{\theta} \log p_{\theta}(\mathbf{x}) = -\nabla_{\theta} E_{\theta}(\mathbf{x}) - \nabla_{\theta} \log Z_{\theta}
\end{equation}
$$

The first term in the right hand side equation can be easily calculated using any AutoDiff engine like Pytorch. We also derived an expression for the second term above :
Now here lies the trick, Monte Carlo methods give us the ability to estimate expectations from averages. Therefore we can write : 

$$
\nabla_{\theta} \log Z_{\theta} \simeq - \nabla_{\theta} E_{\theta} (\tilde{\mathbf{x}}),
$$

This stochastically estimates the second term in the R.H.S of the equation. But the fact still remains that a proper way is needed to get $\tilde{\mathbf{x}}$ which is a sample from the model itself. Here we use Markov chain Monte Carlo (MCMC)techniques to get the data point. More specifically a very famous algorithm used here is Langevin MCMC. The algorithm goes as follows :

$$
\mathbf{x}^{k+1}\leftarrow \mathbf{x}^k+\frac{\epsilon^2}{2} \nabla_{\mathbf{x}} \log p_{\theta} (\mathbf{x}^k) + \epsilon \mathbf{z}^k, \quad k = 0, 1, \cdots, K-1.
$$

For small enough $\epsilon$ and sufficiently large $K$ the algorithm indeed produces results from the original data distribution. Note that the initialization $\mathbf{x}^{0}$ is done randomly.
MCMC not only provides a way to train the model also to infer i.e, to sample examples from it once the model is trained.

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





