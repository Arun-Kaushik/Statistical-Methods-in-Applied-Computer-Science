# Statistical Methods in Applied Computer Science
This repo contains solutions to assignments in the course DD2447 Statistical Methods in Applied Computer Science at KTH.


## Assignment 1
The first assignment in this course was to solve problems mainly about Bayesian probability from Machine Learning A Probabilistic 
Perspective by Kevin P. Murphy. 

## Assignment 2
### Assignment 2.1
The first part of the next assignment was to estimate a posterior distribution with a Gibbs sampler and then to generate synthetic 
data. For a detailed description of the assignment see: Assignment_2_Description.pdf, section: 2.1 Gibbs sampler for the magic word.

#### Excerpt of some results
Histograms of sampled start positions used to estimate the start position distributions, for two different word sequences where the true star position can be found in the plot header:
<p float="left" align='center'>  
  <img src='https://github.com/alexandrahotti/Statistical-Methods-in-Applied-Computer-Science/blob/master/Assignment%202/2.1%20-%20Magic%20Word%20Gibbs%20Sampler/results/dist_word_seq_1.png' width="40%" height="40%"
 /><img src='https://github.com/alexandrahotti/Statistical-Methods-in-Applied-Computer-Science/blob/master/Assignment%202/2.1%20-%20Magic%20Word%20Gibbs%20Sampler/results/dist_word_seq_2.png' width="40%" height="40%"
 />


### Assignment 2.2
The next part of the assignment was to compare different combinations of Markov Chain Monte Carlo methods such as Metropolis Hastings, 
Gibbs sampling and blocked Gibbs sampling to estimate the posterior for the switch settings of a train riding along a
railway. For a detailed description of the assignment see: Assignment_2_Description.pdf, section: 2.2 MCMC for the train.

### Assignment 2.3
The next assignment was to implement sequential Monte Carlo on a Stochastic Volatility model (SVM) using sequential
importance sampling to infer the hidden underlying volatility. For a detailed description of the assignment see: Assignment_2_Description.pdf, section: 2.3 SMC for the stochastic volatility model.

### Assignment 2.4
In the next sub assignment both the variance parameters and the hidden states of a SVM were inferred using a particle marginal 
Metropolis-Hastings sampler (PMMH). For a detailed description of the assignment see: Assignment_2_Description.pdf, section: 2.4 Stochastic volatility unknown parameters part I.

### Assignment 2.4
Lastly the same inference was performed via a conditional SMC sampler. For a detailed description of the assignment see: Assignment_2_Description.pdf, section: 2.5 Stochastic volatility unknown parameters part II.
