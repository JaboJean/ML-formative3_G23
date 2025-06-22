# Statistical Modeling with Poisson Distribution and Bayesian Inference

This project demonstrates two fundamental statistical concepts implemented in Python without relying on specialized statistical libraries:

1. Manual implementation of the Poisson Distribution
2. Bayesian Inference for Email Spam Classification

## Project Overview

The project consists of two Jupyter notebooks that showcase practical applications of probability and statistics:

- `poisson.ipynb`: Implements the Poisson probability mass function from scratch
- `bayesian.ipynb`: Demonstrates Bayesian inference using email spam classification

## How It Works

### 1. Poisson Distribution Implementation

The Poisson distribution models the probability of a given number of events occurring in a fixed interval of time or space, assuming these events occur independently at a constant average rate.

#### Key Components:

- Manual implementation of factorial function
- Custom computation of Euler's number (e)
- Poisson probability mass function (PMF):

```python
def poisson_pmf(k, lam):
    e_neg_lambda = 1 / compute_e(lam)  # e^(-lambda)
    return (lam ** k) * e_neg_lambda / factorial(k)
```

#### Mathematical Formula:
P(X = k) = (λᵏ × e⁻ᵏ) / k!

where:
- λ (lambda) is the average number of events in the interval
- k is the number of events we're calculating the probability for
- e is Euler's number

### 2. Bayesian Inference Implementation

The project implements Bayes' theorem to classify emails as spam or not spam based on the presence of specific words.

#### Bayes' Theorem Formula:
P(A|B) = [P(B|A) × P(A)] / P(B)

In the spam classification context:
- P(A|B): Probability email is spam given it contains the word "free"
- P(B|A): Probability of "free" appearing in spam emails
- P(A): Prior probability of an email being spam
- P(B): Total probability of "free" appearing in any email

```python
def bayes_theorem(prior_A, likelihood_B_given_A, prior_not_A, likelihood_B_given_not_A):
    # Calculate evidence
    evidence_B = (likelihood_B_given_A * prior_A) + (likelihood_B_given_not_A * prior_not_A)
    
    # Calculate posterior
    posterior_A_given_B = (likelihood_B_given_A * prior_A) / evidence_B
    return posterior_A_given_B
```

## Usage

1. Open the Jupyter notebooks:
```bash
jupyter notebook poisson.ipynb
# or
jupyter notebook bayesian.ipynb
```

2. Run all cells in each notebook to:
   - Generate Poisson distribution visualizations
   - Calculate spam classification probabilities

## Output Examples

### Poisson Distribution
- Visualizes probability mass function for λ = 4
- Shows probabilities for 0 to 15 events
- Includes grid lines for better readability

### Bayesian Inference
Example calculation with:
- Prior (spam probability): 40%
- Likelihood ("free" in spam): 70%
- Evidence calculation
- Posterior probability: ~82.35%

## Practical Applications

### Poisson Distribution
- Modeling customer arrivals
- Call center incoming calls
- Network traffic patterns
- Defects in manufacturing

### Bayesian Inference
- Email spam filtering
- Medical diagnosis
- Fraud detection
- Adaptive learning systems

## Dependencies
- Python 3.x
- Matplotlib (for visualization)
- Jupyter Notebook