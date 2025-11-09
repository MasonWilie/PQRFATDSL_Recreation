### About

This project is meant to be a recreation of the results presented in the paper titled "Parsimonious Quantile Regression of Financial Asset Tail Dynamics via Sequential Learning" by Xing Yan, Weizhong Zhang, Lin Ma, Wei Liu, and Qi Wu. The paper was presented in NIPS 2018, and is available at...

https://papers.nips.cc/paper/7430-parsimonious-quantile-regression-of-financial-asset-tail-dynamics-via-sequential-learning

I am not at all associated with this paper or its authors, nor am I attempting to pass their work off as my own. I simply wish to make available code which recreates their results.

### Solution to Multi-Quantile Optimization

**Issue #1 (Now Fixed):** The original implementation only optimized for a single quantile (0.5) instead of all 21 quantiles simultaneously as described in the paper's Equation 9.

**Solution:** The pinball loss function has been updated to implement Equation 9 correctly:

- Removed the global `quantile` variable that locked the model to a single quantile
- Modified `htqf_fun()` to accept a quantile parameter `tau`
- Rewrote `pinball_loss()` to iterate over all K=21 quantiles: τ₁, τ₂, ..., τ₂₁
- For each quantile τₖ, the function:
  1. Computes the HTQF value: Q(τₖ|µₜ, σₜ, uₜ, vₜ)
  2. Calculates the pinball loss: Lτₖ(yₜ, Q(τₖ|...))
  3. Sums these losses across all quantiles
- Returns the average loss: (1/K) Σₖ Lτₖ

**Issue #2 (Now Fixed):** The output layer used ReLU activation, which clipped negative values to zero. This prevented the model from learning:
- Negative mean values (financial returns can be negative)
- Small tail parameters u and v (which get clipped to 0, eliminating tail heaviness)

**Solution:** Changed the output layer activation from `relu` to `tanh` as specified in **Equation 8** of the paper:
```python
model.add(Dense(4, activation='tanh'))  # Matches paper's specification
```

The `tanh` activation allows the network to output values in (-1, 1), enabling proper learning of all HTQF parameters.

This implementation now correctly minimizes pinball losses over all 21 quantiles simultaneously, allowing the model to capture the full conditional distribution as intended in the paper.

### Setup With venv

```bash
$ git clone https://www.github.com/MasonWilie/PQRFATDSL_Recreation.git
$ cd PQRFATDSL_Recreation
$ python -m venv venv
$ source venv/bin/activate  # On Windows use: venv\Scripts\activate
$ pip install -r requirements.txt
```
