### About

This project is meant to be a recreation of the results presented in the paper titled "Parsimonious Quantile Regression of Financial Asset Tail Dynamics via Sequential Learning" by Xing Yan, Weizhong Zhang, Lin Ma, Wei Liu, and Qi Wu. The paper was presented in NIPS 2018, and is available at...

https://papers.nips.cc/paper/7430-parsimonious-quantile-regression-of-financial-asset-tail-dynamics-via-sequential-learning

I am not at all associated with this paper or its authors, nor am I attempting to pass their work off as my own. I simply wish to make available code which recreates their results.

### Issues

This project was not a total success, with the main issue being that I could not figure out how to simultaniously minimize the pinball losses over all 21 quantiles proposed in the paper rather than just at one while still using a Keras loss function. I have not been able to find a solution though theoretically this can be acheived through the paper's equation 9.

No farther work is planned for this project.
