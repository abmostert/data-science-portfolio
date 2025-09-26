For titanic data and files.

This project is based upon Nick McCullum's tutorial https://www.nickmccullum.com/python-machine-learning/logistic-regression-python/#the-data-set-we-will-be-using-in-this-tutorial

I learned about the logistic regression model, which maps continuous variables to a binary outcome. This is done via the logit or, log of the odds (p/(1-p)). 

The advantage of this model is that you get probabilities for each variable, and it is easy to interpret.

Ultimately, it was only somewhat good. The precision was ok, and the recall was ok. I am sure there are better ways to model the titanic data.

Other notes:
For the logistic regression model, people may want to use it, since, it is in essence based upon linear regression, a well understood method of training using data and then making predictions. However, if we have a situation in which we want to get predict a binary outcome, yes/no, 1/0 etc. using the standard linear regression will not work as the values for y can in principle span from -inf to +inf. Instead, we want to model the probability of the binary outcome. The way to use linear regression is to find a way to use a linear function, and relate it to probability. Here you set your linear function to the log of the odds, which is p/(1-p). You use odds, as this is related to probability, and the log allows you to deal with large negative values. Then, you use the Maximum Likelihood Estimation function to estimate the parameters.
The estimator is constructed on equaling an observation with one Bernoulli outcome. Since each observation/person has its own unique probability, you need to multiply the probabilities for each observation together. This function you then try to maximise. 
