The data files in this repository reflects data science projects. These projects include:

1. Titanic Data investigation
2. California Housing Data Investigation
3. Titanic Data investigation 2
4. Moscow Housing Data

-------

1. Titanic Data investigation
   
  For titanic data and files (titanic_data.ipynb & titanic_train.csv).

  This project is based upon Nick McCullum's tutorial https://www.nickmccullum.com/python-machine-learning/logistic-regression-python/#the-data-set-we-will-be-using-in-this-tutorial

  I learned about the logistic regression model, which maps continuous variables to a binary outcome. This is done via the logit or, log of the odds (p/(1-p)). 

  The advantage of this model is that you get probabilities for each variable, and it is easy to interpret.

  Ultimately, it was only somewhat good. The precision was ok, and the recall was ok. I am sure there are better ways to model the titanic data.

  Other notes:
  For the logistic regression model, people may want to use it, since, it is in essence based upon linear regression, a well understood method of training using data and then making predictions. However, if we have a situation in which we want to get predict a binary outcome, yes/no, 1/0 etc. using the standard linear regression will not work as the values for y can in principle span from -inf to +inf. Instead, we want to model the probability of the binary outcome. The way to use linear regression is to find a way to use a linear function, and relate it to probability. Here you set your linear function to the log of the odds, which is p/(1-p). You use odds, as this is related to probability, and the log allows you to deal with large negative values. Then, you use the Maximum Likelihood Estimation function to estimate the parameters. The estimator is constructed on equaling an observation with one Bernoulli outcome. Since each observation/person has its own unique probability, you need to multiply the probabilities for each observation together. This function you then try to maximise. 

-----  

2. California Housing Data Investigation
  
  For the California Housing data and files (scikitlearn - California Housing Dataset Training.ipynb & data extracted using python library scikitlearn).

   The purpose of this project was to use scikitlearn in a Jupyter notebook to build a linear regression model. Then apply a train_test_split, fit a LinearRegression model and perform a 5fold crossvalidation (cross_val_score), compute R² and RMSE, and create a residual plot.

   I learned how the cross validation works, which is to systematically create train/test data sets, rotate through them, and obtain basic metrics such as R² and RMSE to ensure consistency of performance of the linear regression.
   As part of the investigation, I inspected the data, before the cross validation, to see how the linear regression performed, by how the different variables impacted the median house price. From the scatterplots, the worst performer was median income. So I did test the linear regression without using median income, but the model was significantly worst, both in a train/test and cross validation.
   The data seems to be fairly well predicted by the linear regression. What I then did was attempted to put uncertainy values on the coefficients and intercept. I first tested for a normal distribution using box and whisker plot, residual historgram and a QQ plot. All the plots indicated that the distribution is not normal. As such, I did a bootstrapping method to generate multiple coefficients, from which I then extracted percentiles consistent with 2 sigma. I also obtained the standard deviation for comparison. Even though the data is not normally distributed, the percentiles and the standard deviation gave similar results.

-----

3. Titanic Data investigation 2

  The workflow begins with training and tuning a single Decision Tree model. A Decision Tree is simple and interpretable, making it ideal for understanding which features are most influential and for exploring how hyperparameters such as max_depth, min_samples_split, and min_samples_leaf affect bias, variance, and overfitting. Once the Decision Tree’s parameters are optimized and its behaviour understood, this insight is used as an initial guide for building a Random Forest model.

  A Random Forest is essentially an ensemble (an average) of many Decision Trees, each trained on different random subsets of the data and features. This averaging process greatly reduces variance and improves generalization compared to any single tree, at the cost of some interpretability. In practice, the tuned Decision Tree provides intuition about appropriate model complexity, while the Random Forest applies that understanding to achieve stronger and more stable predictive performance.

  In the final model, the most important determinants of survival were Sex (female passengers had a much higher survival rate) and Fare, which reflects ticket class and socioeconomic status—consistent with historical accounts of the Titanic disaster.

  -----

4. Moscow Housing Data

  Exploratory Data Analysis (EDA) was the most critical step in this project. It was during EDA that I identified a power-law relationship between price and area — visible as a linear relationship on a log–log plot. Recognizing this pattern early allowed me to log-transform the target (Price), which stabilized variance and improved model fit. This transformation was crucial, and experience with data transformations helped reveal it quickly.

  Given the very large number of categorical features (especially metro stations), it was not feasible to explore each one individually. Instead, focusing on core structural and locational variables — such as area, district, and building type — yielded the main insights efficiently.

  Modeling Results and Observations

  Linear Regression
  The initial linear regression gave an R² around 0.90 on a single train/test split — a strong start, though possibly optimistic.
  Using 5-fold cross-validation, the mean R² fell to ~0.79, demonstrating why cross-validation is essential: the original split likely benefited from favorable sampling.
  The residuals were not normally distributed, mainly due to outliers in high-end prices, which is common in housing markets.
  Despite that, linear regression offered a strong, interpretable baseline.

  Decision Tree and Random Forest
  The Decision Tree Regressor achieved an R² around 0.94 on its training/validation, but this likely reflects overfitting — a known tendency for single trees to memorize data.
  In contrast, the Random Forest averaged over many trees and produced a more conservative R² ≈ 0.77, closer to the cross-validated linear regression.
  The Random Forest highlighted that area (log-transformed) was by far the most influential predictor — a result consistent with the EDA findings.

  Gradient Boosting
  The Gradient Boosting Regressor produced a cross-validated R² around 0.85–0.89, depending on fold count and early-stopping parameters.
  It improved accuracy compared to both linear regression and random forest but required significantly more training time and parameter tuning.
  Once again, area dominated feature importance, reinforcing the robustness of that relationship.

  Overall Reflections

  The project illustrated how EDA informs modeling decisions — identifying the log–log trend between area and price directly shaped the choice of transformations.

  Cross-validation proved essential to separate genuine predictive power from lucky splits.

  Linear regression performed surprisingly well given its simplicity, offering a balance of interpretability, speed, and reliability.

  Ensemble methods (Random Forest, Gradient Boosting) captured nonlinearities and interactions, but at the cost of interpretability and computation time.

  Interestingly, feature importance rankings varied between methods — a common occurrence since each model type defines “importance” differently (variance reduction vs. coefficient magnitude vs. boosting gain).

  In practice:

  Use Linear Regression for transparency and quick baseline performance.

  Use Gradient Boosting for higher accuracy when time and compute allow.

  Across all models, log(area) remains the single most powerful predictor of price
