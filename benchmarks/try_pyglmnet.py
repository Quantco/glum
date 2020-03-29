import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps
from pyglmnet import GLM, simulate_glm

n_samples, n_features = 1000, 100
distr = "poisson"

# sample a sparse model
np.random.seed(42)
beta0 = np.random.rand()
beta = sps.random(1, n_features, density=0.2).toarray()[0]

# simulate data
Xtrain = np.random.normal(0.0, 1.0, [n_samples, n_features])
ytrain = simulate_glm("poisson", beta0, beta, Xtrain)
Xtest = np.random.normal(0.0, 1.0, [n_samples, n_features])
ytest = simulate_glm("poisson", beta0, beta, Xtest)

# create an instance of the GLM class
glm = GLM(distr="poisson", score_metric="pseudo_R2", reg_lambda=0.01)

# fit the model on the training data
glm.fit(Xtrain, ytrain)

# predict using fitted model on the test data
yhat = glm.predict(Xtest)

# score the model on test data
pseudo_R2 = glm.score(Xtest, ytest)
print("Pseudo R^2 is %.3f" % pseudo_R2)

# plot the true coefficients and the estimated ones
plt.stem(beta, markerfmt="r.", label="True coefficients")
plt.stem(glm.beta_, markerfmt="b.", label="Estimated coefficients")
plt.ylabel(r"$\beta$")
plt.legend(loc="upper right")

# plot the true vs predicted label
plt.figure()
plt.plot(ytest, yhat, ".")
plt.xlabel("True labels")
plt.ylabel("Predicted labels")
plt.plot([0, ytest.max()], [0, ytest.max()], "r--")
plt.savefig("pyglmnet.pdf")
