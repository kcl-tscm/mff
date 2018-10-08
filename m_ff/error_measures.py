import numpy as np


def MAEF(tst_confs, tst_forces, gp):
    """Mean Absolute Error on Force

    Calculated the absolute error done on the force vectors

    Args:
        tst_confs (list): Configurations in the test set
        tst_forces (array): Forces in the test set
        gp (class): Trained gp class

    Returns:
        MAEF (float): the Mean Absolute error on force

    """
    predicted_forces = gp.predict(tst_confs, return_std=False)
    MAEF = np.mean(np.linalg.norm(tst_forces - predicted_forces, axis=1))

    return MAEF


def MAEC(tst_confs, tst_forces, gp):
    """Mean Absolute Error on Components

    Calculated the absolute error done on the force components

    Args:
        tst_confs (list): Configurations in the test set
        tst_forces (array): Forces in the test set
        gp (class): Trained gp class

    Returns:
        MAEF (float): the Mean Absolute error on force components

    """
    predicted_forces = gp.predict(tst_confs, return_std=False)
    MAEC = np.mean(abs(tst_forces - predicted_forces))

    return MAEC


def RMSE(tst_confs, tst_forces, gp):
    """Root Mean Squared Error

    Calculated the root mean squared error made on force

    Args:
        tst_confs (list): Configurations in the test set
        tst_forces (array): Forces in the test set
        gp (class): Trained gp class

    Returns:
        RMSE (float): the Root Mean Square Deviation of predicted forces

    """
    predicted_forces = gp.predict(tst_confs, return_std=False)
    RMSE = np.sqrt(np.mean((tst_forces - predicted_forces) ** 2))

    return RMSE


def neg_log_pred(tst_confs, tst_forces, gp):
    """Negative log predictive probabiltiy of the data

    Calculates the probability assigned by the trained model to the test set.
    The covariance matrix is approximated to be diaginal.

    Args:
        tst_confs (list): Configurations in the test set
        tst_forces (array): Forces in the test set
        gp (class): Trained gp class

    Returns:
        neg_log_lik_diag (float): negative log predictive probability of force data
    """
    predicted_forces, predicted_std = gp.predict(tst_confs, return_std=True)

    neg_log_lik_diag = np.mean((tst_forces - predicted_forces) ** 2 / (2 * predicted_std ** 2) + np.log(
        predicted_std) + 0.5 * np.log(2 * np.pi))

    return neg_log_lik_diag
