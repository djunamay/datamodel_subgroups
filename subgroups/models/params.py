import random

def xgboost_params():
    # Continuous hyperparameters: sample uniformly from a range.
    learning_rate = random.uniform(0.05, 0.3)       # e.g., from 0.05 to 0.3
    reg_lambda = random.uniform(0.5, 3.0)             # L2 regularization: 0.5 to 3.0
    reg_alpha = random.uniform(0.0, 2.0)              # L1 regularization: 0.0 to 2.0
    subsample = random.uniform(0.1, 1.0)              # Subsampling ratio: 0.1 to 1.0
    colsample_bytree = random.uniform(0.5, 1.0)       # Column subsampling: 0.5 to 1.0
    gamma = random.uniform(0, 1.0)                    # Minimum loss reduction: 0 to 1.0

    # Discrete hyperparameters: sample integer values in a given range.
    max_depth = random.randint(2, 5)                  # Tree depth: between 2 and 5
    n_estimators = random.randint(50, 200)            # Number of trees: between 50 and 200
    min_child_weight = random.randint(1, 10)          # Minimum child weight: between 1 and 10

    # Optionally round continuous parameters for readability
    return {
        "learning_rate": round(learning_rate, 4),
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "reg_lambda": round(reg_lambda, 4),
        "reg_alpha": round(reg_alpha, 4),
        "subsample": round(subsample, 4),
        "colsample_bytree": round(colsample_bytree, 4),
        "gamma": round(gamma, 4),
        "min_child_weight": min_child_weight
    }
