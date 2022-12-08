"""
Module with simulation utils
"""
import numpy as np
from scipy.stats import gamma

def simulate_random_data(size, seed, censoring_proportion=0.2):
    np.random.seed(seed)
    x_1 = np.random.exponential(0.1, size=size)
    x_2 = np.random.normal(10, np.sqrt(5), size=size)
    x_3 = np.random.poisson(5, size=size)

    features = np.array([x_1, x_2, x_3]).T

    # Arrange Targets
    survial_model = -3.14*x_1 + 0.318*x_2 + 2.72*x_3

    true_survival_time = gamma.rvs(survial_model, scale=1, random_state=seed)
    ceiling_surv_ = np.ceil(true_survival_time)

    censored_instances = np.random.uniform(size=size) > (1-censoring_proportion)
    censored_surv = [np.ceil(np.random.uniform()*val_) for val_ in ceiling_surv_[censored_instances]]

    observed_survival = ceiling_surv_.copy()
    observed_survival[censored_instances] = censored_surv

    variable_dict = dict({'true_survival': ceiling_surv_,
                        'observed_survival': observed_survival, 
                        'censoring': censored_instances,
                        'features' :  features})

    return variable_dict

def simluate_dependent_data(size, seed):
    size_ = size
    seed_ = seed

    np.random.seed(seed_)
    x_1 = np.random.exponential(0.1, size=size_)
    x_2 = np.random.normal(10, np.sqrt(5), size=size_)
    x_3 = np.random.poisson(5, size=size_)

    features = np.array([x_1, x_2, x_3]).T

    # Arrange Targets
    survial_model = -3.14*x_1 + 0.318*x_2 + 2.72*x_3

    true_survival_time = gamma.rvs(survial_model, scale=1, random_state=seed_)
    ceiling_surv_ = np.ceil(true_survival_time)

    censoring_transformed = np.minimum(2*(survial_model/survial_model.max()),1)

    censored_instances = np.random.uniform(size=size_) > (censoring_transformed)
    censored_surv = [np.ceil(np.random.uniform()*val_) for val_ in ceiling_surv_[censored_instances]]

    observed_survival = ceiling_surv_.copy()
    observed_survival[censored_instances] = censored_surv

    variable_dict = dict({'true_survival': ceiling_surv_,
                        'observed_survival': observed_survival, 
                        'censoring': censored_instances,
                        'features' :  features})

    return variable_dict


def generate_nonlinear_data(size, seed):
    size_ = size
    seed_ = seed

    np.random.seed(seed_)
    x_1 = np.random.exponential(0.1, size=size_)
    x_2 = np.random.normal(10, np.sqrt(5), size=size_)
    x_3 = np.random.poisson(5, size=size_)

    features = np.array([x_1, x_2, x_3]).T

    # Arrange Targets
    survial_model = np.sin(x_1)*3.14 + 0.318*x_2 + 2.72*np.abs(np.cos(x_3))

    true_survival_time = gamma.rvs(survial_model, scale=1, random_state=seed_)
    ceiling_surv_ = np.ceil(true_survival_time)

    censoring_transformed = np.minimum(2*(survial_model/survial_model.max()),1)

    censored_instances = np.random.uniform(size=size_) > (censoring_transformed)
    censored_surv = [np.ceil(np.random.uniform()*val_) for val_ in ceiling_surv_[censored_instances]]

    observed_survival = ceiling_surv_.copy()
    observed_survival[censored_instances] = censored_surv

    variable_dict = dict({'true_survival': ceiling_surv_,
                        'observed_survival': observed_survival, 
                        'censoring': censored_instances,
                        'features' :  features})

    return variable_dict

def generate_heavy_censored_data(size, seed):
    size_ = size
    seed_ = seed

    np.random.seed(seed_)
    x_1 = np.random.exponential(0.1, size=size_)
    x_2 = np.random.normal(10, np.sqrt(5), size=size_)
    x_3 = np.random.poisson(5, size=size_)

    features = np.array([x_1, x_2, x_3]).T

    # Arrange Targets
    survial_model = -3.14*x_1 + 0.318*x_2 + 2.72*x_3

    true_survival_time = gamma.rvs(survial_model, scale=1, random_state=seed_)
    ceiling_surv_ = true_survival_time #np.ceil(true_survival_time)

    censoring_transformed = np.minimum(0.75*(survial_model/survial_model.max()),1)

    censored_instances = np.random.uniform(size=size_) > (censoring_transformed)
    censored_surv = [np.ceil(np.random.uniform()*val_) for val_ in ceiling_surv_[censored_instances]]

    observed_survival = ceiling_surv_.copy()
    observed_survival[censored_instances] = censored_surv

    variable_dict = dict({'true_survival': ceiling_surv_,
                        'observed_survival': observed_survival, 
                        'censoring': censored_instances,
                        'features' :  features})

    return variable_dict
