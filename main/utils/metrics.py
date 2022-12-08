"""
Module containing different metrics and helpers to calculate them
"""
import pandas as pd
import numpy as np
from pycox.evaluation import EvalSurv
from itertools import product as itp

def calculate_preds(output_raw, y_surv_, y_cens_):
    """
    Evaluator for benchmark testings

    Parameters
    ----------
    output_raw: list of arrays
        the predictions from a DeepKaplanMeier model

    y_surv_: np.array
        the array with the observed test survival times

    y_cens_: np.array
        the array specifying whether an event was observed
        if the event was observed it needs to be coded as 
        1, if the observation was censored as 0

    returns:
        concordance index and brier score as floats
    """
    output_ = (pd.DataFrame(list(map(np.ravel, output_raw)))
                                .reset_index()
                                .rename(columns={'index':'timeline'}))

    time_idx = np.array(output_['timeline'].astype(float))
    output_ = output_.set_index(time_idx).drop(columns='timeline')

    evaluation_results = EvalSurv(output_,
                                    y_surv_,
                                    y_cens_,
                                    censor_surv='km')
    
    linspace_brier = np.linspace(y_surv_.min(), 
                                 y_surv_.max(), 
                                 100)
    
    
    concordance_idx = evaluation_results.concordance_td('antolini')
    brier_idx = evaluation_results.integrated_brier_score(linspace_brier) 

    return concordance_idx, brier_idx


def calculate_concordance_surv(survival_, censoring_, pred_matrix):
    """
    Calculates time dependent concordance index

    Parameters
    ----------
    survival_: int
        True observed survival time

    censoring_: bool
        Indicator whether the event happended (=False) or whether the observation
        is censored (=True)

    pred_matrix: pd.DataFrame
        Dataframe with the predicted survival times for all relevant periods

    returns: all concorand pairs, all possible pairs and the time dependent concordance index
    """
    # get all pairs    
    denom_list = []
    nom_list = []

    for prod_ in itp(range(len(survival_)), range(len(survival_))):
        try:
            if survival_[prod_[0]] > survival_[prod_[1]] and not censoring_[prod_[0]] and not censoring_[prod_[1]]:
                denom_list.append(prod_)

                if pred_matrix.at[survival_[prod_[0]], prod_[0]] > pred_matrix.at[survival_[prod_[0]], prod_[1]]:
                    nom_list.append(prod_)
            elif survival_[prod_[0]] > survival_[prod_[1]] and censoring_[prod_[0]] and not censoring_[prod_[1]]:
                denom_list.append(prod_)

                if pred_matrix.at[survival_[prod_[0]], prod_[0]] > pred_matrix.at[survival_[prod_[0]], prod_[1]]:
                    nom_list.append(prod_)
        except:
            continue
    vals_nom= len(nom_list) / len(denom_list)
    return nom_list, denom_list, vals_nom