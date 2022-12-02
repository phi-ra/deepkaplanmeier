from itertools import product as itp
import pandas as pd

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