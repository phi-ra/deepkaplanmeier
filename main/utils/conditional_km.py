"""
Module for the conditional Kaplan-Meier estimator
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

class DeepKaplanMeier:
    
    def __init__(self,
                 periods=None):
        """
        ToDo: Make period counter more flexible (include between period counters)
        
        Parameters
        ----------
        periods: int
            Total number of periods that observation has taken place
        """
        self._set_periods(periods)
        
    def _set_periods(self,
                     periods):
        assert isinstance(periods, int), f"""Specify the periods as integer,
                                             you specified it as {type(periods)}"""
        self.periods = periods

    def prepare_survival(self,
                         survival_time,
                         censoring):
        """
        Prepares target input matrix for estimation
        
        Parameters
        ----------
        survival time: list(int) or np.array(int)
            Indicator of last observed period, either censoring time or
            failure time
            
        censoring: list(bool) or np.array(bool)
            Censoring indicator, if True observation is censored
            
        returns: y-matrix and weights matrix
        """
        if isinstance(survival_time, list):
            observations = len(survival_time)
        elif isinstance(survival_time, np.ndarray):
            observations = survival_time.shape[0]
        else:
            raise Exception(f"""you need to provide either an array or
                      list of survival times, you provided {type(survival_time)}""")
                
        weight_matrix = np.ones(shape=(observations, self.periods+1))
        survival_matrix = pad_sequences([np.repeat(1, surv_time+1) if cens_ else
                                         np.repeat(1, surv_time) for
                                         surv_time, cens_ in zip(survival_time, censoring)],
                                        padding='post')

        weight_matrix[censoring,:] = weight_matrix[censoring,:] * survival_matrix[censoring,:]

        return survival_matrix, weight_matrix

    def create_output_struct(self, 
                             hidden_layer_output,
                             activation,
                             preoutput_units):
        outputs = []
        if preoutput_units is not None:
            for _ in range(self.periods + 1):
                aggregation_layer = tf.keras.layers.Dense(units=preoutput_units, 
                                                        activation=activation)(hidden_layer_output)
                output_unit = tf.keras.layers.Dense(1, activation='sigmoid')(aggregation_layer)
                outputs.append(output_unit)
        else:
            for _ in range(self.periods + 1):
                output_unit = tf.keras.layers.Dense(1, activation='sigmoid')(hidden_layer_output)
                outputs.append(output_unit)

        for unit_idx in np.arange(1, self.periods + 1):
            outputs[unit_idx] = tf.keras.layers.Multiply()([outputs[unit_idx], outputs[unit_idx - 1]])
        
        return outputs

    @staticmethod
    def create_rep_struct(input_mod,
                          hidden_units,
                          activation):
        if isinstance(hidden_units, int):
            input_mod = tf.keras.layers.Dense(hidden_units, 
                              activation=activation)(input_mod)
        elif isinstance(hidden_units, list):
            for units_ in hidden_units:
                input_mod = tf.keras.layers.Dense(units_, 
                              activation=activation)(input_mod)  
        else:
            raise TypeError('hidden_units must be int or list or ints')

        return input_mod

    def compile_model(self,
                      input_shape,
                      hidden_units,
                      activation='relu',
                      preoutput=50):
        """
        Compiles keras model according to inputs

        Parameters
        ----------
        input_shape: tuple
            Tuple specifying input shape. In general of the form (k, )

        hidden_units: int or list(int)
            If integer, the number of hidden units in the single hidden layer
            If list, one integers specifying hidden units per hidden layer

        random_seed: str, optional (Default: 'relu')
            activation function for hidden units
        """
        tf.keras.backend.clear_session()

        input_layer = tf.keras.Input(shape=input_shape)
        input_mod = input_layer

        input_layer = tf.keras.Input(shape=input_shape)
        input_mod = input_layer

        input_mod = self.create_rep_struct(input_mod, hidden_units, activation)
        
        outputs = self.create_output_struct(hidden_layer_output=input_mod,
                                            activation=activation,
                                            preoutput_units=preoutput)

        self.model = tf.keras.Model(inputs=input_layer, outputs=outputs)
        self.model.compile(optimizer="Adam", loss="binary_crossentropy")

    def return_comp_graph_raw(self,
                              input_shape,
                              hidden_units,
                              activation='relu', 
                              preoutput=None):
        tf.keras.backend.clear_session()
        input_layer = tf.keras.Input(shape=input_shape)
        input_mod = input_layer

        input_mod = self.create_rep_struct(input_mod, hidden_units, activation)
        
        outputs = self.create_output_struct(hidden_layer_output=input_mod,
                                            activation=activation, 
                                            preoutput_units=preoutput)
                                            
        model_raw =  tf.keras.Model(inputs=input_layer, outputs=outputs)

        return model_raw

    def fit_model(self, X_train, y_survival, censoring,
                  epochs=50, batch_size = 256, verbose=False):
        
        y_train_np, weights_train_np = self.prepare_survival(y_survival, censoring)
        
        y_train = [y_train_np[:, i] for i in range(y_train_np.shape[1])]
        weights_train = [weights_train_np[:, i] for i in range(weights_train_np.shape[1])]
        
        if verbose:
            hist_ = self.model.fit(X_train,
                                   y_train,
                                   sample_weight=weights_train,
                                   epochs=epochs,
                                   batch_size=batch_size)
            return hist_
        else:
            self.model.fit(x=X_train,
                           y=y_train,
                           sample_weight=weights_train,
                           verbose=verbose,
                           epochs=epochs,
                           batch_size=batch_size)


class SimpleDeepKaplanCensoring(DeepKaplanMeier):
    """
    Add docstring
    """
    def __init__(self, periods=None):
        super().__init__(periods)

    def prepare_survival(self, survival_time, censoring, censoring_matrix):
        weight_matrix = censoring_matrix
        survival_matrix = pad_sequences([np.repeat(1, surv_time+1) if cens_ else
                                         np.repeat(1, surv_time) for
                                         surv_time, cens_ in zip(survival_time, censoring)],
                                        padding='post')

        weight_matrix[censoring,:] = weight_matrix[censoring,:] * survival_matrix[censoring,:]

        return survival_matrix, weight_matrix


    def fit_model(self, X_train, y_survival, censoring, censoring_matrix,
                  epochs=50, batch_size=256, verbose=False):
        y_train_np, weights_train_np = self.prepare_survival(y_survival, censoring, censoring_matrix)
        
        y_train = [y_train_np[:, i] for i in range(y_train_np.shape[1])]
        weights_train = [weights_train_np[:, i] for i in range(weights_train_np.shape[1])]
        
        if verbose:
            hist_ = self.model.fit(X_train,
                                   y_train,
                                   sample_weight=weights_train,
                                   epochs=epochs,
                                   batch_size=batch_size)
            return hist_
        else:
            self.model.fit(x=X_train,
                           y=y_train,
                           sample_weight=weights_train,
                           verbose=verbose,
                           epochs=epochs,
                           batch_size=batch_size)

