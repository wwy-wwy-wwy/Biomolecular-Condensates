import numpy as np
import pymc3 as pm

def set_model(data,quantization):
    
    """
    input: 
    data: ndarray, 
    quantization: float
    return: AR1 model
    This function read in the experiment data, and generate an autoregressive model based on it.
    """

    # Bayesian parameter estimation with pymc3
    ar1_model = pm.Model()

    with ar1_model:
        # 'phi'is ln(-1/tau) used in our generative model
        decay_time = pm.Uniform("decay_time",lower = 0, upper = 5000) 
        stationarity = pm.Deterministic("stationarity", np.exp(-1/decay_time))
        # 'precision' is 1/(variance of innovation). As we use normalized data, this term has to be divided by intensity_mean squared
        precision_AR1 = pm.Uniform("precision", lower = 0 , upper = 1) 
        # process mean: use observed mean since process is assumed to be stationary, and there should be
        # weak correlation with the other parameters anyway
        observed_mean = np.mean(data)   
        #camera_noise_std=8.787413879857576
        camera_noise_std = pm.Uniform("noise_std", lower=0, upper=quantization)
        
        true = pm.AR1("y", k=stationarity, tau_e=precision_AR1, shape=len(data))
        likelihood = pm.Normal("likelihood", mu=(true + observed_mean), sigma=camera_noise_std, observed=data)
        

    return ar1_model
    