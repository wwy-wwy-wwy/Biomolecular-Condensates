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
    
def set_double_scale_model(data, quantization,tao1):
    ar1_two_timescales_model = pm.Model()

    with ar1_two_timescales_model:
        # 'phi'is ln(-1/tau) used in our generative model
        decay_time1 = pm.Uniform("decay_time_1",lower = 0, upper = 500, testval=tao1) 
        decay_time_split = pm.Uniform("decay_time_split",lower = 0, upper = 500)
        decay_time2 = pm.Deterministic("decay_time_2",decay_time1 + decay_time_split)
    
        stationarity1 = pm.Deterministic("stationarity_1", np.exp(-1/decay_time1))
        stationarity2 = pm.Deterministic("stationarity_2", np.exp(-1/decay_time2))

        # 'precision' is 1/(variance of innovation). As we use normalized data, this term has to be divided by intensity_mean squared
        precision_1 = pm.Uniform("precision_1", lower = 0 , upper = 1) 
        precision_2 = pm.Uniform("precision_2", lower = 0 , upper = 1) 
    
        # process mean: use observed mean since process is assumed to be stationary, and there should be
        # weak correlation with the other parameters anyway
        observed_mean = np.mean(data)
        
        #camera_noise_std=8.787413879857576
        camera_noise_std = pm.Uniform("noise_std", lower=0, upper=quantization)
    
        true1 = pm.AR1("y1", k=stationarity1, tau_e=precision_1, shape=len(data))
        true2 = pm.AR1("y2", k=stationarity2, tau_e=precision_2, shape=len(data))
        likelihood = pm.Normal("likelihood", mu=(true1+ true2 + observed_mean), sigma=camera_noise_std, observed=data)
        
    return ar1_two_timescales_model
    