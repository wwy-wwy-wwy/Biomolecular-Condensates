import numpy as np
import pymc3 as pm

def set_model(data,quantization):
    
    """
    This function read in the experiment data, and generate an autoregressive model based on it.
    """
    
    #precision_upper = 10/(np.var(data)*(1-np.exp(-2/10)))
    
    ar1_model = pm.Model()
    with ar1_model:
    
        # 'phi'is ln(-1/tau) used in our generative model
        decay_time = pm.Uniform("decay_time",lower = 0, upper = 50) 
        stationarity = np.exp(-1/decay_time)
        # 'precision' is 1/(variance of innovation). As we use normalized data, this term has to be divided by  intensity_mean squared
        precision_AR1 = pm.Uniform("precision", lower = 0 , upper = 1) 
        # process mean
        center = pm.Uniform("center", lower = 0, upper = quantization) # this is the mean of normalized data
   
        likelihood = pm.AR1("y", k=stationarity, tau_e=precision_AR1, observed = data[1] - center)

    return ar1_model