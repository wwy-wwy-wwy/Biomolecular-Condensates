import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm

def simulate_single_decay_data(number_data_point,tau, quantization, intensity_mean, intensity_std, camera_noise_mean, camera_noise_std):
    phi = np.exp(-1/tau)

    # A probabilistic generator for the innovation term.
    innovation_mean = intensity_mean*(1-phi) # this is the c term in the wikipedia construction
    innovation_std = intensity_std*np.sqrt(1-phi**2) # this is the standard deviation of the innovation 
    innovation = np.random.normal(loc=0.0, scale=innovation_std, size = number_data_point)
    intensity_variance = intensity_std**2 # this is the expectation value of all intensity signals

    print('The set mean of innovation is {}'.format(innovation_mean))
    print('The set std of innovation is {}'.format(innovation_std))
    
    #Generate simulated data with statsmodel

    ar = np.r_[1, -phi] # add zero-lag and negate
    ma = np.r_[1] # add zero-lag
    X_statsmodel_generated = sm.tsa.arma_generate_sample(ar, ma, number_data_point, scale=innovation_std)

    X = X_statsmodel_generated + intensity_mean

    #Add camera noise
    camera_noise = np.random.normal(loc = camera_noise_mean, scale = camera_noise_std, size = number_data_point)

    X_simulated_data = X+camera_noise

    #Plot simulated data
    t_simulated=np.arange(number_data_point)
    plt.plot(t_simulated,X_simulated_data,'-')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.ylim([0,quantization])
    plt.show()

    print('the mean of simulated data is {}'.format(np.average(X_simulated_data)))
    print('the std of simulated data is {}'.format(np.std(X_simulated_data)))
    return X_simulated_data, t_simulated

def simulate_double_decay_data(number_data_point,tau_1, tau_2, relative_var, quantization, intensity_mean, intensity_std, camera_noise_mean, camera_noise_std):
    # Generate simulated data with statsmodel
    # relative variance is variance of the fast process divided by the sum of the variance of both AR1 processes
    # This number will show up in the autocorrelation function as the maginitude of the first jumb
    # divided by the initial point.
    intensity_std1 = np.sqrt(relative_var)*intensity_std
    intensity_std2 = np.sqrt(1 - relative_var)*intensity_std

    phi1 = np.exp(-1/tau_1)
    innovation_std1 = intensity_std1*np.sqrt(1-phi1**2) # this is the standard deviation of the innovation 
    print('precision 1 is {}'.format(1/innovation_std1/innovation_std1))
    ar1 = np.r_[1, -phi1] # add zero-lag and negate
    ma = np.r_[1] # add zero-lag
    X_statsmodel_generated1 = sm.tsa.arma_generate_sample(ar1, ma, number_data_point, scale=innovation_std1)

    phi2 = np.exp(-1/(tau_2))
    innovation_std2 = intensity_std2*np.sqrt(1-phi2**2) # this is the standard deviation of the innovation 
    print('precision 2 is {}'.format(1/innovation_std2/innovation_std2))
    ar2 = np.r_[1, -phi2] # add zero-lag and negate
    ma = np.r_[1] # add zero-lag
    X_statsmodel_generated2 = sm.tsa.arma_generate_sample(ar2, ma, number_data_point, scale=innovation_std2)

    #Relative variance of the intensity std of the fast process 
    #vs. the variance of the intensity std of the slow process

    X = X_statsmodel_generated1 + X_statsmodel_generated2 + intensity_mean

    #Add camera noise
    camera_noise = np.random.normal(loc = camera_noise_mean, scale = camera_noise_std, size = number_data_point)

    X_simulated_data = X+camera_noise

    #Plot simulated data
    t_simulated=np.arange(number_data_point)
    plt.plot(t_simulated,X_simulated_data,'-')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    #plt.ylim([0,quantization])
    plt.show()

    X_simulated_data_realmean = np.average(X_simulated_data)
    X_simulated_data_realstd = np.std(X_simulated_data)

    camera_noise_realmean = np.average(camera_noise)
    camera_noise_realstd = np.std(camera_noise)
    print('the mean of camera noise data is {}'.format(camera_noise_realmean))
    print('the std of camera noise data is {}'.format(camera_noise_realstd))
    print('the mean of fast simulated data is {}'.format(np.mean(X_statsmodel_generated1)))
    print('the mean of slow simulated data is {}'.format(np.mean(X_statsmodel_generated2)))
    print('the mean of simulated data is {}'.format(X_simulated_data_realmean))
    print('the std of simulated data is {}'.format(X_simulated_data_realstd))
    #Initial point of the autocorrelation function
    print('the (variance/mean^2) of simulated data is {}'.format((X_simulated_data_realstd/X_simulated_data_realmean)**2))
    return X_simulated_data,t_simulated