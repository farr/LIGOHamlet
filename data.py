import numpy as np

def process_coinc_triggers(coinc_file, d1_file, d2_file):
    coincs = np.loadtxt(coinc_file, skiprows=1, dtype=[('GPS1sec', np.int),
                                                       ('GPS1nsec', np.int),
                                                       ('SNR1', np.float),
                                                       ('GPS2sec', np.int),
                                                       ('GPS2nsec', np.int),
                                                       ('SNR2', np.float)])
    trig1 = np.loadtxt(d1_file, skiprows=1, dtype=[('GPSsec', np.int),
                                                   ('GPSnsec', np.int),
                                                   ('SNR', np.float)])
    trig2 = np.loadtxt(d2_file, skiprows=1, dtype=[('GPSsec', np.int),
                                                   ('GPSnsec', np.int),
                                                   ('SNR', np.float)])

    for coinc in coincs:
        trig1 = trig1[(trig1['GPSsec'] != coinc['GPS1sec']) |
                      (trig1['GPSnsec'] != coinc['GPS1nsec']) |
                      (trig1['SNR'] != coinc['SNR1'])]
        trig2 = trig2[(trig2['GPSsec'] != coinc['GPS2sec']) |
                      (trig2['GPSnsec'] != coinc['GPS2nsec']) |
                      (trig2['SNR'] != coinc['SNR2'])]

    return coincs, trig1, trig2
        
