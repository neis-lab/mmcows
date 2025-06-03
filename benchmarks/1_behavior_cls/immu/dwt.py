import pywt
import numpy as np

waveletname = 'haar'

def process_dwt(data, window_size):
    dwt_output = []
    for sample in data:
        sample_dwt = []
        for i in range(3):  # Apply DWT to first three axes
            coeffs = pywt.wavedec(sample[:, i], waveletname)  # Using Daubechies wavelet db1
            sample_dwt.append(np.concatenate(coeffs))
        sample_dwt = np.array(sample_dwt).T[:window_size,:]
        sample_dwt = np.hstack((sample_dwt, sample[:,3].reshape((-1,1)))) # relative angle
        dwt_output.append(sample_dwt)
    return np.array(dwt_output)