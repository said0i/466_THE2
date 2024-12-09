import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.metrics import mean_squared_error

def apply_haar_transform(image, N_values, wavelet='haar'):
    coeffs2 = pywt.dwt2(image, wavelet)
    cA, (cH, cV, cD) = coeffs2
    all_coeffs = np.concatenate([cA.flatten(), cH.flatten(), cV.flatten(), cD.flatten()])
    results = {}
    
    for N in N_values:
        threshold_idx = int((1 - N / 100.0) * len(all_coeffs))
        threshold = np.partition(np.abs(all_coeffs), threshold_idx)[threshold_idx]
        
        # Filter coefficients
        cA_filt = np.where(np.abs(cA) >= threshold, cA, 0)
        cH_filt = np.where(np.abs(cH) >= threshold, cH, 0)
        cV_filt = np.where(np.abs(cV) >= threshold, cV, 0)
        cD_filt = np.where(np.abs(cD) >= threshold, cD, 0)
        
        # Reconstruct image
        compressed_image = pywt.idwt2((cA_filt, (cH_filt, cV_filt, cD_filt)), wavelet)
        results[N] = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return results

def apply_dct(image, N_values):
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    flattened = dct_transformed.flatten()
    results = {}
    
    for N in N_values:
        threshold_idx = int((1 - N / 100.0) * len(flattened))
        threshold = np.partition(np.abs(flattened), threshold_idx)[threshold_idx]
        
        # Filter coefficients
        filtered = np.where(np.abs(dct_transformed) >= threshold, dct_transformed, 0)
        
        # Reconstruct image
        compressed_image = idct(idct(filtered.T, norm='ortho').T, norm='ortho')
        results[N] = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return results

def calculate_mse(original, compressed):
    return mean_squared_error(original.flatten(), compressed.flatten())

# Example Usage
image_folder = 'THE2_Images/'
image_names = ["c1.jpg", "c2.jpg", "c3.jpg"]
image_paths = [image_folder+ name for name in image_names]
N_values = [1, 10, 50]

for img_path in image_paths:
    # Load grayscale image
    original = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Haar Transform
    haar_results = apply_haar_transform(original, N_values)
    
    # Apply DCT
    dct_results = apply_dct(original, N_values)
    
    # Compare results
    for N in N_values:
        # Save compressed images
        cv2.imwrite(f"{img_path.split('.')[0]}_haar_N{N}.jpg", haar_results[N])
        cv2.imwrite(f"{img_path.split('.')[0]}_dct_N{N}.jpg", dct_results[N])
        
        # Compute MSE
        mse_haar = calculate_mse(original, haar_results[N])
        mse_dct = calculate_mse(original, dct_results[N])
        
        print(f"Image: {img_path}, N={N}%")
        print(f"  Haar MSE: {mse_haar}")
        print(f"  DCT MSE: {mse_dct}")
