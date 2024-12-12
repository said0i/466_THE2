import os
import cv2
import pywt
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import dct, idct
from sklearn.metrics import mean_squared_error

def scale_coeffs(coeffs, max_level):
    scaled_coeffs=[]
    if(np.abs(coeffs[0]).max()!=0):
        scaled_coeffs.append(coeffs[0]*(256/np.abs(coeffs[0]).max()).astype(np.uint8))
    else:
        scaled_coeffs.append(coeffs[0])
    for detail_level in range(max_level):
        scaled_coeffs.append( [(d*256/np.abs(d).max()).astype(np.uint8) if np.abs(d).max()>0 else d for d in coeffs[detail_level + 1]])
        
    return scaled_coeffs
def descale_haar_coeffs(scaled_coeffs, max_level):
    coeffs=[]
    coeffs.append(scaled_coeffs[0]*(np.abs(scaled_coeffs[0]).max()/256).astype(np.uint8))
    for detail_level in range(max_level):
        coeffs.append([(d*(np.abs(d).max()/256)).astype(np.uint8) for d in scaled_coeffs[detail_level + 1]])
    return coeffs

def apply_haar_transform(image, N_values, wavelet_type,max_level,img_name):
    coeffs = pywt.wavedec2(image, wavelet=wavelet_type,level=max_level)

    coeffs_array, coeffs_slices = pywt.coeffs_to_array(coeffs)
    sorted_coeffs = np.sort(np.abs(coeffs_array).flatten())
    results={}
    for i,N in enumerate(N_values):
        threshold_idx = int((1 - N / 100.0) * len(sorted_coeffs))
        threshold = sorted_coeffs[threshold_idx]
        filtered_coeffs_array = np.where(np.abs(coeffs_array) >= threshold, coeffs_array, 0)
        
        #before saving normalize the coefficients to 0-255 for better compression
        max=np.nanmax(np.array(filtered_coeffs_array))
        min=np.nanmin(np.array(filtered_coeffs_array))

        normalized_filtered_coeffs = 255 * (filtered_coeffs_array - min) / (max - min)
        normalized_filtered_coeffs = filtered_coeffs_array / (2**(max_level+1)) +122.5 #scale the coefficients to 0-255

        # Save wavelet coefficients as compressed
        coeffs_uint8 = normalized_filtered_coeffs.astype(np.uint8)
        np.savez_compressed(f'Outputs/{img_name}_haar_transformed_N{N}.npz', coeffs=normalized_filtered_coeffs,min_max=np.array([min,max]), slices=np.array(coeffs_slices, dtype=object))
        #np.savez_compressed(f'Outputs/{img_name}_haar_transformed_4level_N{N}.npz', coeffs=filtered_coeffs_array,min_max=np.array([min,max]), slices=np.array(coeffs_slices, dtype=object))

        # Load wavelet coefficients
        loaded = np.load(f'Outputs/{img_name}_haar_transformed_N{N}.npz', allow_pickle=True)
        loaded_coeffs_array =loaded['coeffs']
        loaded_slices = loaded['slices']
        loaded_min_max=loaded['min_max']
        loaded_coeffs_array = loaded_coeffs_array.astype(np.float64)
# (x-mi)/(ma-mi) * 255       /255
        denormalized_loaded_coeffs_array = loaded_coeffs_array / 255.  * (loaded_min_max[1] - loaded_min_max[0]) + loaded_min_max[0]
        denormalized_loaded_coeffs_array = (loaded_coeffs_array -122.5) * (2**(max_level+1))
        
        loaded_coeffs=pywt.array_to_coeffs(denormalized_loaded_coeffs_array,loaded_slices,output_format='wavedec2')
        compressed_image = pywt.waverec2(loaded_coeffs, wavelet=wavelet_type)
        results[N] = np.clip(compressed_image, 0, 255).astype(np.uint8)
    return results

    
def apply_dct(image, N_values,img_name):
    dct_transformed = dct(dct(image.T, norm='ortho').T, norm='ortho')
    flattened = dct_transformed.flatten()
    results = {}
    
    for N in N_values:
        threshold_idx = int((1 - N / 100.0) * len(flattened))
        threshold = np.partition(np.abs(flattened), threshold_idx)[threshold_idx]
        
        # Filter coefficients
        filtered_dct_transformed = np.where(np.abs(dct_transformed) >= threshold, dct_transformed, 0)
        #store the filtered coefficients as npz file
        np.savez_compressed(f'Outputs/{img_name}_dct_transformed_N{N}.npz', coeffs=filtered_dct_transformed)
        # Load DCT coefficients
        loaded = np.load(f'Outputs/{img_name}_dct_transformed_N{N}.npz')
        loaded_dct_transformed = loaded['coeffs']
        # Reconstruct image
        compressed_image = idct(idct(loaded_dct_transformed.T, norm='ortho').T, norm='ortho')
        results[N] = np.clip(compressed_image, 0, 255).astype(np.uint8)
    
    return results

def calculate_mse(original, compressed):
    return mean_squared_error(original.flatten(), compressed.flatten())

# Example Usage
os.makedirs('Outputs', exist_ok=True)
image_folder = 'THE2_Images/'
image_names = ["c1", "c2", "c3"]
image_paths = [image_folder+ name for name in image_names]
N_values = [1, 10, 50,100]

for i,img_path in enumerate(image_paths):
    # Load grayscale image
    original = cv2.imread(img_path+".jpg", cv2.IMREAD_GRAYSCALE)
    
    # Apply Haar Transform
    haar_results = apply_haar_transform(original, N_values,'haar',1,image_names[i])
    
    # Apply DCT
    dct_results = apply_dct(original, N_values,image_names[i])
    
    # Compare results
    for N in N_values:
        cv2.imwrite(f"Outputs/{image_names[i]}_haar_N{N}.jpg", haar_results[N])   
        cv2.imwrite(f"Outputs/{image_names[i]}_dct_N{N}.jpg", dct_results[N])  
        # Compute MSE
        mse_haar = calculate_mse(original, haar_results[N])
        mse_dct = calculate_mse(original, dct_results[N])
        
        print(f"Image: {img_path}, N={N}%")
        print(f"  Haar MSE: {mse_haar}")
        print(f"  DCT MSE: {mse_dct}")
