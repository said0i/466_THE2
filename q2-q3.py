# Student1 Name: Ahmed Said Gençkol Student1 ID: 2539377
# Student2 Name: Batuhan Teberoğu Student2 ID: 2581056

import numpy as np
import cv2 
import os

input_folder = 'THE2_Images/'
output_folder = 'THE2_Outputs/'

ddepth = cv2.CV_32F # Change this to -1 to have the same depth as the input images.

# Filters
roberts_dx = np.array([[1, 0], [0, -1]])
roberts_dy = np.array([[0, 1], [-1, 0]])
prewitt_dx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_dy = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
sobel_dx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_dy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

def read_image(filename):
    img = cv2.imread(input_folder + filename, cv2.IMREAD_UNCHANGED)
    return img

def write_image(img, filename):
    cv2.imwrite(output_folder+filename, img)

def apply_mask(image, kernel_dx, kernel_dy, step_idx, prefix):
    dx = cv2.filter2D(image, ddepth, kernel_dx)
    dy = cv2.filter2D(image, ddepth, kernel_dy)

    dx = cv2.convertScaleAbs(dx, alpha=255/dx.max())
    dy = cv2.convertScaleAbs(dy, alpha=255/dy.max())
    # Try to approximate the gradiant by adding dx and dy.
    # The operation below is actually: result = dx * 0.5 + dy * 0.5 + 0
    grad = cv2.addWeighted(dx, 0.5, dy, 0.5, 0)

    # write_image(dx, f'q1/{step_idx}/{prefix}_dx_result.png')
    # write_image(dy, f'q1/{step_idx}/{prefix}_dy_result.png')
    write_image(grad, f'q1/{step_idx}/{prefix}.png')

    return grad

def apply_edge_detectors(step_idx, images):
    results = []
 
    for idx, img in enumerate(images):
        prefix = f'img_{idx+1}'
        if type(img) == tuple:
            # Roberts, Prewitt, Sobel
            roberts = apply_mask(img[0], roberts_dx, roberts_dy, step_idx, f'{prefix}_roberts')
            prewitt = apply_mask(img[1], prewitt_dx, prewitt_dy, step_idx, f'{prefix}_prewitt')
            sobel = apply_mask(img[2], sobel_dx, sobel_dy, step_idx, f'{prefix}_sobel')
            results.append((roberts, prewitt, sobel))
        else:
            # Roberts, Prewitt, Sobel
            roberts = apply_mask(img, roberts_dx, roberts_dy, step_idx, f'{prefix}_roberts')
            prewitt = apply_mask(img, prewitt_dx, prewitt_dy, step_idx, f'{prefix}_prewitt')
            sobel = apply_mask(img, sobel_dx, sobel_dy, step_idx, f'{prefix}_sobel')
            results.append((roberts, prewitt, sobel))

    return results

def apply_blur(step_idx, images):
    kernel_sizes = {
        "roberts": (7, 7),
        "prewitt": (7, 7),
        "sobel": (7, 7)
    }

    results = []
    for idx, (roberts_grad, prewitt_grad, sobel_grad) in enumerate(images):
        roberts_blur = cv2.GaussianBlur(roberts_grad, kernel_sizes["roberts"], 0)
        prewitt_blur = cv2.GaussianBlur(prewitt_grad, kernel_sizes["prewitt"], 0)
        sobel_blur = cv2.GaussianBlur(sobel_grad, kernel_sizes["sobel"], 0)

        write_image(roberts_blur, f'q1/{step_idx}/img_{idx+1}_roberts_blur.png')
        write_image(prewitt_blur, f'q1/{step_idx}/img_{idx+1}_prewitt_blur.png')
        write_image(sobel_blur, f'q1/{step_idx}/img_{idx+1}_sobel_blur.png')

        results.append((roberts_blur, prewitt_blur, sobel_blur))

    return results

def binarize(images):
    results = []
    for idx, img in enumerate(images):
        prefix = f'img_{idx+1}'
        h, w = img[0].shape
        msb_mask = np.full((h,w), 0x80, dtype=np.uint8)

        roberts = np.bitwise_and(img[0], msb_mask) 
        prewitt = np.bitwise_and(img[1], msb_mask) 
        sobel = np.bitwise_and(img[2], msb_mask) 

        #_, roberts = cv2.threshold(img[0],127,255,cv2.THRESH_BINARY)
        #_, prewitt = cv2.threshold(img[1],127,255,cv2.THRESH_BINARY)
        #_, sobel = cv2.threshold(img[2],127,255,cv2.THRESH_BINARY)

        write_image(roberts, f'q1/step5/{prefix}_roberts_binary.png')
        write_image(prewitt, f'q1/step5/{prefix}_prewitt_binary.png')
        write_image(sobel, f'q1/step5/{prefix}_sobel_binary.png')

        results.append((roberts, prewitt, sobel))

    return results

# Q2 ##################
def apply_mask_q2(color_channel, mask):
    dft_shift = np.fft.fftshift(cv2.dft(np.float32(color_channel), flags=cv2.DFT_COMPLEX_OUTPUT))
    filtered_shift = dft_shift * mask[..., np.newaxis]
    inverse_shift = np.fft.ifftshift(filtered_shift)
    inverse_dft = cv2.idft(inverse_shift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    return cv2.normalize(inverse_dft, None, 0, 255, cv2.NORM_MINMAX)

def create_write_magnitude_spectrum(list_of_image_tuples):
    for name, image in list_of_image_tuples:
        channels = cv2.split(image)
        channel_names = ['B', 'G', 'R']
        
        for idx, channel in enumerate(channels):
            f = np.fft.fft2(channel)
            fshift = np.fft.fftshift(f)   
            magnitude = 20*np.log(1 + np.abs(fshift))
            write_image(magnitude, f"{name}_magnitude_{channel_names[idx]}.png")

def apply_gaussian_blur(list_of_image_tuples):
    for name, ksize, img in list_of_image_tuples:
        blurred_img = cv2.GaussianBlur(src=img, ksize=ksize, sigmaX=0)
        write_image(blurred_img, f"q2/gaussian/{name}_gaussian.png")

def apply_median_blur(list_of_image_tuples):
    for name, ksize, img in list_of_image_tuples:
        blurred_img = cv2.blur(src=img, ksize=ksize)
        write_image(blurred_img, f"q2/median/{name}_median.png")

def ilpf(list_of_image_tuples):
    for name, r, img in list_of_image_tuples:
        rows, cols, _= img.shape
        crow, ccol = rows//2, cols//2

        if isinstance(r, tuple):
            r_b, r_g, r_r = r  # Different cutoff for blue, green, and red
        else:
            r_b = r_g = r_r = r

        # Create mask
        def create_mask(r_value):
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol)**2 + (y - crow)**2)
            mask =  (distance <= r_value).astype(np.uint8)
            return mask
        mask_b = create_mask(r_b)
        mask_g = create_mask(r_g)
        mask_r = create_mask(r_r)

        # Split channels and apply Fourier Transform
        b_img, g_img, r_img = cv2.split(img)
        b_filtered = apply_mask_q2(b_img, mask_b)
        g_filtered = apply_mask_q2(g_img, mask_g)
        r_filtered = apply_mask_q2(r_img, mask_r)

        r_img = cv2.merge((b_filtered, g_filtered, r_filtered))
        
        # You can see the effect of applying ilpf.
        # create_write_magnitude_spectrum([(name+'ilpf',r_img)])

        write_image(r_img, f"q2/ilpf/{name}_ilpf.png")

def bp(list_of_image_tuples):
    for name, (r1, r2), img in list_of_image_tuples:
        rows, cols, _ = img.shape
        crow, ccol = rows // 2, cols // 2

        # Handle tuple or single value for r1 and r2
        if isinstance(r1, tuple):
            r1_b, r1_g, r1_r = r1
        else:
            r1_b = r1_g = r1_r = r1

        if isinstance(r2, tuple):
            r2_b, r2_g, r2_r = r2
        else:
            r2_b = r2_g = r2_r = r2

        # Create masks for each channel
        def create_mask(r1_value, r2_value):
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
            mask =  ((distance >= r1_value) & (distance <= r2_value)).astype(np.uint8)
            return mask

        mask_b = create_mask(r1_b, r2_b)
        mask_g = create_mask(r1_g, r2_g)
        mask_r = create_mask(r1_r, r2_r)

        # Split channels and apply Fourier Transform
        b_img, g_img, r_img = cv2.split(img)
        b_filtered = apply_mask_q2(b_img, mask_b)
        g_filtered = apply_mask_q2(g_img, mask_g)
        r_filtered = apply_mask_q2(r_img, mask_r)

        r_img = cv2.merge((b_filtered, g_filtered, r_filtered))

        # You can see the effect of applying bp.
        # create_write_magnitude_spectrum([(name+'bp',r_img)])

        write_image(r_img, f"q2/bp/{name}_bp.png")


def br(list_of_image_tuples):
    for name, (r1, r2), img in list_of_image_tuples:
        rows, cols, _ = img.shape
        crow, ccol = rows // 2, cols // 2

        # Different cutoff for different channels
        if isinstance(r1, tuple):
            r1_b, r1_g, r1_r = r1
        else:
            r1_b = r1_g = r1_r = r1

        if isinstance(r2, tuple):
            r2_b, r2_g, r2_r = r2
        else:
            r2_b = r2_g = r2_r = r2

        # Create masks for each channel
        def create_mask(r1_value, r2_value):
            y, x = np.ogrid[:rows, :cols]
            distance = np.sqrt((x - ccol) ** 2 + (y - crow) ** 2)
            mask = ((distance <= r1_value) | (distance >= r2_value)).astype(np.uint8)
            return mask

        mask_b = create_mask(r1_b, r2_b)
        mask_g = create_mask(r1_g, r2_g)
        mask_r = create_mask(r1_r, r2_r)

        # Split channels and apply Fourier Transform
        b_img, g_img, r_img = cv2.split(img)
        b_filtered = apply_mask_q2(b_img, mask_b)
        g_filtered = apply_mask_q2(g_img, mask_g)
        r_filtered = apply_mask_q2(r_img, mask_r)

        r_img = cv2.merge((b_filtered, g_filtered, r_filtered))

        # You can see the effect of applying br.
        # create_write_magnitude_spectrum([(name+'br',r_img)])

        write_image(r_img, f"q2/br/{name}_br.png")
################## Q2 END ##################

def create_necessary_folders():
    folders = {
        "q1": ["step1", "step2", "step3", "step4", "step5", "step6", "step7", "step8"],
        "q2": ["bp", "br", "gaussian", "ilpf", "median"],
    }

    # Create the folders
    for parent, subfolders in folders.items():
        parent_path = os.path.join(output_folder, parent)
        os.makedirs(parent_path, exist_ok=True)  # Create the parent folder if it doesn't exist
        for subfolder in subfolders:
            os.makedirs(os.path.join(parent_path, subfolder), exist_ok=True)  # Create subfolders

if __name__ == '__main__':

    create_necessary_folders()
    # Q1 - Pattern Extraction
    # Step 1: Convert to grayscale
    img_a1 = read_image('a1.png')
    img_a2 = read_image('a2.png')
    gray_img_a1 = cv2.cvtColor(img_a1, cv2.COLOR_BGR2GRAY)
    gray_img_a2 = cv2.cvtColor(img_a2, cv2.COLOR_BGR2GRAY)
    write_image(gray_img_a1, 'q1/step1/a1_grayscale.png')
    write_image(gray_img_a2, 'q1/step1/a2_grayscale.png')

    # Step 2: Apply edge detection
    results = apply_edge_detectors('step2', [gray_img_a1, gray_img_a2])

    # Step 3: Blur the images
    results = apply_blur('step3', results)

    # Step 4: Apply edge detection
    results = apply_edge_detectors('step4', results)

    # Step 5: Binarize the gray scale image, most sig bit
    results = binarize(results)

    # Step 6: Apply edge detection
    results = apply_edge_detectors('step6', results)

    # Step 7: Blur the images
    results = apply_blur('step7', results)

    # Step 8: Apply edge detection
    results = apply_edge_detectors('step8', results)

    ###################### Q2
    # Q2 - Image Enhancement
    img_b1 = read_image('b1.jpg')
    img_b2 = read_image('b2.jpg')
    img_b3 = read_image('b3.jpg')

    # Arbitrary cutoff are used, change whenever needed.
    # Spatial domain
    # Arguments type: (Image name, kernel size, image)
    # Gaussian Filter
    apply_gaussian_blur([('b1', (21,21), img_b1), ('b2', (5,5), img_b2), ('b3', (5,5), img_b3)])

    # Median Filter
    apply_median_blur([('b1', (15,15), img_b1), ('b2', (11,11), img_b2), ('b3', (7,7), img_b3)])

    # Frequency domain
    create_write_magnitude_spectrum([('b1', img_b1), ('b2', img_b2), ('b3', img_b3)])
    # Arguments type: (Image name, cutoff, image)
    # Ideal Low Pass Filter
    ilpf([('b1', (50, 35, 35), img_b1), ('b2', 30, img_b2), ('b3', (35, 50, 35), img_b3)])

    # Band Pass Filter
    common_cutoff=((3,3,3),(40,40,40))
    bp([('b1', common_cutoff, img_b1), ('b2', common_cutoff, img_b2), ('b3', common_cutoff, img_b3)])

    # Band Reject Filter
    common_cutoff=((35, 50, 30),(220, 220, 220))
    br([('b1', ((100, 50, 30),(150, 150, 150)), img_b1), ('b2', common_cutoff, img_b2), ('b3', common_cutoff, img_b3)])

    ###################### Q3