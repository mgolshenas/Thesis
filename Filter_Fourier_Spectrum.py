import numpy as np
import matplotlib.pyplot as plt

def apply_fourier_filter(image, filter_radius):
    """
    Apply a circular low-pass filter to the Fourier spectrum of a grayscale image.

    Args:
        image (ndarray): Input 2D grayscale image.
        filter_radius (int): Radius of the circular low-pass filter.

    Returns:
        filtered_image (ndarray): Filtered image after inverse Fourier transform.
    """
    # Step 1: Compute 2D FFT and shift
    dft = np.fft.fft2(image)
    dft_shift = np.fft.fftshift(dft)

    # Step 2: Create circular mask
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - ccol)**2 + (Y - crow)**2) #Euclidean distance formula
    mask = distance <= filter_radius

    # Step 3: Apply mask
    filtered_dft = dft_shift * mask

    # Step 4: Inverse FFT
    idft_shift = np.fft.ifftshift(filtered_dft)
    filtered_image = np.fft.ifft2(idft_shift)
    filtered_image = np.abs(filtered_image)

    return filtered_image
