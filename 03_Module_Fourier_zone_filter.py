

# In[1]:


#fourier_concentric_zone_filter

import numpy as np
import matplotlib.pyplot as plt


def fourier_concentric_zone_filter(image, zone_code):
    """
    Applies a custom Fourier filter based on concentric frequency zones.

    Parameters:
    - image: 2D grayscale image (NumPy array)
    - zone_code: String like '1010' (keep zones where value is '1')

    Returns:
    - Filtered image (NumPy array)
    - Filtered spectrum (log-magnitude, float32)
    """
    zone_code = str(zone_code)
    num_zones = len(zone_code)

    img = image.astype(float)  # Safe type for FFT and numeric ops
    rows, cols = img.shape
    cx, cy = cols // 2, rows // 2

    # FFT and shift
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    # Create meshgrid of distances from center
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radius = np.max(distance)

    # Define zone thresholds
    thresholds = [max_radius * i / num_zones for i in range(num_zones + 1)]

    # Create mask
    mask = np.zeros_like(img)
    for i in range(num_zones):
        if zone_code[i] == '1':
            mask[(distance >= thresholds[i]) & (distance < thresholds[i + 1])] = 1

    # Apply filter in frequency domain
    fshift_filtered = fshift * mask

    # Create log-magnitude spectrum for visualization
    spectrum = np.log1p(np.abs(fshift_filtered))
    spectrum = 255 * (spectrum - spectrum.min()) / (np.ptp(spectrum) + 1e-8)

    # Inverse FFT to reconstruct image
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = 255 * (img_back - img_back.min()) / (np.ptp(img_back) + 1e-8)

    return img_back, spectrum

# In[ ]:




