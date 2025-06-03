# In[1]:



def process_and_display_fourier_zone_filter(image, zone_code):
    import numpy as np

    # Load image and convert to grayscale
    image_gray = np.mean(image, axis=2)  # Uses float64 by default
    zone_code = str(zone_code)
    num_zones = len(zone_code)

    # Setup
    rows, cols = image_gray.shape
    cx, cy = cols // 2, rows // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Zone thresholds based on image width (not diagonal)
    max_radius = 0.5 * cols
    normalized_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    thresholds = [t * max_radius for t in normalized_thresholds]

    # Fourier Transform
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    original_spectrum = np.log1p(np.abs(fshift))
    original_spectrum = 255 * (original_spectrum - original_spectrum.min()) / (np.ptp(original_spectrum) + 1e-8)

    # Build mask and apply filter
    mask = np.zeros_like(image_gray)
    zone_mask_visual = np.zeros_like(image_gray)
    for i in range(num_zones):
        ring_mask = (distance >= thresholds[i]) & (distance < thresholds[i + 1])
        if zone_code[i] == '1':
            mask[ring_mask] = 1
        zone_mask_visual[ring_mask] = int(255 * (i + 1) / num_zones)

    fshift_filtered = fshift * mask

    # Filtered spectrum
    filtered_spectrum = np.log1p(np.abs(fshift_filtered))
    filtered_spectrum = 255 * (filtered_spectrum - filtered_spectrum.min()) / (np.ptp(filtered_spectrum) + 1e-8)

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = 255 * (img_back - img_back.min()) / (np.ptp(img_back) + 1e-8)

    return img_back, filtered_spectrum
    

    




# In[ ]:
