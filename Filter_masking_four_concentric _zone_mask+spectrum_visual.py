# Python --version 3.12.3
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def process_and_display_fourier_zone_filter(zone_code):
    # Load image and convert to grayscale
    image = data.astronaut()
    image_gray = np.mean(image, axis=2).astype(np.float32)
    zone_code = str(zone_code) 
    num_zones = len(zone_code)

    # Setup
    rows, cols = image_gray.shape
    cx, cy = cols // 2, rows // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radius = np.max(distance)

    # Define normalized thresholds for zones
    normalized_thresholds = [0.0, 0.25, 0.5, 0.75, 1.0]
    thresholds = [t * max_radius for t in normalized_thresholds]

    # Fourier Transform
    f = np.fft.fft2(image_gray)
    fshift = np.fft.fftshift(f)
    original_spectrum = np.log(1 + np.abs(fshift))
    original_spectrum = 255 * (original_spectrum - original_spectrum.min()) / (np.ptp(original_spectrum) + 1e-8)
    original_spectrum = original_spectrum.astype(np.uint8)

    # Build mask and apply filter
    mask = np.zeros_like(image_gray, dtype=np.float32)
    zone_mask_visual = np.zeros_like(image_gray, dtype=np.float32)
    for i in range(num_zones):
        ring_mask = (distance >= thresholds[i]) & (distance < thresholds[i + 1])
        if zone_code[i] == '1':
            mask[ring_mask] = 1
        zone_mask_visual[ring_mask] = int(255 * (i + 1) / num_zones)

    fshift_filtered = fshift * mask

    # Filtered spectrum
    filtered_spectrum = np.log(1 + np.abs(fshift_filtered))
    filtered_spectrum = 255 * (filtered_spectrum - filtered_spectrum.min()) / (np.ptp(filtered_spectrum) + 1e-8)
    filtered_spectrum = filtered_spectrum.astype(np.uint8)

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = 255 * (img_back - img_back.min()) / (np.ptp(img_back) + 1e-8)
    img_back = img_back.astype(np.float32)

    # Plotting
    fig, ax = plt.subplots(1, 5, figsize=(30, 6))
    ax[0].imshow(image_gray, cmap='gray')
    ax[0].set_title('Original Image')

    ax[1].imshow(img_back, cmap='gray')
    ax[1].set_title('Filtered Image')

    ax[2].imshow(original_spectrum, cmap='gray')
    ax[2].set_title('Original Fourier Spectrum')

    ax[3].imshow(filtered_spectrum, cmap='gray')
    ax[3].set_title(f'Filtered Spectrum\n(Code: {zone_code})')

    # Zone mask visualization
    ax[4].imshow(zone_mask_visual, cmap='gray')
    ax[4].set_title(f'Zone Mask ({zone_code})')

    # Draw zone labels and radii lines
    for i in range(num_zones):
        # Zone label
        mid_norm_r = (normalized_thresholds[i] + normalized_thresholds[i + 1]) / 2
        r = mid_norm_r * max_radius
        label_x = cx + r / np.sqrt(2)
        label_y = cy - r / np.sqrt(2)
        ax[4].text(label_x, label_y, f'zone {i + 1}', color='black', fontsize=16, fontweight='bold')

    # Add radius width/lines and text
    for t in normalized_thresholds[1:-1]:  # skip 0.0 and 1.0
        r = t * max_radius
        # Horizontal line
        ax[4].plot([cx, cx + r], [cy, cy], color='gold', linewidth=1.5)
        # Vertical line
        ax[4].plot([cx + r, cx + r], [cy, cy + 20], color='gold', linewidth=1)
        # Text label
        ax[4].text(cx + r - 10, cy + 30, f'{t:.2f}', color='black', fontsize=12, fontweight='bold')

    ax[4].axis('off')
    for a in ax[:4]:
        a.axis('off')

    plt.tight_layout()
    plt.show()


# === Run the function with a zone code ===
process_and_display_fourier_zone_filter('1101')
