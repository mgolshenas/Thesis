import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def fourier_concentric_zone_filter(image, zone_code):
    zone_code = str(zone_code)
    num_zones = len(zone_code)

    img = image.astype(np.float32)
    rows, cols = img.shape
    cx, cy = cols // 2, rows // 2

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)

    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radius = np.max(distance)

    thresholds = [max_radius * i / num_zones for i in range(num_zones + 1)]

    mask = np.zeros_like(img, dtype=np.float32)
    for i in range(num_zones):
        if zone_code[i] == '1':
            mask[(distance >= thresholds[i]) & (distance < thresholds[i + 1])] = 1

    fshift_filtered = fshift * mask

    spectrum = np.log(1 + np.abs(fshift_filtered))
    spectrum = 255 * (spectrum - spectrum.min()) / (np.ptp(spectrum) + 1e-8)
    spectrum = spectrum.astype(np.uint8)

    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    img_back = 255 * (img_back - img_back.min()) / (np.ptp(img_back) + 1e-8)
    img_back = img_back.astype(np.float32)

    return img_back, spectrum

def plot_fourier_zones_and_spectrum(image_gray, zone_code, filtered_image, filtered_spectrum):
    rows, cols = image_gray.shape
    cx, cy = cols // 2, rows // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)

    num_zones = len(zone_code)
    max_radius_visual = np.max(distance)
    thresholds_visual = [max_radius_visual * i / num_zones for i in range(num_zones + 1)]

    zone_mask_visual = np.zeros_like(image_gray, dtype=np.float32)
    for i in range(num_zones):
        if zone_code[i] == '1':
            zone_mask_visual[(distance >= thresholds_visual[i]) & (distance < thresholds_visual[i + 1])] = int(255 * (i + 1) / num_zones)

    fig, ax = plt.subplots(1, 4, figsize=(24, 6))

    ax[0].imshow(image_gray, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(filtered_image, cmap='gray')
    ax[1].set_title('Filtered Image')
    ax[1].axis('off')

    ax[2].imshow(filtered_spectrum, cmap='gray')
    ax[2].set_title('Filtered Fourier Spectrum')
    ax[2].axis('off')

    ax[3].imshow(zone_mask_visual, cmap='gray')
    ax[3].set_title(f'Zone Mask ({zone_code})')
    for i in range(num_zones):
        r = (thresholds_visual[i] + thresholds_visual[i + 1]) / 2
        label_x = cx + r / np.sqrt(2)
        label_y = cy - r / np.sqrt(2)
        ax[3].text(label_x, label_y, f'zone {i + 1}', color='black', fontsize=16, fontweight='bold')
    ax[3].axis('off')

    plt.tight_layout()
    plt.show()

# === Load and process image ===
image = data.astronaut()
image_gray = np.mean(image, axis=2).astype(np.float32)

zone_code = '1101'
filtered_image, filtered_spectrum = fourier_concentric_zone_filter(image_gray, zone_code)

# === Plot everything ===
plot_fourier_zones_and_spectrum(image_gray, zone_code, filtered_image, filtered_spectrum)
