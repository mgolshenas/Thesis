import numpy as np
import rasterio
import os
import gc

# Fourier filter function (returns float32)
def fourier_concentric_zone_filter(image, zone_code):
    zone_code = str(zone_code)
    num_zones = len(zone_code)

    img = image.astype(np.float32)
    rows, cols = img.shape
    cx, cy = cols // 2, rows // 2

    # FFT and shift
    f = np.fft.fft2(img).astype(np.complex64)
    fshift = np.fft.fftshift(f)

    # Create distance mask
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - cx)**2 + (y - cy)**2)
    max_radius = np.max(distance)
    thresholds = [max_radius * i / num_zones for i in range(num_zones + 1)]

    # Create mask
    mask = np.zeros_like(img, dtype=np.float32)
    for i in range(num_zones):
        if zone_code[i] == '1':
            mask[(distance >= thresholds[i]) & (distance < thresholds[i + 1])] = 1

    # Apply filter
    fshift_filtered = fshift * mask

    # Inverse FFT
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back).astype(np.float32)

    return img_back

# Paths
input_path = r"C:\Users\M\Downloads\DEM01.gtif.tif"
input_path = input_path.encode('ascii', 'ignore').decode()

output_folder = r"C:\Users\M\Downloads\Output"
os.makedirs(output_folder, exist_ok=True)

# Generate 4-digit binary zone codes (excluding 0000)
zone_codes = [format(i, '04b') for i in range(1, 16)]

# Process image
with rasterio.open(input_path) as src:
    image = src.read(1)

    for zone_code in zone_codes:
        filtered_img = fourier_concentric_zone_filter(image, zone_code)

        output_path = os.path.join(output_folder, f"filtered_img_{zone_code}.tif")

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
        )

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(filtered_img, 1)

        print(f"Saved: {output_path}")
        gc.collect()


    
