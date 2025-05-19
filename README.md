1. Fourier_Spectrum
   This script reads an input image (Earth.png), converts it to grayscale, and computes its 2D Fourier Transform using NumPy. It then displays:
    The original grayscale image 
    The log-magnitude spectrum of its Fourier Transform
2.The_inverse_Fourier_Transform
   Image reconstruction progressively using pairs of symmetric frequency components from its 2D Fourier Transform.
   Starts from the most central frequencies (low frequency) and gradually adds higher frequencies.
   Visualizes each step of the reconstruction alongside the individual sinusoidal (grating) component.
   Helps understand the contribution of each frequency to the overall image.
3. Module_Fourier_zone_filter
    Applies a custom Fourier filter based on concentric frequency zones.
    Parameters:
    - image: 2D grayscale image (NumPy array)
    - zone_code: String like '1010' (keep zones where value is '1')
    Returns:
    - Filtered image (NumPy array)
    - Filtered spectrum (log-magnitude, float32)
4.Process_and_display_fourier_zone_filter
5.GeoTIFF_Filtered_Saved
