1. Fourier_Spectrum

   Purpose:
   Reads an image (Earth.png), converts it to grayscale, and computes its 2D Fourier Transform. It displays both the image and its frequency spectrum.
   Main Functionalities:
      cv2.imread to read the image.
      np.fft.fft2 and np.fft.fftshift to compute the FFT.
   Displays:
      Grayscale image.
      Log-magnitude Fourier spectrum.
2. GeoTIFF_Filtered_Spectrum
    
    -Input & Output Setup
      Loads a GeoTIFF (DEM01.gtif.tif) using rasterio.

      Creates an output folder for storing results.

    -Zone Code Generation
   Generates binary zone codes from 0001 to 1111 (i.e., integers 1–15).

   Each 4-bit code controls which of the 4 frequency zones to keep.

   - Processing Loop
   For each zone_code:

   Reads the input image.

   Applies the process_and_display_fourier_zone_filter() function.

   Saves:

   filtered_img_{zone_code}.tif – Spatial-domain filtered image.

   spectrum_{zone_code}.tif – Frequency spectrum after masking.
 3. Module_Fourier_zone_filter

      Applies a custom Fourier filter based on concentric frequency zones.
       Parameters:
       - image: 2D grayscale image (NumPy array)
       - zone_code: String like '1010' (keep zones where value is '1')
       Returns:
       - Filtered image (NumPy array)
       - Filtered spectrum (log-magnitude, float32)
    
   
4. Module_process_and_display_fourier_zone_filter

   Parameters
         image: 2D NumPy array (image) 
         zone_code: 4-character binary string (e.g., "1100"), where each digit enables/disables a radial frequency band:

      Processing Steps
      - Normalize & Convert Image: Converts the input image to float32 for processing.

      - FFT & Frequency Masking:

      - Computes the 2D Fourier Transform.

      - Constructs a Concentric frequency mask with 4 radial bands.

      - Keeps or discards bands based on the zone_code.

   - Filter Application:

      -Applies the mask to the frequency spectrum.

      -Computes the inverse FFT to reconstruct the filtered image.

      -Spectrum Visualization:

        Returns the filtered spectrum image (for visualization).

        Returns
        img_back: The filtered image (spatial domain).

        filtered_spectrum: The masked frequency magnitude spectrum.
   
  5. The_inverse_Fourier_Transform

      Image reconstruction progressively using pairs of symmetric frequency components from its 2D Fourier Transform.
      Starts from the most central frequencies (low frequency) and gradually adds higher frequencies.
      Visualizes each step of the reconstruction alongside the individual sinusoidal (grating) component.
      Helps understand the contribution of each frequency to the overall image.
