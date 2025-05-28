1. Module_Fourier_Spectrum

   Purpose:
   Reads an image (Earth.png), converts it to grayscale, and computes its 2D Fourier Transform. It displays both the image and its frequency spectrum.

   Main Functionalities:
      cv2.imread to read the image.
      np.fft.fft2 and np.fft.fftshift to compute the FFT.

   Displays:
      Grayscale image.
      Log-magnitude Fourier spectrum.

2. GeoTIFF_Filtered_Spectrum
    
    Purpose:
      Applies zone-based frequency filtering to a GeoTIFF image (DEM01.gtif.tif), saving filtered outputs for multiple frequency masks.

   Input & Output Setup:

      Loads image via rasterio.

   Creates output folder.

      Zone Code Generation:

      Generates 4-bit binary codes (0001 to 1111) to selectively enable frequency zones.

      Processing Loop:

      For each zone_code:

      Calls process_and_display_fourier_zone_filter() from the module.

   Saves:

      filtered_img_{zone_code}.tif (spatial image).

      spectrum_{zone_code}.tif (Fourier spectrum).
   
 3. Module_Fourier_zone_filter
    
    Purpose:
      Reusable function to apply a concentric-zone frequency filter using a binary zone code.
      def fourier_zone_filter(image: np.ndarray, zone_code: str) -> Tuple[np.ndarray, np.ndarray]
    
    Parameters:

      image: 2D grayscale NumPy array.

      zone_code: 4-character binary string like '1010'.
    
    Returns:

      filtered_img: Filtered image in the spatial domain.

      filtered_spectrum: Log-magnitude spectrum after filtering
   
4. Module_process_and_display_fourier_zone_filter

   Purpose:
      Handles the full processing pipeline: FFT, zone filtering, inverse FFT, and spectrum visualization.
      def process_and_display_fourier_zone_filter(image: np.ndarray, zone_code: str) -> Tuple[np.ndarray, np.ndarray]
     Parameters:

      image: Grayscale image (2D NumPy array).

      zone_code: 4-digit binary string controlling which radial frequency bands to preserve.

   Steps:

      Normalize and convert image.
   
      Compute FFT and shift spectrum.

      Generate frequency mask from zone_code.

      Apply mask, inverse FFT.

      Return filtered image and spectrum for display.

   
  5. The_inverse_Fourier_Transform
     
     Purpose:
      Visualizes how different frequency components reconstruct an image over time.

      Main Functionalities:

      Starts with low-frequency pairs and adds higher frequencies step-by-step.

      Shows how each pair contributes to the final image.

      Educational visualization for understanding image reconstruction from FFT.
6. Transformation_GEOTIFF_Filter
      Processes a GeoTIFF using Fourier zone filtering. For each of 15 zone codes (binary values from '0001' to '1111', excluding '0000'), it       performs the following steps:

 


      
