print("Core/processor.py script started execution at the very top.")

import numpy as np
from astropy.stats import sigma_clip # For kappa-sigma clipping, replacing scipy.stats.sigmaclip

print("Imports in core/processor.py completed.")

class ImageProcessor:
    def __init__(self):
        # It's good practice to initialize any default parameters or state here if needed.
        # For now, processor is stateless between calls for these functions.
        pass

    # --- Alignment and Stacking Methods ---

    def align_images(self, images: list[np.ndarray], method: str = 'none') -> list[np.ndarray]:
        """
        Aligns a list of images.
        For now, this is a placeholder. Actual alignment is complex.
        If method is 'none', it returns the images as is, assuming they are pre-aligned.

        Args:
            images: A list of NumPy arrays, each representing an image.
                    Assumes all images have the same dimensions.
            method: Alignment method (e.g., 'feature_based', 'phase_cross_correlation').
                    Currently only 'none' is implemented.

        Returns:
            A list of aligned NumPy arrays.
            Or the original list if alignment fails or method is 'none'.
        """
        if not images:
            return []
        
        if method.lower() == 'none':
            print("Alignment method is 'none'. Images will be stacked without explicit alignment.")
            return images
        else:
            # Placeholder for actual alignment logic (e.g., using OpenCV, scikit-image)
            # This would involve:
            # 1. Selecting a reference image (e.g., the first one, or highest quality).
            # 2. For each other image:
            #    a. Detect features or use correlation to find transformation matrix.
            #    b. Warp the image to align with the reference.
            # This is a significant task and will be implemented later.
            print(f"Warning: Alignment method '{method}' is not yet implemented. Returning original images.")
            return images

    def stack_images(self, images: list[np.ndarray], method: str = 'average', 
                     kappa: float = 2.0, max_iterations: int = 5) -> np.ndarray | None:
        """
        Stacks a list of images using the specified method.

        Args:
            images: A list of NumPy arrays (images). Assumes they are aligned and have the same dimensions.
            method: Stacking method. Options: 'average', 'median', 'kappa_sigma_clip'.
            kappa: Kappa value for sigma clipping (if method is 'kappa_sigma_clip').
                   Number of standard deviations to clip at.
            max_iterations: Maximum iterations for kappa-sigma clipping.

        Returns:
            A NumPy array representing the stacked image, or None if an error occurs.
        """
        if not images:
            print("Error: No images provided for stacking.")
            return None

        # Ensure all images are NumPy arrays and have the same shape
        try:
            image_stack = np.stack([np.asarray(img, dtype=np.float32) for img in images], axis=0)
        except ValueError as e:
            print(f"Error: Images must have the same dimensions to be stacked. {e}")
            # Optionally, print shapes of images for debugging
            # for i, img in enumerate(images):
            #     print(f"Image {i} shape: {np.asarray(img).shape}")
            return None
        
        if image_stack.ndim != 3: # (num_images, height, width)
            print(f"Error: Image stack has unexpected dimensions: {image_stack.shape}. Expected 3D array.")
            return None

        print(f"Stacking {image_stack.shape[0]} images of size {image_stack.shape[1:]} using '{method}' method.")

        if method.lower() == 'average':
            stacked_image = np.mean(image_stack, axis=0)
        elif method.lower() == 'median':
            stacked_image = np.median(image_stack, axis=0)
        elif method.lower() == 'kappa_sigma_clip':
            # Perform kappa-sigma clipping pixel by pixel.
            # This can be memory intensive if done naively for large images by creating many copies.
            # scipy.stats.sigmaclip can operate on an array and returns the clipped values.
            # We want to calculate the mean of the values that are *not* clipped at each pixel position.
            
            # Reshape for pixel-wise processing: (num_pixels, num_images)
            num_images, height, width = image_stack.shape
            pixel_values_over_time = image_stack.reshape(num_images, height * width).T # Shape: (height*width, num_images)
            
            stacked_pixels = np.empty(height * width, dtype=np.float32)

            for i in range(height * width):
                values_for_pixel = pixel_values_over_time[i, :]
                
                # Debug print for the specific pixel in the test case (0,2) which is i=2 for a 3x3 image
                if width == 3 and height == 3 and i == (0 * width + 2): # (row 0, col 2)
                    print(f"\nDebug for pixel index i={i} (target pixel 0,2):")
                    print(f"  values_for_pixel = {values_for_pixel}")
                    print(f"  kappa = {kappa}")

                # astropy's sigma_clip returns a masked array
                # Using cenfunc='median' for more robustness against extreme outliers.
                clipped_masked_array = sigma_clip(values_for_pixel, sigma=kappa, maxiters=max_iterations, cenfunc='median', stdfunc='std')
                
                # Get the data from the masked array where mask is False (i.e., not clipped)
                valid_values = clipped_masked_array.data[~clipped_masked_array.mask]
                
                if width == 3 and height == 3 and i == (0 * width + 2): # Debug for specific pixel
                    print(f"  sigma_clip results for pixel {i} (cenfunc='median'):")
                    print(f"    Input values: {values_for_pixel}")
                    print(f"    Masked array: {clipped_masked_array}")
                    print(f"    Valid values (unmasked): {valid_values}")


                if valid_values.size > 0:
                    stacked_pixels[i] = np.mean(valid_values)
                else:
                    # If all values are clipped, fall back to mean of original values for that pixel
                    print(f"Warning: All values clipped for pixel index {i} (values: {values_for_pixel}). Using mean of original values.")
                    stacked_pixels[i] = np.mean(values_for_pixel) 
            
            stacked_image = stacked_pixels.reshape(height, width)

            # Alternative kappa-sigma using iterative clipping (more robust but slower)
            # This is a more standard implementation of iterative sigma clipping:
            # 1. Calculate mean and std of the stack along the image axis.
            # 2. Create a mask for values outside mean +/- kappa * std.
            # 3. Exclude masked values and recalculate mean and std.
            # 4. Repeat until no more values are masked or max_iterations is reached.
            # 5. Final stack is the mean of the unmasked pixels.
            #
            # This is more complex to implement efficiently pixel-wise without large memory overhead.
            # The scipy.stats.sigmaclip is a good compromise for a first pass.
            # For a more robust per-pixel iterative approach:
            # if method.lower() == 'kappa_sigma_clip_iterative': # Example for a more advanced method
            #     print("Using iterative kappa-sigma clipping (placeholder - using scipy.stats.sigmaclip for now)")
            #     # ... (implementation would go here) ...

        else:
            print(f"Error: Unknown stacking method '{method}'. Supported: 'average', 'median', 'kappa_sigma_clip'.")
            return None

        return stacked_image.astype(np.float32)

    # --- Basic Image Editing Methods ---

    def adjust_brightness_contrast(self, image: np.ndarray, brightness: float = 0.0, contrast: float = 1.0) -> np.ndarray | None:
        """
        Adjusts the brightness and contrast of an image.
        Assumes image data is float, typically in range [0, ...] (e.g. [0,1] or [0,65535] post-calibration).
        Brightness: Additive. 0 means no change. Positive values brighten, negative values darken.
        Contrast: Multiplicative. 1 means no change. >1 increases contrast, <1 decreases.

        Args:
            image: NumPy array of the image.
            brightness: Float value for brightness adjustment.
                        Interpreted relative to the image's current range after initial scaling.
                        E.g., if image is 0-1, brightness 0.1 shifts values up by 0.1.
            contrast: Float value for contrast adjustment. Must be non-negative.

        Returns:
            Adjusted NumPy array, or None if input is invalid.
        """
        if image is None:
            print("Error: No image provided for brightness/contrast adjustment.")
            return None
        if contrast < 0:
            print("Error: Contrast value must be non-negative.")
            return image # Or None, depending on desired strictness

        # For contrast, it's often applied around a mean or mid-point.
        # A simple approach: (value - mean) * contrast + mean + brightness
        # Or, if data is normalized 0-1: (value - 0.5) * contrast + 0.5 + brightness
        
        # Let's assume the image is float and we operate directly on its values.
        # A common formula is: new_value = (old_value - 0.5) * contrast + 0.5 + brightness (for 0-1 range)
        # Or more generally: new_value = old_value * contrast + brightness (simpler, but contrast pivot is 0)

        # Using a more standard formula: output = image * contrast + brightness
        # This is simple but contrast pivot is 0.
        # A better way for contrast: scale around the mean of the image or a fixed mid-point (e.g., 0.5 if normalized)
        
        img_float = image.astype(np.float32)
        
        # To make brightness and contrast adjustments more intuitive,
        # let's consider the image data range. If it's not normalized (0-1),
        # the effect of a fixed brightness value can vary.
        # For now, apply directly. User needs to be aware of data range.
        
        # Contrast adjustment: (value - pivot) * contrast_factor + pivot
        # Using image mean as pivot
        mean_val = np.mean(img_float)
        adjusted_image = (img_float - mean_val) * contrast + mean_val
        
        # Brightness adjustment
        adjusted_image = adjusted_image + brightness
        
        # Clipping might be necessary if the operations push values out of a displayable range (e.g. 0-255 or 0-1)
        # However, astrophotos often have high dynamic range. Clipping should be a deliberate step,
        # usually at the end (e.g., when saving to an LDR format or for display).
        # For internal processing, it's often better to keep the float values unclipped.
        # We can add a min_val, max_val clipping as an option later if needed.
        # For example: np.clip(adjusted_image, 0, 1.0) if data is normalized 0-1.
        
        return adjusted_image

    def adjust_levels(self, image: np.ndarray, black_point: float, white_point: float, mid_point: float = 1.0) -> np.ndarray | None:
        """
        Adjusts the image levels (input black, input white, gamma/mid-point).
        Assumes image data is float.
        Input levels are mapped to output levels 0 and 1 (or min/max of data type if not float).

        Args:
            image: NumPy array of the image.
            black_point: The input level to be mapped to black (output 0).
            white_point: The input level to be mapped to white (output 1).
            mid_point: The gamma correction factor (power law). 1.0 means linear.
                       > 1.0 makes mid-tones brighter, < 1.0 makes mid-tones darker.

        Returns:
            Adjusted NumPy array.
        """
        if image is None:
            return None
        if black_point >= white_point:
            print("Error: Black point must be less than white point.")
            return image # Or None

        img_float = image.astype(np.float32)
        
        # Scale the image based on black_point and white_point
        # Values below black_point become 0, values above white_point become 1 (or original max)
        # (image - black_point) / (white_point - black_point)
        adjusted_image = (img_float - black_point) / (white_point - black_point)
        
        # Clip to [0, 1] range after scaling
        adjusted_image = np.clip(adjusted_image, 0.0, 1.0)
        
        # Apply gamma correction (mid_point)
        if mid_point != 1.0 and mid_point > 0:
            # Gamma is applied as value^(1/gamma_factor)
            # Our mid_point is the gamma_factor.
            adjusted_image = np.power(adjusted_image, 1.0 / mid_point)
        elif mid_point <= 0:
            print("Warning: Mid-point (gamma) must be positive. Ignoring gamma correction.")
            
        # The output is now normalized to [0, 1].
        # Depending on subsequent processing or display, it might need to be rescaled
        # to the original data's typical range or a display range (e.g., 0-255).
        # For now, return the [0,1] normalized image.
        return adjusted_image

    def adjust_curves(self, image: np.ndarray, points: list[tuple[float, float]]) -> np.ndarray | None:
        """
        Adjusts the image tones using a user-defined curve.
        Assumes input image is float, typically normalized to [0, 1] before applying curves.
        Points define the curve: (input_value, output_value).
        Input/output values for points are also expected to be in [0, 1] range.

        Args:
            image: NumPy array of the image (ideally normalized to [0,1]).
            points: A list of (x, y) tuples defining points on the curve.
                    Must include (0,0) and (1,1) for a full mapping, or values will be
                    extrapolated/clamped based on np.interp's behavior.
                    Points must be sorted by x-values for np.interp.

        Returns:
            Adjusted NumPy array.
        """
        if image is None:
            return None
        if not points:
            print("Warning: No points provided for curves adjustment. Returning original image.")
            return image

        img_float = image.astype(np.float32)

        # Ensure points are sorted by x-value and cover the 0-1 range for predictable results.
        # It's good practice for the calling UI to ensure (0,0) and (1,1) are part of points
        # or handled appropriately if not.
        # For simplicity, we assume points are reasonably well-behaved here.
        # Extract x and y coordinates for interpolation
        x_coords = np.array([p[0] for p in points])
        y_coords = np.array([p[1] for p in points])

        # Sort points by x_coords to ensure np.interp works correctly
        sort_indices = np.argsort(x_coords)
        x_coords_sorted = x_coords[sort_indices]
        y_coords_sorted = y_coords[sort_indices]
        
        # Interpolate image values using the curve
        # np.interp requires the x-points for interpolation (xp) to be increasing.
        adjusted_image = np.interp(img_float, x_coords_sorted, y_coords_sorted)
        
        # Output of np.interp will be within the range of y_coords_sorted.
        # If y_coords are within [0,1], then adjusted_image will also be (mostly) within [0,1].
        # Clipping might be good practice if y_coords can go outside [0,1].
        adjusted_image = np.clip(adjusted_image, 0.0, 1.0)

        return adjusted_image

    def sharpen_image(self, image: np.ndarray, strength: float = 1.0, method: str = 'unsharp_mask') -> np.ndarray | None:
        """
        Sharpens an image.

        Args:
            image: NumPy array of the image.
            strength: Sharpening strength/amount. Interpretation depends on method.
                      For unsharp_mask, it's the multiplication factor for the high-pass signal.
            method: 'unsharp_mask' or 'convolution'.

        Returns:
            Sharpened NumPy array.
        """
        if image is None:
            return None
        
        img_float = image.astype(np.float32)

        if method.lower() == 'unsharp_mask':
            from skimage.filters import unsharp_mask
            # unsharp_mask params: image, radius, amount
            # radius: Gaussian blur radius (controls scale of details to sharpen)
            # amount: Strength of sharpening (similar to our strength)
            # Let's use a default radius and map our strength to amount.
            # skimage's unsharp_mask works well on float images.
            # The output range should be similar to input if preserve_range=True (default)
            radius = 1.0 # Default radius, can be made a parameter
            sharpened_image = unsharp_mask(img_float, radius=radius, amount=strength, preserve_range=True)
        elif method.lower() == 'convolution':
            from scipy.ndimage import convolve
            # Basic sharpening kernel
            kernel = np.array([[0, -1, 0],
                               [-1,  5, -1],
                               [0, -1, 0]], dtype=np.float32) 
            # Strength for convolution can be tricky. This kernel has a built-in strength.
            # To modulate strength, one might blend the sharpened and original image.
            # sharpened = original * (1-s) + convolved * s
            # For now, just apply kernel. A strength < 1 could mean less of this effect.
            if strength != 1.0:
                 print("Warning: Convolution sharpening strength parameter is not fully implemented for custom values yet, using default kernel effect.")

            sharpened_image = convolve(img_float, kernel)
        else:
            print(f"Error: Unknown sharpening method '{method}'.")
            return image
        
        return sharpened_image

    def reduce_noise(self, image: np.ndarray, method: str = 'nl_means', strength: float = 0.1) -> np.ndarray | None:
        """
        Reduces noise in an image.

        Args:
            image: NumPy array of the image.
            method: 'gaussian', 'median_filter', 'nl_means' (non-local means), 'bilateral'.
            strength: Denoising strength. Interpretation depends on method.
                      For Gaussian: sigma.
                      For nl_means: h (patch_distance cutoff).
                      For median_filter: footprint size (approx).
                      For bilateral: sigma_spatial.

        Returns:
            Denoised NumPy array.
        """
        if image is None:
            return None

        img_float = image.astype(np.float32)
        
        # skimage.restoration functions often expect images in range [0,1] or specific dtypes.
        # We need to handle data range carefully.
        # For now, let's assume img_float can be used directly if functions support it,
        # or normalize if necessary.

        if method.lower() == 'gaussian':
            from scipy.ndimage import gaussian_filter
            # strength here is sigma
            denoised_image = gaussian_filter(img_float, sigma=strength)
        elif method.lower() == 'median_filter':
            from scipy.ndimage import median_filter
            # strength here is footprint size (e.g., 3 for 3x3)
            size = int(max(1, strength)) # Ensure it's an int >= 1
            denoised_image = median_filter(img_float, size=size)
        elif method.lower() == 'nl_means':
            from skimage.restoration import denoise_nl_means, estimate_sigma
            # nl_means works best on images with values in [0,1] or specific integer types.
            # We may need to normalize/scale.
            # Parameter 'h' is related to strength. It's often set relative to noise sigma.
            
            # Estimate noise standard deviation
            sigma_est = estimate_sigma(img_float, average_sigmas=True) # Use average_sigmas for multichannel=False
            if sigma_est == 0: sigma_est = 1e-6 # Avoid division by zero if image is flat

            # h parameter often set as sigma_est * some_factor.
            # Let's make 'strength' this factor. Typical 'strength' might be 0.5 to 1.5
            h_param = strength * sigma_est 
            
            # preserve_range=True is crucial for float images not in [0,1]
            denoised_image = denoise_nl_means(img_float, h=h_param,
                                              patch_size=5, patch_distance=6, # Common defaults
                                              preserve_range=True, channel_axis=None) # channel_axis for grayscale
        elif method.lower() == 'bilateral':
            from skimage.restoration import denoise_bilateral
            # Parameters: image, sigma_color, sigma_spatial
            # Let strength be sigma_spatial. sigma_color needs to be estimated or set.
            # Estimate sigma_color based on data range or noise.
            sigma_color_est = 0.05 * (np.max(img_float) - np.min(img_float)) # Heuristic
            if sigma_color_est == 0: sigma_color_est = 0.01

            denoised_image = denoise_bilateral(img_float, sigma_color=sigma_color_est,
                                               sigma_spatial=strength,
                                               preserve_range=True, channel_axis=None)
        else:
            print(f"Error: Unknown noise reduction method '{method}'.")
            return image
            
        return denoised_image


def run_stacking_tests(processor):
    print("--- Testing ImageProcessor Stacking ---")

    # Create dummy images (all float32)
    img1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32) * 10
    img2 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.float32) * 1.1
    img3 = np.array([[100, 200, 300], [400, 500, 600], [700, 800, 999]], dtype=np.float32) # img3 has an outlier (999)
    
    images_regular = [img1, img2]
    images_with_outlier = [img1, img2, img3]
    images_for_kappa_test = [
        np.array([[10,20,1000],[30,40,50],[60,70,80]], dtype=np.float32), # Outlier at (0,2)
        np.array([[12,22,25],[32,42,52],[62,72,82]], dtype=np.float32),
        np.array([[11,21,26],[31,41,51],[61,71,81]], dtype=np.float32),
        np.array([[13,23,24],[33,43,53],[63,73,83]], dtype=np.float32),
    ]

    # Test alignment (currently a passthrough)
    print("\n--- Testing Alignment (passthrough) ---")
    aligned_imgs = processor.align_images(images_regular, method='none')
    assert np.array_equal(aligned_imgs[0], img1), "Alignment 'none' failed"
    aligned_imgs_fake_method = processor.align_images(images_regular, method='fake')
    assert np.array_equal(aligned_imgs_fake_method[0], img1), "Alignment 'fake' method failed"

    # Test stacking with different methods
    print("\n--- Testing Average Stacking ---")
    avg_stack = processor.stack_images(images_regular, method='average')
    expected_avg = (img1 + img2) / 2
    assert avg_stack is not None, "Average stack resulted in None"
    assert np.allclose(avg_stack, expected_avg), f"Average stack incorrect. Expected:\n{expected_avg}\nGot:\n{avg_stack}"
    print("Average Stack:\n", avg_stack)

    print("\n--- Testing Median Stacking ---")
    median_stack = processor.stack_images(images_with_outlier, method='median')
    expected_median_val_00 = np.median([img1[0,0], img2[0,0], img3[0,0]])
    expected_median_val_22 = np.median([img1[2,2], img2[2,2], img3[2,2]])
    assert median_stack is not None, "Median stack resulted in None"
    assert np.isclose(median_stack[0,0], expected_median_val_00), f"Median stack [0,0] incorrect. Expected {expected_median_val_00}, Got {median_stack[0,0]}"
    assert np.isclose(median_stack[2,2], expected_median_val_22), f"Median stack [2,2] incorrect. Expected {expected_median_val_22}, Got {median_stack[2,2]}"
    print("Median Stack (from 3 images with one outlier):\n", median_stack)


    print("\n--- Testing Kappa-Sigma Clipping Stacking ---")
    # For images_for_kappa_test, pixel (0,2) has values [1000, 25, 26, 24]
    # sigmaclip([1000,25,26,24], 2.0, 2.0) should clip 1000. Mean of [25,26,24] is 25.
    kappa_stack = processor.stack_images(images_for_kappa_test, method='kappa_sigma_clip', kappa=2.0)
    assert kappa_stack is not None, "Kappa-sigma stack resulted in None"
    
    # Check pixel (0,2)
    expected_val_02_kappa = np.mean([25,26,24]) 
    assert np.isclose(kappa_stack[0,2], expected_val_02_kappa), \
        f"Kappa-sigma stack [0,2] incorrect. Expected {expected_val_02_kappa}, Got {kappa_stack[0,2]}"

    # Check a regular pixel, e.g. (0,0): [10,12,11,13]. Mean is 11.5. sigma_clip should not clip these.
    vals_00 = np.array([img[0,0] for img in images_for_kappa_test])
    # Use astropy.stats.sigma_clip for the test assertion as well
    clipped_mask_array_00 = sigma_clip(vals_00, sigma=2.0, maxiters=5, cenfunc='median') # Using same params as in main func
    valid_vals_00 = clipped_mask_array_00.data[~clipped_mask_array_00.mask]
    expected_val_00_kappa = np.mean(valid_vals_00) if valid_vals_00.size > 0 else np.mean(vals_00)

    assert np.isclose(kappa_stack[0,0], expected_val_00_kappa), \
        f"Kappa-sigma stack [0,0] incorrect. Expected {expected_val_00_kappa}, Got {kappa_stack[0,0]}"
    print("Kappa-Sigma Stack:\n", kappa_stack)

    print("\n--- Testing Edge Cases ---")
    # No images
    no_img_stack = processor.stack_images([], method='average')
    assert no_img_stack is None, "Stacking no images should return None"

    # Images with different dimensions
    img_diff_dim = np.ones((2,2), dtype=np.float32)
    diff_dim_stack = processor.stack_images([img1, img_diff_dim], method='average')
    assert diff_dim_stack is None, "Stacking images with different dimensions should return None"

    # Unknown method
    unknown_method_stack = processor.stack_images(images_regular, method='unknown')
    assert unknown_method_stack is None, "Stacking with unknown method should return None"
    
    # Kappa-sigma with all values clipped for a pixel (test fallback)
    # Create a scenario where all values for one pixel might be clipped if kappa is very small
    # or if only 1-2 values exist for a pixel after some other processing (not directly testable here easily)
    # The current fallback is to np.mean of original values for that pixel.
    # For the test with images_for_kappa_test, if kappa was extremely low, e.g., 0.1, it might clip more.
    # For now, the existing test covers a basic outlier rejection.
    # A specific test for the "all clipped" fallback:
    one_pixel_imgs = [np.array([[10]]), np.array([[1000]]), np.array([[1001]])] # one pixel images
    one_pixel_stack_kappa = processor.stack_images(one_pixel_imgs, method='kappa_sigma_clip', kappa=0.5)
    # With kappa=0.5, [10, 1000, 1001], mean ~670, std ~500. 0.5*std ~250.
    # 10 is outside 670 +/- 250. 1000, 1001 might be kept or one clipped.
    # With astropy.stats.sigma_clip(cenfunc='median', sigma=0.5), the values [10, 1000, 1001]
    # will likely result in all values being clipped eventually, triggering the fallback.
    # Iteration 1 (median=1000): 10 is clipped. Remaining: [1000, 1001].
    # Iteration 2 (median=1000.5, std=0.5): 1000 and 1001 are clipped against their own tight distribution.
    # Fallback is np.mean([10, 1000, 1001])
    expected_one_pixel_kappa = np.mean([10., 1000., 1001.]) 
    assert np.isclose(one_pixel_stack_kappa[0,0], expected_one_pixel_kappa), \
        f"Kappa-sigma for one_pixel_imgs failed. Expected {expected_one_pixel_kappa}, Got {one_pixel_stack_kappa[0,0]}"
    print(f"Kappa-sigma for one_pixel_imgs (kappa=0.5, with fallback): {one_pixel_stack_kappa[0,0]}")


    print("\nAll ImageProcessor stacking tests seem to pass.")


def run_editing_tests(processor):
    print("\n--- Testing ImageProcessor Editing ---")
    
    # Test image (float, mimics normalized 0-1 range for easier testing of levels)
    test_img_norm = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, 0.1]], dtype=np.float32)
    # Test image (float, mimics general FITS data range)
    test_img_astro = np.array([[100, 2000, 5000], [10000, 15000, 60000]], dtype=np.float32)

    # Test Brightness/Contrast
    print("\n--- Brightness/Contrast ---")
    bc_adjusted = processor.adjust_brightness_contrast(test_img_astro.copy(), brightness=100.0, contrast=1.5)
    assert bc_adjusted is not None, "Brightness/contrast adjustment failed"
    # Expected: ((val - mean_val) * 1.5 + mean_val) + 100
    mean_astro = np.mean(test_img_astro)
    expected_bc_val_00 = (test_img_astro[0,0] - mean_astro) * 1.5 + mean_astro + 100.0
    assert np.isclose(bc_adjusted[0,0], expected_bc_val_00), \
        f"B/C adj [0,0] expected {expected_bc_val_00}, got {bc_adjusted[0,0]}"
    print(f"Original [0,0]: {test_img_astro[0,0]}, B/C adjusted [0,0]: {bc_adjusted[0,0]}")

    # Test Levels
    print("\n--- Levels ---")
    # Using test_img_norm (0-1 range)
    # Map input 0.25 to black, 0.75 to white, linear gamma (1.0)
    levels_adjusted = processor.adjust_levels(test_img_norm.copy(), black_point=0.25, white_point=0.75, mid_point=1.0)
    assert levels_adjusted is not None, "Levels adjustment failed"
    # Expected:
    # 0.0  -> (0.0 - 0.25) / (0.75-0.25) = -0.25 / 0.5 = -0.5 -> clip to 0
    # 0.25 -> (0.25 - 0.25) / 0.5 = 0 -> 0
    # 0.5  -> (0.5 - 0.25) / 0.5 = 0.25 / 0.5 = 0.5
    # 0.75 -> (0.75 - 0.25) / 0.5 = 0.5 / 0.5 = 1.0
    # 1.0  -> (1.0 - 0.25) / 0.5 = 0.75 / 0.5 = 1.5 -> clip to 1.0
    # 0.1  -> (0.1 - 0.25) / 0.5 = -0.15 / 0.5 = -0.3 -> clip to 0
    expected_levels = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 0.0]], dtype=np.float32)
    assert np.allclose(levels_adjusted, expected_levels), \
        f"Levels adjustment incorrect. Expected:\n{expected_levels}\nGot:\n{levels_adjusted}"
    print("Levels Adjusted (0.25->0, 0.75->1):\n", levels_adjusted)

    # Test Levels with gamma
    levels_gamma_adj = processor.adjust_levels(test_img_norm.copy(), black_point=0.0, white_point=1.0, mid_point=2.0) # brighten midtones
    # (0.5 - 0) / (1-0) = 0.5. Then 0.5 ^ (1/2.0) = sqrt(0.5) approx 0.707
    assert np.isclose(levels_gamma_adj[0,2], np.sqrt(0.5)), "Levels with gamma failed for mid-tone"
    print("Levels Adjusted (gamma=2.0, mid-point 0.5 -> ~0.707):\n", levels_gamma_adj)

    # Test Levels with astro data (min/max could be used for black/white points)
    min_astro, max_astro = np.min(test_img_astro), np.max(test_img_astro)
    levels_astro_adj = processor.adjust_levels(test_img_astro.copy(), black_point=min_astro, white_point=max_astro)
    # Should normalize astro data to 0-1 range
    assert np.isclose(np.min(levels_astro_adj), 0.0) and np.isclose(np.max(levels_astro_adj), 1.0), \
        "Levels on astro data did not normalize to 0-1"
    print(f"Levels on astro data (min={np.min(levels_astro_adj):.2f}, max={np.max(levels_astro_adj):.2f})")


    print("\nAll ImageProcessor editing tests seem to pass so far.")

    # Placeholder for Color Balance test - tricky for single channel (grayscale)
    # For now, just ensure it runs without error if called.
    # print("\n--- Color Balance (Placeholder) ---")
    # color_balanced_img = processor.adjust_color_balance(test_img_astro.copy(), {'r':1.0, 'g':1.0, 'b':1.0})
    # assert color_balanced_img is not None

    # Test Sharpen
    print("\n--- Sharpen ---")
    sharpened_unsharp = processor.sharpen_image(test_img_astro.copy(), strength=1.0, method='unsharp_mask')
    assert sharpened_unsharp is not None
    # Check if different from original (unsharp mask should change it)
    assert not np.allclose(sharpened_unsharp, test_img_astro), "Unsharp mask did not change the image"
    print(f"Unsharp mask applied. Original mean: {np.mean(test_img_astro):.2f}, Sharpened mean: {np.mean(sharpened_unsharp):.2f}")

    sharpened_conv = processor.sharpen_image(test_img_astro.copy(), strength=1.0, method='convolution')
    assert sharpened_conv is not None
    assert not np.allclose(sharpened_conv, test_img_astro), "Convolution sharpen did not change the image"
    print(f"Convolution sharpen applied. Original mean: {np.mean(test_img_astro):.2f}, Sharpened mean: {np.mean(sharpened_conv):.2f}")

    # Test Noise Reduction
    print("\n--- Noise Reduction ---")
    # Create a noisy image (original + Gaussian noise)
    noisy_img = test_img_norm + np.random.normal(0, 0.1, test_img_norm.shape).astype(np.float32)
    
    denoised_gauss = processor.reduce_noise(noisy_img.copy(), method='gaussian', strength=0.5) # sigma=0.5
    assert denoised_gauss is not None
    # Denoised should have smaller std dev or be closer to original than noisy one
    assert np.std(denoised_gauss) < np.std(noisy_img) or np.sum((denoised_gauss - test_img_norm)**2) < np.sum((noisy_img - test_img_norm)**2)
    print(f"Gaussian denoise applied. Noisy std: {np.std(noisy_img):.3f}, Denoised std: {np.std(denoised_gauss):.3f}")

    # nl_means might be slow on CPU for large images, but test on small one
    denoised_nl = processor.reduce_noise(noisy_img.copy(), method='nl_means', strength=0.8) # strength here is factor for h=factor*sigma_est
    assert denoised_nl is not None
    assert np.std(denoised_nl) < np.std(noisy_img) or np.sum((denoised_nl - test_img_norm)**2) < np.sum((noisy_img - test_img_norm)**2)
    print(f"NL Means denoise applied. Noisy std: {np.std(noisy_img):.3f}, Denoised std: {np.std(denoised_nl):.3f}")

    # Test Curves
    print("\n--- Curves ---")
    # Image normalized 0-1
    # Simple S-curve: (0,0), (0.25, 0.15), (0.75, 0.85), (1,1)
    # Input 0.5 should be remapped to output 0.5 (linear interpolation between 0.15 and 0.85)
    # (0.5 - 0.25) / (0.75 - 0.25) = 0.25 / 0.5 = 0.5
    # Output = 0.15 + 0.5 * (0.85 - 0.15) = 0.15 + 0.5 * 0.7 = 0.15 + 0.35 = 0.5
    s_curve_points = [(0.0, 0.0), (0.25, 0.15), (0.75, 0.85), (1.0, 1.0)]
    curves_adjusted = processor.adjust_curves(test_img_norm.copy(), points=s_curve_points)
    assert curves_adjusted is not None, "Curves adjustment failed"
    
    # Check specific values based on interpolation
    # test_img_norm[0,2] is 0.5. With the S-curve, it should remain 0.5.
    assert np.isclose(curves_adjusted[0,2], 0.5), \
        f"S-Curve did not map 0.5 to 0.5. Got {curves_adjusted[0,2]}"
    # test_img_norm[0,1] is 0.25. Should map to 0.15
    assert np.isclose(curves_adjusted[0,1], 0.15), \
        f"S-Curve did not map 0.25 to 0.15. Got {curves_adjusted[0,1]}"
    # test_img_norm[1,0] is 0.75. Should map to 0.85
    assert np.isclose(curves_adjusted[1,0], 0.85), \
        f"S-Curve did not map 0.75 to 0.85. Got {curves_adjusted[1,0]}"
    print("S-Curve applied. Mid-point 0.5 -> 0.5 (as per this S-curve)")
    print(curves_adjusted)


    print("\nAll ImageProcessor editing tests completed.")


if __name__ == '__main__':
    print("Starting __main__ execution...")
    processor = ImageProcessor()
    
    print("Running stacking tests...")
    run_stacking_tests(processor)
    print("Stacking tests completed.")
    
    # Temporarily comment out editing tests to isolate timeout
    # print("Running editing tests...")
    # run_editing_tests(processor)
    # print("Editing tests completed.")
    
    print("__main__ execution finished.")
