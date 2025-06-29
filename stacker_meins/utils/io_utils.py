import numpy as np
from astropy.io import fits
from PIL import Image, UnidentifiedImageError

def load_image(filepath: str) -> tuple[np.ndarray | None, dict | None]:
    """
    Loads an image from the given filepath.
    Supports FITS, TIFF, and JPG formats.

    Args:
        filepath: Path to the image file.

    Returns:
        A tuple containing:
            - A NumPy array of the image data if successful, None otherwise.
            - A dictionary of metadata (e.g., FITS header) if available, None otherwise.
    """
    filepath_lower = filepath.lower()
    metadata = {}

    try:
        if filepath_lower.endswith(('.fits', '.fit', '.fts')):
            try:
                with fits.open(filepath, memmap=False) as hdul: # memmap=False for easier debugging and small files
                    # Try to find the first HDU with image data
                    image_hdu = None
                    for hdu_index, hdu in enumerate(hdul):
                        if hdu.is_image and hdu.data is not None:
                            # Check for common non-image HDUs by name if possible
                            if hdu.name and hdu.name.upper() in ['MASK', 'DQ', 'ERROR', 'WEIGHT']:
                                print(f"Info: Skipping HDU '{hdu.name}' as it might be a mask/quality plane.")
                                continue
                            image_hdu = hdu
                            print(f"Info: Selected HDU {hdu_index} (Name: {hdu.name if hdu.name else 'N/A'}) for image data from {filepath}.")
                            break
                    
                    if image_hdu is None:
                        # If no image HDU found by typical means, try the primary if it has data
                        if hdul[0].data is not None and hdul[0].is_image:
                            print(f"Warning: No explicit image HDU found. Using Primary HDU from {filepath}.")
                            image_hdu = hdul[0]
                        else:
                            print(f"Error: No suitable image data found in FITS file: {filepath}")
                            # Optionally, list HDUs for debugging:
                            # hdul.info()
                            return None, None

                    # Ensure data is loaded as a NumPy array and convert to float32
                    # Astropy typically returns data in its native type (e.g. >i2, >f4)
                    # Convert to a consistent float type for processing.
                    data = image_hdu.data 
                    if data is None: # Should have been caught by hdu.data is not None, but double check
                        print(f"Error: Selected HDU in {filepath} has None for data.")
                        return None, None

                    # Handle BZERO/BSCALE for correct data representation (especially for unsigned integers)
                    # The .section attribute applies scaling by default, but .data does not always.
                    # Forcing it here for consistency.
                    # However, astropy's fits.CompImageHDU and fits.ImageHDU. sección access often handle this.
                    # Direct .data access might need manual scaling if not using .section or convenience functions.
                    # Let's assume astropy handles scaling if data is accessed, but be mindful.
                    # If BITPIX > 0 (integers), BZERO and BSCALE might apply.
                    # Astropy's default behavior with .data on an ImageHDU usually gives scaled data.
                    # Forcing to float32 should be safe.
                    data = np.array(data, dtype=np.float32)
                    
                    metadata = dict(image_hdu.header)
                    
                    # If image is multi-dimensional (e.g., data cube), take the first 2D slice.
                    # This is a common simplification for basic display/processing.
                    # More sophisticated handling (e.g., user selection of slice) would be a UI feature.
                    if data.ndim > 2:
                        original_shape = data.shape
                        if data.shape[0] == 1 and data.ndim == 3: # (1, height, width)
                            data = data[0, :, :]
                        elif data.shape[-1] in [1,3,4] and data.ndim == 3 and original_shape[0] > 10 and original_shape[1] > 10 : # (height, width, channels) like some FITS from cameras
                             if data.shape[-1] == 3 or data.shape[-1] == 4: # Color FITS, convert to grayscale
                                 print(f"Info: FITS file {filepath} appears to be a color image {original_shape}. Converting to grayscale (average of channels).")
                                 data = np.mean(data, axis=-1).astype(np.float32)
                             elif data.shape[-1] == 1:
                                 data = data[...,0] # (height, width, 1) -> (height, width)
                        else: # True cube (e.g. spectral cube)
                            data = data[0, :, :] # Take the first slice
                        print(f"Warning: FITS file {filepath} has {original_shape} dimensions. Extracted a 2D slice of shape {data.shape}.")

                    if data.ndim != 2:
                        print(f"Error: Could not extract a 2D image from FITS file: {filepath}. Final shape: {data.shape}")
                        return None, None
                        
                    return data, metadata
            except Exception as e:
                print(f"Error loading FITS file {filepath}: {e}")
                import traceback
                traceback.print_exc()
                return None, None

        elif filepath_lower.endswith(('.tiff', '.tif', '.jpg', '.jpeg', '.png')): # Added PNG
            try:
                with Image.open(filepath) as img:
                    # Preserve metadata before any conversion
                    if hasattr(img, 'tag_v2'):
                        metadata.update(img.tag_v2)
                    if hasattr(img, 'info'): # General metadata dictionary
                        metadata.update(img.info)
                    if hasattr(img, '_getexif'): # EXIF for JPEGs
                        exif_data = img._getexif()
                        if exif_data:
                            # Decode EXIF tags for readability
                            from PIL import ExifTags
                            decoded_exif = {ExifTags.TAGS.get(key, key): value for key, value in exif_data.items()}
                            metadata['EXIF'] = decoded_exif

                    # Convert to a common mode for processing.
                    # For astro, we often want grayscale. If it's color, convert.
                    # 'F' mode (32-bit float) is ideal for preserving precision.
                    # 'L' mode (luminance/grayscale) is good for 8-bit.
                    # 'I' mode (32-bit signed integer)
                    
                    print(f"Info: Loading {filepath} with Pillow. Original mode: {img.mode}, Original dtype guess: {np.array(img).dtype if img.mode not in ['P', 'PA'] else 'Palette'}")

                    if img.mode == 'P' or img.mode == 'PA': # Palette modes
                        print(f"Info: Converting palettized image {filepath} (mode {img.mode}) to RGB then to float32 grayscale.")
                        img = img.convert('RGB')


                    if img.mode.startswith('I;16') or img.mode == 'I': # 16-bit or 32-bit integer
                        data = np.array(img, dtype=np.float32)
                    elif img.mode == 'F': # 32-bit float
                        data = np.array(img, dtype=np.float32)
                    elif img.mode in ['RGB', 'RGBA', 'CMYK', 'YCbCr']:
                        print(f"Info: Converting color image {filepath} (mode {img.mode}) to grayscale float32.")
                        # Convert to 'L' (luminance) first, then to float array
                        img_gray = img.convert('L')
                        data = np.array(img_gray, dtype=np.float32)
                    elif img.mode == 'L': # 8-bit grayscale
                        data = np.array(img, dtype=np.float32) # Promote to float32
                    else:
                        print(f"Warning: Image {filepath} has unhandled mode {img.mode}. Attempting conversion to L (grayscale) then float32.")
                        try:
                            img_gray = img.convert('L')
                            data = np.array(img_gray, dtype=np.float32)
                        except Exception as conv_e:
                            print(f"Error: Could not convert image {filepath} from mode {img.mode} to L: {conv_e}")
                            return None, None
                    
                    return data, metadata
            except UnidentifiedImageError:
                print(f"Error: Cannot identify image file {filepath}. Pillow does not recognize it or it's corrupted.")
                return None, None
            except Exception as e:
                print(f"Error loading image file {filepath} with Pillow: {e}")
                import traceback
                traceback.print_exc()
                return None, None
        else:
            print(f"Error: Unsupported file format for {filepath}. Supported: FITS, TIFF, JPG/JPEG, PNG.")
            return None, None
    except Exception as e:
        print(f"An unexpected error occurred while trying to load {filepath}: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def save_image(filepath: str, image_data: np.ndarray, format_str: str = None, metadata: dict = None, overwrite: bool = True) -> bool:
    """
    Saves a NumPy array as an image file.
    Supports FITS, TIFF, JPG/JPEG, and PNG formats.

    Args:
        filepath: Path to save the image file.
        image_data: NumPy array of the image data.
        format_str: The desired output format (e.g., 'FITS', 'TIFF', 'JPG', 'PNG').
                    If None, it's inferred from the filepath extension.
        metadata: Optional FITS header to save with FITS files, or dict for some TIFF tags.
        overwrite: Whether to overwrite the file if it exists.

    Returns:
        True if successful, False otherwise.
    """
    if image_data is None:
        print("Error: No image data provided to save.")
        return False

    filepath_lower = filepath.lower()
    output_format = format_str.lower() if format_str else None

    if output_format is None:
        if filepath_lower.endswith(('.fits', '.fit', '.fts')):
            output_format = 'fits'
        elif filepath_lower.endswith(('.tiff', '.tif')):
            output_format = 'tiff'
        elif filepath_lower.endswith(('.jpg', '.jpeg')):
            output_format = 'jpeg'
        elif filepath_lower.endswith('.png'):
            output_format = 'png'
        else:
            print(f"Error: Cannot infer output format from filepath {filepath} and no format specified.")
            print("Please specify format as 'FITS', 'TIFF', 'JPG', or 'PNG'.")
            return False

    try:
        if output_format == 'fits':
            hdr = fits.Header(metadata) if metadata else fits.Header()
            # Add basic FITS keywords if not present (Astropy usually handles this, but good practice)
            if 'SIMPLE' not in hdr: hdr['SIMPLE'] = True
            
            # Initialize image_data_for_hdu and then create HDU once
            image_data_for_hdu = image_data # Default, will be changed if type conversion needed

            if image_data.dtype == np.uint8:
                final_bzero = 32768
                final_bscale = 1
                image_data_for_hdu = image_data.astype(np.int32) - final_bzero
                image_data_for_hdu = image_data_for_hdu.astype(np.int16)
                # hdr will be used when creating HDU, then BSCALE/BZERO explicitly set on hdu.header
            elif image_data.dtype == np.uint16:
                final_bzero = 2**15
                final_bscale = 1
                image_data_for_hdu = image_data.astype(np.int32) - final_bzero
                # image_data_for_hdu is now int32
                # hdr will be used, then BSCALE/BZERO/BITPIX explicitly set on hdu.header
            elif image_data.dtype not in [np.float32, np.float64, np.int16, np.int32]:
                print(f"Warning: Data type {image_data.dtype} for FITS. Converting to float32.")
                image_data_for_hdu = image_data.astype(np.float32)

            # Create HDU with potentially transformed data and initial header (from metadata or empty)
            hdu = fits.PrimaryHDU(data=image_data_for_hdu, header=hdr)

            # Explicitly set BITPIX, BSCALE, BZERO if needed, overriding defaults if necessary
            if image_data.dtype == np.uint8:
                hdu.header['BITPIX'] = 16
                hdu.header['BSCALE'] = 1
                hdu.header['BZERO'] = 32768
            elif image_data.dtype == np.uint16:
                hdu.header['BITPIX'] = 32 # Storing uint16 as int32 with BZERO
                hdu.header['BSCALE'] = 1
                hdu.header['BZERO'] = 2**15
            elif image_data.dtype == np.float32:
                hdu.header['BITPIX'] = -32
            elif image_data.dtype == np.float64:
                hdu.header['BITPIX'] = -64
            elif image_data.dtype == np.int16:
                hdu.header['BITPIX'] = 16
            elif image_data.dtype == np.int32:
                hdu.header['BITPIX'] = 32
            # If image_data_for_hdu was converted (e.g. from an unknown type to float32),
            # ensure BITPIX matches image_data_for_hdu.dtype
            elif image_data_for_hdu.dtype == np.float32:
                 hdu.header['BITPIX'] = -32


            # Ensure NAXIS keywords are correct for the data being written
            hdu.header['NAXIS'] = image_data_for_hdu.ndim
            for i in range(image_data_for_hdu.ndim):
                ax_num = i + 1
                ax_key = f'NAXIS{ax_num}'
                expected_ax_len = image_data_for_hdu.shape[image_data_for_hdu.ndim - 1 - i]
                hdu.header[ax_key] = expected_ax_len
            
            # Remove problematic keywords from metadata if they ended up in hdr
            if metadata: 
                keywords_to_remove = ['XTENSION', 'EXTNAME', 'EXTVER', 
                                      'PCOUNT', 'GCOUNT', 'THEAP', 'TFIELDS'] 
                for kw in keywords_to_remove:
                    if kw in hdu.header: 
                        del hdu.header[kw]
            
            hdul = fits.HDUList([hdu])
            hdul.writeto(filepath, overwrite=overwrite, checksum=True)
            print(f"Image saved as FITS: {filepath}")
            return True

        elif output_format in ['tiff', 'jpeg', 'jpg', 'png']:
            # Prepare data for Pillow: usually expects uint8, uint16, or float32 (for some TIFFs)
            # A simple min-max normalization is often insufficient for astro images.
            # For now, we'll do a basic scaling if the input is float.
            # A more sophisticated stretching (log, sqrt, percentiles) should be handled
            # by the processing part of the application before saving to LDR formats.

            current_dtype = image_data.dtype
            img_to_save = image_data.copy() # Work on a copy

            if output_format in ['jpeg', 'jpg', 'png']: # These typically expect 0-255 (uint8) or 0-65535 (uint16 for PNG)
                if current_dtype == np.float32 or current_dtype == np.float64:
                    print(f"Info: Scaling float data for {output_format.upper()} output. This is a basic linear stretch.")
                    min_val, max_val = np.percentile(img_to_save, [0.1, 99.9]) # Robust min/max
                    if max_val <= min_val: max_val = min_val + 1e-6 # Avoid division by zero
                    img_to_save = (img_to_save - min_val) / (max_val - min_val)
                    img_to_save = np.clip(img_to_save, 0, 1)
                    if output_format == 'png' and metadata and metadata.get('SAVE_AS_16BIT_PNG', False): # Hypothetical metadata flag
                        img_to_save = (img_to_save * 65535).astype(np.uint16)
                        pil_mode = 'I;16'
                    else:
                        img_to_save = (img_to_save * 255).astype(np.uint8)
                        pil_mode = 'L' # Grayscale
                elif current_dtype == np.uint16:
                    if output_format == 'png':
                        pil_mode = 'I;16'
                    else: # JPG
                        img_to_save = (img_to_save / 256).astype(np.uint8) # Scale 16-bit to 8-bit
                        pil_mode = 'L'
                elif current_dtype == np.uint8:
                    pil_mode = 'L'
                else: # Other integer types
                    print(f"Warning: Converting data from {current_dtype} to uint8 for {output_format.upper()}.")
                    img_to_save = img_to_save.astype(np.float32) # intermediate step
                    min_val, max_val = np.min(img_to_save), np.max(img_to_save)
                    if max_val <= min_val: max_val = min_val + 1e-6
                    img_to_save = (img_to_save - min_val) / (max_val - min_val)
                    img_to_save = (img_to_save * 255).astype(np.uint8)
                    pil_mode = 'L'
            
            elif output_format == 'tiff': # TIFF can handle more data types
                if current_dtype == np.float32:
                    pil_mode = 'F'
                elif current_dtype == np.float64:
                    img_to_save = img_to_save.astype(np.float32) # Convert to float32 for TIFF
                    pil_mode = 'F'
                elif current_dtype == np.uint16 or current_dtype == np.int16:
                    pil_mode = 'I;16' if current_dtype == np.uint16 else 'I;16S' # Pillow might map I;16 to unsigned
                elif current_dtype == np.uint8:
                    pil_mode = 'L'
                elif current_dtype == np.int32:
                    pil_mode = 'I'
                else:
                    print(f"Warning: Data type {current_dtype} for TIFF. Converting to float32.")
                    img_to_save = img_to_save.astype(np.float32)
                    pil_mode = 'F'
            
            pil_image = Image.fromarray(img_to_save, mode=pil_mode)

            # Handle metadata for TIFF (simplified)
            tiffinfo_dict = {}
            if metadata and output_format == 'tiff':
                # Pillow's TIFF metadata handling is via libtiff tags.
                # Example: ImageDescription (tag 270)
                if "COMMENT" in metadata:
                    tiffinfo_dict[270] = str(metadata["COMMENT"])
                if "ARTIST" in metadata:
                    tiffinfo_dict[315] = str(metadata["ARTIST"])
                # More complex metadata would require mapping to specific TIFF tag numbers/types.
                # For EXIF-like data, Pillow can write exif using img.save(..., exif=exif_bytes)
                # but constructing exif_bytes from a dict is non-trivial.

            save_kwargs = {}
            if overwrite is False and os.path.exists(filepath):
                 print(f"Error: File {filepath} exists and overwrite is False.")
                 return False

            if output_format == 'jpeg' or output_format == 'jpg':
                save_kwargs['quality'] = 95
                pil_image.save(filepath, **save_kwargs)
            elif output_format == 'png':
                save_kwargs['compress_level'] = 6 # 0-9, default 6
                if pil_mode == 'I;16': # Ensure 16-bit PNGs are saved correctly
                    save_kwargs['bits'] = 16 # This might not be directly supported by all Pillow versions/backends
                pil_image.save(filepath, **save_kwargs)
            elif output_format == 'tiff':
                if tiffinfo_dict:
                    save_kwargs['tiffinfo'] = tiffinfo_dict
                pil_image.save(filepath, **save_kwargs)
            
            print(f"Image saved as {output_format.upper()}: {filepath} (mode: {pil_mode})")
            return True
            
        else:
            print(f"Error: Unsupported output format '{output_format}'. Supported: FITS, TIFF, JPG/JPEG, PNG.")
            return False
    except Exception as e:
        print(f"Error saving image to {filepath} (format: {output_format}): {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    import os
    # Create dummy files for testing
    # Ensure a directory for test files exists
    test_dir = "test_io_files"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    def get_test_filepath(name):
        return os.path.join(test_dir, name)

    # Test FITS
    print("--- Testing FITS ---")
    dummy_fits_data = np.arange(100, dtype=np.float32).reshape(10, 10) + 1000
    hdr = fits.Header()
    hdr['TELESCOP'] = 'MyScope'
    hdr['EXPTIME'] = 120.5
    hdr['OBJECT'] = 'Dummy Object'
    fits_path = get_test_filepath('dummy_test.fits')
    fits.writeto(fits_path, dummy_fits_data, header=hdr, overwrite=True)
    
    loaded_data, metadata = load_image(fits_path)
    if loaded_data is not None:
        print(f"Loaded FITS. Shape: {loaded_data.shape}, Data type: {loaded_data.dtype}, Min: {np.min(loaded_data)}, Max: {np.max(loaded_data)}")
        print(f"Metadata (TELESCOP): {metadata.get('TELESCOP')}")
        assert loaded_data.shape == (10,10)
        assert metadata.get('TELESCOP') == 'MyScope'
        assert np.allclose(loaded_data, dummy_fits_data)
        save_fits_path = get_test_filepath('dummy_test_save.fits')
        save_image(save_fits_path, loaded_data, metadata=metadata)
        # Verify saved FITS
        with fits.open(save_fits_path) as hdul_saved:
            assert hdul_saved[0].header['OBJECT'] == 'Dummy Object'
            assert np.allclose(hdul_saved[0].data, dummy_fits_data)


    # Test FITS cube loading
    dummy_fits_cube = np.arange(2*3*4, dtype=np.int16).reshape(2,3,4)
    fits_cube_path = get_test_filepath('dummy_cube.fits')
    fits.writeto(fits_cube_path, dummy_fits_cube, overwrite=True)
    loaded_cube_slice, _ = load_image(fits_cube_path)
    if loaded_cube_slice is not None:
        print(f"Loaded FITS cube slice. Shape: {loaded_cube_slice.shape}")
        assert loaded_cube_slice.shape == (3,4)
        assert np.allclose(loaded_cube_slice, dummy_fits_cube[0,:,:])

    # Test FITS with uint8 data (should be scaled with BZERO)
    uint8_fits_data = np.arange(256, dtype=np.uint8).reshape(16,16)
    uint8_fits_path = get_test_filepath('dummy_uint8.fits')
    # astropy handles BZERO/BSCALE automatically when creating HDU from uint8
    primary_hdu_uint8 = fits.PrimaryHDU(data=uint8_fits_data)
    primary_hdu_uint8.writeto(uint8_fits_path, overwrite=True)
    
    loaded_uint8_data, meta_uint8 = load_image(uint8_fits_path)
    if loaded_uint8_data is not None:
        print(f"Loaded uint8 FITS. Shape: {loaded_uint8_data.shape}, dtype: {loaded_uint8_data.dtype}, Min: {np.min(loaded_uint8_data)}, Max: {np.max(loaded_uint8_data)}")
        assert loaded_uint8_data.dtype == np.float32
        assert np.allclose(loaded_uint8_data, uint8_fits_data.astype(np.float32))
        # Test saving this back
        save_uint8_fits_path = get_test_filepath('dummy_uint8_save.fits')
        save_image(save_uint8_fits_path, loaded_uint8_data.astype(np.uint8), format_str='fits') # Save as uint8
        with fits.open(save_uint8_fits_path) as hdul_saved_uint8:
            # Astropy automatically applies BSCALE and BZERO when accessing .data
            print(f"Header of saved uint8 FITS ('{save_uint8_fits_path}'):")
            print(repr(hdul_saved_uint8[0].header))
            raw_reloaded_data_from_fits = hdul_saved_uint8[0].data # This should be scaled by Astropy
            print(f"Data type from hdul[0].data: {raw_reloaded_data_from_fits.dtype}")
            print(f"Min/Max of raw_reloaded_data_from_fits: {np.min(raw_reloaded_data_from_fits)}, {np.max(raw_reloaded_data_from_fits)}")

            reloaded_data = raw_reloaded_data_from_fits.astype(np.float32)
            
            print(f"Min/Max of reloaded_data (float32): {np.min(reloaded_data)}, {np.max(reloaded_data)}")
            print(f"Min/Max of original uint8_fits_data: {np.min(uint8_fits_data)}, {np.max(uint8_fits_data)}")

            if not np.allclose(reloaded_data, uint8_fits_data.astype(np.float32)):
                print("Assertion failed for reloaded uint8 FITS data.")
                mismatches = np.where(np.abs(reloaded_data - uint8_fits_data.astype(np.float32)) > 1e-5)
                if len(mismatches[0]) > 0:
                    print(f"First few mismatches (expected vs got) at indices {mismatches[0][:5]}, {mismatches[1][:5]}:")
                    for i in range(min(5, len(mismatches[0]))):
                        r, c = mismatches[0][i], mismatches[1][i]
                        print(f"  idx ({r},{c}): Expected {uint8_fits_data[r,c]}, Got {reloaded_data[r,c]} (raw from fits: {raw_reloaded_data_from_fits[r,c]})")
            
            assert np.allclose(reloaded_data, uint8_fits_data.astype(np.float32))


    # Test TIFF
    print("\n--- Testing TIFF ---")
    tiff_path = get_test_filepath('dummy_test.tif')
    # Create a sample TIFF with Pillow (16-bit grayscale)
    dummy_tiff_data_uint16 = (np.arange(100, dtype=np.float32).reshape(10, 10) * 600).astype(np.uint16) # Values up to ~60000
    img_tiff_uint16 = Image.fromarray(dummy_tiff_data_uint16, mode='I;16')
    img_tiff_uint16.save(tiff_path)
    
    loaded_data, metadata_tiff = load_image(tiff_path)
    if loaded_data is not None:
        print(f"Loaded TIFF (uint16). Shape: {loaded_data.shape}, Data type: {loaded_data.dtype}, Min: {np.min(loaded_data)}, Max: {np.max(loaded_data)}")
        assert loaded_data.shape == (10,10)
        assert loaded_data.dtype == np.float32 # Should be converted to float32
        assert np.allclose(loaded_data, dummy_tiff_data_uint16.astype(np.float32))
        save_tiff_path = get_test_filepath('dummy_test_save.tif')
        save_image(save_tiff_path, loaded_data.astype(np.uint16)) # Save back as uint16

    # Test Float32 TIFF saving and loading
    tiff_float_path = get_test_filepath('dummy_float32.tif')
    dummy_tiff_data_float32 = (np.random.rand(10,10) * 1000).astype(np.float32)
    save_image(tiff_float_path, dummy_tiff_data_float32, format_str='tiff')
    loaded_float_tiff, _ = load_image(tiff_float_path)
    if loaded_float_tiff is not None:
        print(f"Loaded float32 TIFF. Shape: {loaded_float_tiff.shape}, dtype: {loaded_float_tiff.dtype}, Min: {np.min(loaded_float_tiff)}, Max: {np.max(loaded_float_tiff)}")
        assert loaded_float_tiff.dtype == np.float32
        assert np.allclose(loaded_float_tiff, dummy_tiff_data_float32, atol=1e-5)


    # Test JPG
    print("\n--- Testing JPG ---")
    jpg_path = get_test_filepath('dummy_test.jpg')
    dummy_jpg_data_uint8 = (np.arange(100, dtype=np.float32).reshape(10, 10) * 2.5).astype(np.uint8) # Values 0-247
    img_jpg_uint8 = Image.fromarray(dummy_jpg_data_uint8, mode='L')
    img_jpg_uint8.save(jpg_path)

    loaded_data, metadata_jpg = load_image(jpg_path)
    if loaded_data is not None:
        print(f"Loaded JPG. Shape: {loaded_data.shape}, Data type: {loaded_data.dtype}, Min: {np.min(loaded_data)}, Max: {np.max(loaded_data)}")
        assert loaded_data.shape == (10,10)
        assert loaded_data.dtype == np.float32 # Converted to float32
        # JPG is lossy, so allow some tolerance
        assert np.allclose(loaded_data, dummy_jpg_data_uint8.astype(np.float32), atol=5) # Increased tolerance for JPG
        save_jpg_path = get_test_filepath('dummy_test_save.jpg')
        save_image(save_jpg_path, loaded_data)

    # Test loading a color JPG (should be converted to grayscale)
    jpg_color_path = get_test_filepath('dummy_color_test.jpg')
    dummy_color_jpg_data = (np.random.rand(10, 10, 3) * 255).astype(np.uint8)
    img_color_jpg = Image.fromarray(dummy_color_jpg_data, mode='RGB')
    img_color_jpg.save(jpg_color_path)
    loaded_color_as_gray, _ = load_image(jpg_color_path)
    if loaded_color_as_gray is not None:
        print(f"Loaded color JPG as grayscale. Shape: {loaded_color_as_gray.shape}, dtype: {loaded_color_as_gray.dtype}")
        assert loaded_color_as_gray.ndim == 2
        assert loaded_color_as_gray.shape == (10,10)

    # Test PNG
    print("\n--- Testing PNG ---")
    png_path_8bit = get_test_filepath('dummy_test_8bit.png')
    dummy_png_data_uint8 = np.array([[0, 50], [150, 255]], dtype=np.uint8)
    save_image(png_path_8bit, dummy_png_data_uint8, format_str='png')
    loaded_png_8bit, _ = load_image(png_path_8bit)
    if loaded_png_8bit is not None:
        print(f"Loaded 8-bit PNG. Shape: {loaded_png_8bit.shape}, dtype: {loaded_png_8bit.dtype}, Min: {np.min(loaded_png_8bit)}, Max: {np.max(loaded_png_8bit)}")
        assert np.allclose(loaded_png_8bit, dummy_png_data_uint8.astype(np.float32))

    png_path_16bit = get_test_filepath('dummy_test_16bit.png')
    dummy_png_data_uint16 = (np.arange(4, dtype=np.uint16).reshape(2,2) * 10000) + 1000 # Values: 1000, 11000, 21000, 31000
    save_image(png_path_16bit, dummy_png_data_uint16, format_str='png')
    loaded_png_16bit, _ = load_image(png_path_16bit)
    if loaded_png_16bit is not None:
        print(f"Loaded 16-bit PNG. Shape: {loaded_png_16bit.shape}, dtype: {loaded_png_16bit.dtype}, Min: {np.min(loaded_png_16bit)}, Max: {np.max(loaded_png_16bit)}")
        assert loaded_png_16bit.dtype == np.float32
        assert np.allclose(loaded_png_16bit, dummy_png_data_uint16.astype(np.float32))


    print("\n--- Testing Unsupported Format ---")
    txt_path = get_test_filepath("dummy_test.txt")
    with open(txt_path, "w") as f:
        f.write("this is not an image")
    loaded_data, metadata = load_image(txt_path)
    assert loaded_data is None

    print("\n--- Testing Non-existent File ---")
    loaded_data, metadata = load_image(get_test_filepath('non_existent_file.fits'))
    assert loaded_data is None
    
    print(f"\nAll io_utils tests completed. Check '{test_dir}/' for output files.")
