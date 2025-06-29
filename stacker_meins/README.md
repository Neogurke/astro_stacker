# Astro Photo Stacker & Editor

A Python application for stacking and editing astronomical photos.

## Features (Planned/Implemented)

*   **Input Formats:** FITS, TIFF, JPG, PNG
*   **Stacking Techniques:**
    *   Average
    *   Median
    *   Kappa-Sigma Clipping
    *   (Alignment is a planned prerequisite for effective stacking of unaligned images)
*   **Editing Features:**
    *   Brightness & Contrast
    *   Levels (Black point, White point, Mid-point/Gamma)
    *   Curves
    *   Sharpening (Unsharp Mask, Convolution)
    *   Noise Reduction (Gaussian, Median, Non-Local Means, Bilateral)
*   **User Interface:** GUI based (developed using PyQt6).

## Project Structure

```
astro_stack_edit/
├── main.py               # Main application entry point
├── requirements.txt      # Python dependencies
├── README.md             # This file
├── core/                 # Core processing logic
│   ├── __init__.py
│   ├── processor.py      # ImageProcessor class for stacking and editing
│   └── (calibration.py)  # Placeholder for calibration logic
├── ui/                   # User interface components
│   ├── __init__.py
│   └── main_window.py    # MainWindow class for the GUI
├── utils/                # Utility functions
│   ├── __init__.py
│   └── io_utils.py       # Image loading and saving functions
└── resources/            # (Planned) For icons, etc.
    └── icons/
```

## Setup and Running from Source

These instructions assume you have Python 3.9+ installed.

**1. Create a Virtual Environment (Recommended):**

   It's highly recommended to use a virtual environment to manage dependencies for this project.

   *macOS/Linux:*
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   *Windows:*
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

**2. Install Dependencies:**

   With your virtual environment activated, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

**3. Running the Application:**

   Once dependencies are installed, you can run the application:
   ```bash
   python main.py
   ```

**Important Notes for Linux Users (GUI):**
*   The application uses PyQt6, which relies on system libraries for display. You may need to install Qt platform plugins and XCB libraries if they are not present. For Debian/Ubuntu based systems, this might include packages like:
    `sudo apt-get install -y libxcb-cursor0 libxkbcommon-x11-0 qt6-qpa-plugins libgl1 libegl1`
    (And potentially others, depending on your specific distribution and setup).
*   If running in a headless environment or encountering display connection errors, you might need to use a virtual X server like Xvfb:
    ```bash
    sudo apt-get install xvfb
    xvfb-run -a python main.py
    ```

## Code Development and Testing

*   Core image processing logic is in `core/processor.py`.
*   Image file I/O utilities are in `utils/io_utils.py`.
*   Both `processor.py` and `io_utils.py` contain `if __name__ == '__main__':` blocks with unit tests that can be run directly:
    ```bash
    python utils/io_utils.py
    python core/processor.py
    ```
    *(Note: Recent environmental issues in the development sandbox caused timeouts when running `core/processor.py` tests. These tests were passing prior to the installation of all editing-related dependencies like scikit-image.)*

## Future Development / Packaging

*   **Alignment:** Implement robust image alignment before stacking.
*   **Calibration:** Add dark, flat, and bias frame calibration.
*   **Advanced Editing Tools:** More sophisticated curve controls, selective adjustments, etc.
*   **GUI Enhancements:**
    *   Proper image display instead of placeholder.
    *   Interactive controls for all editing features.
    *   Progress bars for long operations.
    *   Project management (handling multiple files, remembering settings).
*   **Standalone Executables:** For easier distribution without requiring users to install Python and dependencies, tools like [PyInstaller](https://pyinstaller.readthedocs.io/en/stable/) or [cx_Freeze](https://cx-freeze.readthedocs.io/en/latest/) could be used to package the application into standalone executables for Windows, macOS, and Linux. This would involve creating a build script (e.g., a `.spec` file for PyInstaller).

## Troubleshooting (Development Environment)

*   **Qt Platform Plugin Errors:** If you see errors like "Could not load the Qt platform plugin 'xcb'", ensure the necessary XCB libraries and Qt platform adapter plugins are installed on your system (see "Important Notes for Linux Users").
*   **Display Connection Errors:** If running in a headless environment, ensure an X server (like Xvfb) is available and the `DISPLAY` environment variable is set correctly.
*   **Slow Imports / Timeouts:** If the script hangs or times out very early (before Python code execution seems to begin), it might indicate a conflict between C-extension modules of installed libraries (e.g., NumPy, SciPy, scikit-image, PyQt6, Astropy). This can be hard to debug and might require experimenting with library versions or a cleaner environment.

This README provides a basic guide to get started with the application from source.
```
