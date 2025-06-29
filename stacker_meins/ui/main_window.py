import sys
from PyQt6.QtWidgets import (QMainWindow, QVBoxLayout, QHBoxLayout, QWidget,
                             QLabel, QFrame, QSizePolicy, QToolBar, QStatusBar,
                             QDockWidget, QListWidget, QFileDialog, QMessageBox)
from PyQt6.QtGui import QAction, QIcon, QPixmap, QImage
from PyQt6.QtCore import Qt, QSize
import numpy as np

# Assuming io_utils and ImageProcessor will be imported when project structure is more complete
# For now, to make save testable, we can have a mock save_image or assume its presence.
from utils.io_utils import save_image # Let's assume this can be imported
# from core.processor import ImageProcessor # For actual image data later

# It's good practice to manage resources like icons properly.
# For now, we might use placeholder text or require a 'resources' folder.
# from PyQt6.QtGui import QPixmap
# import os
# SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
# RESOURCES_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), 'resources', 'icons')


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Astro Photo Stacker & Editor")
        self.setGeometry(100, 100, 1200, 800) # x, y, width, height

        # Central Widget and Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)

        # Image Display Area (Placeholder)
        self.image_display_label = QLabel("Image Display Area") # Renamed for clarity
        self.image_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.image_display_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.main_layout.addWidget(self.image_display_label, 3) # Give more stretch factor

        # Control Panel (Placeholder - using QDockWidget for collapsibility)
        self.create_control_panel_dock()
        
        self._create_actions()
        self._create_menu_bar()
        self._create_tool_bar()
        self._create_status_bar()

        # Store current image data (e.g., as a NumPy array)
        self.current_image_data: np.ndarray | None = None 
        self.current_image_filepath: str | None = None # To store original filepath or last saved path

        # Create a dummy image for testing save functionality
        self.load_dummy_image_for_testing()

        # TODO: Initialize ProjectManager and ImageProcessor from core
        # self.project_manager = ProjectManager()
        # self.image_processor = ImageProcessor() # core.processor.ImageProcessor()

        print("MainWindow initialized.")

    def load_dummy_image_for_testing(self):
        """Loads a simple NumPy array as a dummy image for testing save."""
        # Create a small grayscale image
        self.current_image_data = np.random.randint(0, 256, size=(100, 150), dtype=np.uint8)
        # Display this dummy image (basic QPixmap conversion)
        if self.current_image_data is not None:
            try:
                height, width = self.current_image_data.shape
                bytes_per_line = width
                q_image = QImage(self.current_image_data.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
                pixmap = QPixmap.fromImage(q_image)
                self.image_display_label.setPixmap(pixmap.scaled(self.image_display_label.size(), 
                                                               Qt.AspectRatioMode.KeepAspectRatio, 
                                                               Qt.TransformationMode.SmoothTransformation))
                self.statusBar.showMessage("Dummy image loaded for testing.", 3000)
            except Exception as e:
                print(f"Error displaying dummy image: {e}")
                self.image_display_label.setText("Error displaying dummy image.")


    def _create_actions(self):
        # File actions
        self.open_action = QAction(QIcon.fromTheme("document-open", QIcon(None)), "&Open...", self) # Placeholder icon
        self.open_action.setStatusTip("Open image(s)")
        self.open_action.triggered.connect(self.open_file_dialog) 
        
        self.save_action = QAction(QIcon.fromTheme("document-save", QIcon(None)), "&Save", self)
        self.save_action.setStatusTip("Save current image")
        self.save_action.triggered.connect(self.save_file_dialog)
        
        self.save_as_action = QAction(QIcon.fromTheme("document-save-as", QIcon(None)), "Save &As...", self)
        self.save_as_action.setStatusTip("Save current image to a new file")
        self.save_as_action.triggered.connect(self.save_as_file_dialog)

        self.exit_action = QAction(QIcon.fromTheme("application-exit", QIcon(None)), "&Exit", self)
        self.exit_action.setStatusTip("Exit application")
        self.exit_action.triggered.connect(self.close)

        # Edit actions (placeholders)
        self.undo_action = QAction(QIcon.fromTheme("edit-undo", QIcon(None)), "&Undo", self)
        self.redo_action = QAction(QIcon.fromTheme("edit-redo", QIcon(None)), "&Redo", self)

        # View actions (placeholders)
        self.zoom_in_action = QAction(QIcon.fromTheme("zoom-in", QIcon(None)), "Zoom &In", self)
        self.zoom_out_action = QAction(QIcon.fromTheme("zoom-out", QIcon(None)), "Zoom &Out", self)
        self.fit_to_view_action = QAction(QIcon.fromTheme("zoom-fit-best", QIcon(None)), "&Fit to View", self)

        # Stacking actions (placeholder)
        self.stack_images_action = QAction("Stack Images...", self)
        self.stack_images_action.setStatusTip("Open stacking dialog")
        # self.stack_images_action.triggered.connect(self.open_stacking_dialog) # TODO

        # Help actions
        self.about_action = QAction("&About", self)
        # self.about_action.triggered.connect(self.show_about_dialog) # TODO


    def _create_menu_bar(self):
        menu_bar = self.menuBar()

        # File menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addAction(self.save_action)
        file_menu.addAction(self.save_as_action)
        file_menu.addSeparator()
        file_menu.addAction(self.exit_action)

        # Edit menu
        edit_menu = menu_bar.addMenu("&Edit")
        edit_menu.addAction(self.undo_action)
        edit_menu.addAction(self.redo_action)
        # ... more editing actions (brightness, levels, curves etc. will be here or in a dedicated Image menu)

        # View menu
        view_menu = menu_bar.addMenu("&View")
        view_menu.addAction(self.zoom_in_action)
        view_menu.addAction(self.zoom_out_action)
        view_menu.addAction(self.fit_to_view_action)
        view_menu.addSeparator()
        # Action to toggle control panel visibility
        self.view_control_panel_action = self.control_panel_dock.toggleViewAction()
        self.view_control_panel_action.setText("Show/Hide &Control Panel")
        self.view_control_panel_action.setStatusTip("Show or hide the Control Panel")
        view_menu.addAction(self.view_control_panel_action)


        # Image Menu (for processing operations)
        image_menu = menu_bar.addMenu("&Image")
        # Example: image_menu.addAction(self.brightness_contrast_action)
        # Example: image_menu.addAction(self.levels_action)
        
        # Stacking Menu
        stacking_menu = menu_bar.addMenu("&Stacking")
        stacking_menu.addAction(self.stack_images_action)

        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self.about_action)
        
    def _create_tool_bar(self):
        # Main toolbar for common actions
        main_toolbar = QToolBar("Main Toolbar")
        main_toolbar.setIconSize(QSize(24, 24)) # Default is often small
        self.addToolBar(main_toolbar)
        main_toolbar.addAction(self.open_action)
        main_toolbar.addAction(self.save_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.undo_action)
        main_toolbar.addAction(self.redo_action)
        main_toolbar.addSeparator()
        main_toolbar.addAction(self.zoom_in_action)
        main_toolbar.addAction(self.zoom_out_action)
        main_toolbar.addAction(self.fit_to_view_action)

        # Could have more toolbars, e.g., for editing tools
        # edit_tools_toolbar = QToolBar("Editing Tools")
        # self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, edit_tools_toolbar) # Example placement
        # ... add specific editing tool actions ...

    def _create_status_bar(self):
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready", 3000) # Message disappears after 3 seconds

    def create_control_panel_dock(self):
        self.control_panel_dock = QDockWidget("Control Panel", self)
        self.control_panel_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        # Create a dummy widget for the dock content for now
        self.control_panel_widget = QWidget()
        self.control_panel_layout = QVBoxLayout(self.control_panel_widget)
        
        # Example: File list
        self.file_list_label = QLabel("Loaded Files:")
        self.file_list_widget = QListWidget()
        self.control_panel_layout.addWidget(self.file_list_label)
        self.control_panel_layout.addWidget(self.file_list_widget)

        # Example: Placeholder for tool options
        self.tool_options_label = QLabel("Tool Options:")
        self.tool_options_area = QFrame()
        self.tool_options_area.setFrameShape(QFrame.Shape.StyledPanel)
        self.tool_options_area.setMinimumHeight(200)
        self.control_panel_layout.addWidget(self.tool_options_label)
        self.control_panel_layout.addWidget(self.tool_options_area)
        
        self.control_panel_layout.addStretch() # Pushes content to the top

        self.control_panel_dock.setWidget(self.control_panel_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.control_panel_dock)


    # --- Placeholder methods for actions ---
    def open_file_dialog(self):
        self.statusBar.showMessage("Open file dialog triggered... (Not implemented yet)", 2000)
        # TODO: Actual implementation will use QFileDialog and interact with core.ImageProcessor / core.ProjectManager
        # filenames, _ = QFileDialog.getOpenFileNames(self, "Open Images", "", 
        #                                           "Images (*.fits *.fit *.fts *.tif *.tiff *.jpg *.jpeg *.png);;All Files (*)")
        # if filenames:
        #     # For simplicity, load the first selected image
        #     self.load_image_into_gui(filenames[0]) 
        pass


    def _save_image_to_path(self, filepath: str):
        """Helper function to save the current image data to a given path."""
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "There is no image data to save.")
            return False

        # Infer format from filepath for save_image function, or use filter from dialog
        # QFileDialog returns a selected filter string like "FITS (*.fits)"
        # We need to extract the actual format like "FITS" or "PNG".
        # For now, io_utils.save_image can infer from extension.
        
        # Add more sophisticated error handling and success messages
        try:
            success = save_image(filepath, self.current_image_data, overwrite=True) # format_str can be None
            if success:
                self.statusBar.showMessage(f"Image saved to {filepath}", 5000)
                self.current_image_filepath = filepath # Update current filepath
                return True
            else:
                QMessageBox.critical(self, "Save Error", f"Could not save image to {filepath}.\nCheck console for details.")
                return False
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"An unexpected error occurred while saving:\n{e}")
            return False

    def save_file_dialog(self):
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "There is no image to save.")
            return

        if self.current_image_filepath: # If there's a current path (image was opened or saved before)
            reply = QMessageBox.question(self, "Save Image", 
                                         f"Save to current file: {self.current_image_filepath}?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Yes:
                self._save_image_to_path(self.current_image_filepath)
            elif reply == QMessageBox.StandardButton.No: # Effectively means "Save As"
                self.save_as_file_dialog()
            # else (Cancel): do nothing
        else: # No current path, so always "Save As"
            self.save_as_file_dialog()


    def save_as_file_dialog(self):
        if self.current_image_data is None:
            QMessageBox.warning(self, "No Image", "There is no image to save.")
            return

        # Define file filters
        # The string format is "Description (*.ext1 *.ext2);;Another Description (*.ext3)"
        filters = "FITS images (*.fits *.fit *.fts);;" \
                  "TIFF images (*.tiff *.tif);;" \
                  "PNG images (*.png);;" \
                  "JPEG images (*.jpg *.jpeg);;" \
                  "All files (*)"
        
        # Suggest a filename if one was loaded (e.g. change extension)
        suggested_path = self.current_image_filepath if self.current_image_filepath else "untitled.fits"

        filepath, selected_filter = QFileDialog.getSaveFileName(self, "Save Image As...", 
                                                              suggested_path, 
                                                              filters)
        if filepath:
            # Note: selected_filter can be used to pass an explicit format to save_image if needed,
            # but io_utils.save_image already infers from extension.
            # Example of how you might parse selected_filter if needed:
            # if "FITS" in selected_filter: format_hint = "FITS"
            # elif "TIFF" in selected_filter: format_hint = "TIFF" ... etc.
            self._save_image_to_path(filepath)


    def show_about_dialog(self):
        self.statusBar.showMessage("About dialog triggered... (Not implemented yet)", 2000)
        # QMessageBox.about(self, "About AstroStacker", "Astro Photo Stacker & Editor\nVersion 0.1")

    def closeEvent(self, event):
        # Override closeEvent to ask for confirmation or save unsaved changes if needed
        print("Application closing.")
        # reply = QMessageBox.question(self, 'Message',
        #                              "Are you sure to quit?", QMessageBox.StandardButton.Yes |
        #                              QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        # if reply == QMessageBox.StandardButton.Yes:
        #     event.accept()
        # else:
        #     event.ignore()
        event.accept() # For now, just accept


if __name__ == '__main__':
    # This part is for testing the MainWindow directly if needed
    # Normally, main.py would run this.
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec())
