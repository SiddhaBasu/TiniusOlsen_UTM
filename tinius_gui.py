#!/usr/bin/env python3
import os
import sys
import cv2
import time
from datetime import datetime

from PyQt5 import QtCore, QtGui, QtWidgets

# Import your measurement function
import outline  # outline.py must be in the same directory


CAPTURE_DIR = "captures"
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Which camera index is "top-down" / primary?
PRIMARY_CAM_INDEX = 0
SECONDARY_CAM_INDEX = 1

# Approx Tinius Olsen teal-ish accent
TIN_TEAL = "#00656B"
TIN_DARK = "#101820"
TIN_LIGHT_BG = "#F5F7FA"


class MeasureWorker(QtCore.QObject):
    """
    Worker run in a separate thread to call outline.measure(image_path)
    so the GUI doesn't freeze during processing.
    """
    finished = QtCore.pyqtSignal(dict, str)  # metrics, outline_path
    error = QtCore.pyqtSignal(str)

    def __init__(self, image_path: str):
        super().__init__()
        self.image_path = image_path

    @QtCore.pyqtSlot()
    def run(self):
        try:
            metrics = outline.measure(self.image_path)

            # Try to find an outline image path from metrics, else assume "outline.png"
            outline_path = metrics.get("outline_path")
            if not outline_path:
                # fallback: same directory as outline.py or current working dir
                # adjust if your outline.py uses a different convention
                possible = [
                    os.path.join(os.path.dirname(self.image_path), "outline.png"),
                    "outline.png",
                ]
                outline_path = None
                for p in possible:
                    if os.path.exists(p):
                        outline_path = p
                        break
            if not outline_path:
                outline_path = ""  # will be handled on GUI side

            self.finished.emit(metrics, outline_path)
        except Exception as e:
            self.error.emit(str(e))


class CameraWidget(QtWidgets.QLabel):
    """
    Simple QLabel that shows frames from an OpenCV VideoCapture.
    """

    def __init__(self, cap_index: int, parent=None):
        super().__init__(parent)
        self.cap_index = cap_index
        self.cap = cv2.VideoCapture(self.cap_index)
        # Set a reasonable resolution; adjust as needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #222; border: 1px solid #444;")

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def update_view(self):
        frame = self.read_frame()
        if frame is None:
            return
        # Convert BGR to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        # Scale to label size while keeping aspect ratio
        self.setPixmap(pix.scaled(self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

    def close(self):
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        super().close()

    def grab_current_frame(self):
        """
        Return the latest BGR frame (non-resized) for saving/processing.
        """
        frame = self.read_frame()
        return frame


class TiniusMainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Tinius Olsen - Specimen Measurement")
        self.resize(1600, 900)

        # Global style
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {TIN_LIGHT_BG};
                color: {TIN_DARK};
                font-family: Arial, Helvetica, sans-serif;
            }}
            QPushButton {{
                background-color: {TIN_TEAL};
                color: white;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
            }}
            QPushButton:disabled {{
                background-color: #7a8a8c;
            }}
            QPlainTextEdit {{
                background-color: white;
                border: 1px solid #aaa;
            }}
        """)

        # --- Header / logo bar ---
        header = QtWidgets.QFrame()
        header.setFixedHeight(70)
        header.setStyleSheet(f"background-color: {TIN_DARK}; color: white;")

        logo_label = QtWidgets.QLabel("Tinius Olsen")
        logo_font = QtGui.QFont("Arial", 20, QtGui.QFont.Bold)
        logo_label.setFont(logo_font)
        logo_label.setStyleSheet("color: white;")

        subtitle_label = QtWidgets.QLabel("Universal Testing Machine - Vision Measurement")
        subtitle_label.setStyleSheet("color: #cfd8dc;")
        subtitle_font = QtGui.QFont("Arial", 11)
        subtitle_label.setFont(subtitle_font)

        header_layout = QtWidgets.QVBoxLayout(header)
        header_layout.setContentsMargins(20, 8, 20, 8)
        header_layout.addWidget(logo_label, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        header_layout.addWidget(subtitle_label, alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        # --- Cameras ---
        self.cam_primary = CameraWidget(PRIMARY_CAM_INDEX)
        self.cam_secondary = CameraWidget(SECONDARY_CAM_INDEX)

        cameras_frame = QtWidgets.QFrame()
        cams_layout = QtWidgets.QHBoxLayout(cameras_frame)
        cams_layout.setContentsMargins(10, 10, 10, 10)
        cams_layout.setSpacing(10)
        cams_layout.addWidget(self.cam_primary, stretch=1)
        cams_layout.addWidget(self.cam_secondary, stretch=1)

        # --- Capture & status controls ---
        self.btn_capture = QtWidgets.QPushButton("Capture & Measure (Primary Camera)")
        self.btn_capture.clicked.connect(self.on_capture)

        self.status_label = QtWidgets.QLabel("Ready.")
        self.status_label.setStyleSheet(f"color: {TIN_DARK};")

        ctrl_layout = QtWidgets.QHBoxLayout()
        ctrl_layout.setContentsMargins(10, 0, 10, 0)
        ctrl_layout.addWidget(self.btn_capture)
        ctrl_layout.addStretch(1)
        ctrl_layout.addWidget(self.status_label)

        # --- Outline display ---
        self.outline_label = QtWidgets.QLabel()
        self.outline_label.setAlignment(QtCore.Qt.AlignCenter)
        self.outline_label.setMinimumSize(400, 300)
        self.outline_label.setStyleSheet("background-color: #ffffff; border: 1px solid #aaaaaa;")

        outline_title = QtWidgets.QLabel("Outline")
        outline_title.setAlignment(QtCore.Qt.AlignLeft)
        outline_title.setStyleSheet(f"color: {TIN_DARK}; font-weight: bold;")

        outline_layout = QtWidgets.QVBoxLayout()
        outline_layout.addWidget(outline_title)
        outline_layout.addWidget(self.outline_label, stretch=1)

        # --- Metrics / data display ---
        metrics_title = QtWidgets.QLabel("Measurement Data")
        metrics_title.setAlignment(QtCore.Qt.AlignLeft)
        metrics_title.setStyleSheet(f"color: {TIN_DARK}; font-weight: bold;")

        self.metrics_text = QtWidgets.QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setMinimumWidth(380)

        metrics_layout = QtWidgets.QVBoxLayout()
        metrics_layout.addWidget(metrics_title)
        metrics_layout.addWidget(self.metrics_text, stretch=1)

        # Bottom pane: outline on left, metrics on right
        bottom_frame = QtWidgets.QFrame()
        bottom_layout = QtWidgets.QHBoxLayout(bottom_frame)
        bottom_layout.setContentsMargins(10, 10, 10, 10)
        bottom_layout.setSpacing(10)
        bottom_layout.addLayout(outline_layout, stretch=3)
        bottom_layout.addLayout(metrics_layout, stretch=2)

        # Main layout
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(header)
        main_layout.addWidget(cameras_frame, stretch=3)
        main_layout.addLayout(ctrl_layout)
        main_layout.addWidget(bottom_frame, stretch=3)

        # Timer to refresh camera views
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_cameras)
        self.timer.start(40)  # ~25 fps

        # Thread holder
        self.measure_thread = None
        self.measure_worker = None

    # -------- Camera refresh --------
    def update_cameras(self):
        self.cam_primary.update_view()
        self.cam_secondary.update_view()

    # -------- Capture & measure --------
    def on_capture(self):
        self.btn_capture.setEnabled(False)
        self.status_label.setText("Capturing image...")

        frame = self.cam_primary.grab_current_frame()
        if frame is None:
            self.status_label.setText("Error: primary camera frame not available.")
            self.btn_capture.setEnabled(True)
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = os.path.join(CAPTURE_DIR, f"specimen_{ts}.jpg")
        cv2.imwrite(img_path, frame)

        self.status_label.setText("Processing image in outline.measure()...")
        self.metrics_text.setPlainText(f"Processing {os.path.basename(img_path)}...")

        # Background worker
        self.measure_thread = QtCore.QThread()
        self.measure_worker = MeasureWorker(img_path)
        self.measure_worker.moveToThread(self.measure_thread)
        self.measure_thread.started.connect(self.measure_worker.run)
        self.measure_worker.finished.connect(self.on_measure_finished)
        self.measure_worker.error.connect(self.on_measure_error)
        self.measure_worker.finished.connect(self.measure_thread.quit)
        self.measure_worker.error.connect(self.measure_thread.quit)
        self.measure_worker.finished.connect(self.measure_worker.deleteLater)
        self.measure_worker.error.connect(self.measure_worker.deleteLater)
        self.measure_thread.finished.connect(self.measure_thread.deleteLater)
        self.measure_thread.start()

    @QtCore.pyqtSlot(dict, str)
    def on_measure_finished(self, metrics: dict, outline_path: str):
        # Update outline image
        if outline_path and os.path.exists(outline_path):
            pix = QtGui.QPixmap(outline_path)
            self.outline_label.setPixmap(
                pix.scaled(self.outline_label.size(),
                           QtCore.Qt.KeepAspectRatio,
                           QtCore.Qt.SmoothTransformation)
            )
        else:
            self.outline_label.setText("Outline image not found.")

        # Format metrics nicely
        lines = []
        def add(label, key):
            if key in metrics:
                lines.append(f"{label}: {metrics[key]}")

        add("ASTM standard", "astm_standard")
        add("Inferred shape", "shape")
        add("Length (mm)", "length")
        add("Max width (mm)", "width")
        add("Neck width (mm)", "neck_width")
        add("Surface area (mm^2)", "surface_area")
        add("Pixels per mm", "ppmm")
        add("Material (NN)", "nn_material")
        add("NN material confidence", "nn_material_confidence")
        add("NN material votes", "nn_material_votes")

        if not lines:
            lines.append("No metrics returned from outline.measure().")

        self.metrics_text.setPlainText("\n".join(lines))

        self.status_label.setText("Done.")
        self.btn_capture.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def on_measure_error(self, msg: str):
        self.metrics_text.setPlainText(f"Error during measurement:\n{msg}")
        self.status_label.setText("Error.")
        self.btn_capture.setEnabled(True)

    # -------- Cleanup --------
    def closeEvent(self, event):
        try:
            self.timer.stop()
        except Exception:
            pass
        try:
            self.cam_primary.close()
            self.cam_secondary.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = TiniusMainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
