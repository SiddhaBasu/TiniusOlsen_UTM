import sys
import os
import time

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

# ---------------------------------------------------------------------
# Prevent outline.py from auto-running its CLI on import
# ---------------------------------------------------------------------
_original_argv = list(sys.argv)
sys.argv = [sys.argv[0], "--test", "NONE"]
import outline  # this will NOT run the default D638 test now
sys.argv = _original_argv

from outline import measure, Material

# ---------------------------------------------------------------------
# Paths / config
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# outline.measure expects image_name relative to ./images
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(IMAGES_DIR, exist_ok=True)

# outline saves to ./outputs
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

CAM_INDEX_TOP = 0   # top-down USB camera
CAM_INDEX_SIDE = 1  # side USB camera


# ---------------------------------------------------------------------
# Worker thread for running measure() so GUI does not freeze
# ---------------------------------------------------------------------

class MeasureWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict, str)   # metrics, outline_path
    failed = QtCore.pyqtSignal(str)

    def __init__(self, image_name: str, material: Material):
        super().__init__()
        self.image_name = image_name
        self.material = material

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # Call outline.measure(image_name, material)
            metrics = measure(self.image_name, self.material)

            base = os.path.splitext(os.path.basename(self.image_name))[0]
            outline_path = os.path.join(OUT_DIR, f"{base}_outline.png")
            if not os.path.exists(outline_path):
                # outline.py *should* have written this; if not, complain
                raise FileNotFoundError(f"Expected outline image not found: {outline_path}")

            self.finished.emit(metrics, outline_path)
        except Exception as e:
            self.failed.emit(str(e))


# ---------------------------------------------------------------------
# Main GUI
# ---------------------------------------------------------------------

class TiniusGui(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tinius Olsen Specimen Measurement")
        self.resize(1400, 900)
        self.setStyleSheet("background-color: #f3f5f7;")

        # Cameras
        self.cam_top = cv2.VideoCapture(CAM_INDEX_TOP)
        self.cam_side = cv2.VideoCapture(CAM_INDEX_SIDE)

        # If a cam fails, do not crash; we just show blank
        if not self.cam_top.isOpened():
            print("WARNING: could not open camera index 0 (top).")
        if not self.cam_side.isOpened():
            print("WARNING: could not open camera index 1 (side).")

        self.last_frame_top = None
        self.last_frame_side = None

        # Live view labels
        self.label_top = QtWidgets.QLabel("Top Camera")
        self.label_side = QtWidgets.QLabel("Side Camera")

        common_cam_style = (
            "background-color: #222; color: #ddd; "
            "border-radius: 8px; border: 1px solid #444;"
        )
        for lab in (self.label_top, self.label_side):
            lab.setAlignment(QtCore.Qt.AlignCenter)
            lab.setMinimumSize(480, 360)
            lab.setStyleSheet(common_cam_style)

        # Outline preview
        self.label_outline = QtWidgets.QLabel("Outline preview")
        self.label_outline.setAlignment(QtCore.Qt.AlignCenter)
        self.label_outline.setMinimumSize(640, 360)
        self.label_outline.setStyleSheet(common_cam_style)

        # Metrics panel
        self.metrics_label = QtWidgets.QLabel("Measurements")
        self.metrics_label.setStyleSheet(
            "font-size: 18px; font-weight: 600; color: #222;"
        )

        self.metrics_text = QtWidgets.QTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setStyleSheet(
            "background-color: #ffffff; border-radius: 8px; "
            "border: 1px solid #ccc; font-family: 'Consolas', monospace; "
            "font-size: 13px;"
        )

        # Material selection (rough)
        self.material_combo = QtWidgets.QComboBox()
        self.material_combo.addItems(["Metal", "Plastic"])
        self.material_combo.setStyleSheet(
            "background-color: #ffffff; border-radius: 6px; "
            "border: 1px solid #aaa; padding: 3px; font-size: 13px;"
        )

        # Measure button
        self.btn_capture = QtWidgets.QPushButton("Capture & Measure")
        self.btn_capture.setFixedHeight(40)
        self.btn_capture.setStyleSheet(
            "background-color: #0067a6; color: white; "
            "border-radius: 8px; font-size: 16px; font-weight: 600;"
        )
        self.btn_capture.clicked.connect(self.capture_and_measure)

        # Status label
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("color: #555; font-size: 13px;")

        # Logo / company branding
        self.logo_label = QtWidgets.QLabel("Tinius Olsen")
        self.logo_label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
        self.logo_label.setStyleSheet(
            "font-family: 'Arial'; font-weight: 700; font-size: 26px; "
            "letter-spacing: 3px; color: #004b63;"
        )

        self.tagline_label = QtWidgets.QLabel("Testing Machines & Instruments")
        self.tagline_label.setStyleSheet(
            "color: #666; font-size: 11px; font-style: italic;"
        )

        # ---- Layout ----
        header_layout = QtWidgets.QVBoxLayout()
        header_layout.addWidget(self.logo_label)
        header_layout.addWidget(self.tagline_label)
        header_layout.setSpacing(0)

        top_bar = QtWidgets.QHBoxLayout()
        top_bar.addLayout(header_layout)
        top_bar.addStretch(1)

        cams_layout = QtWidgets.QHBoxLayout()
        cams_layout.addWidget(self.label_top, 1)
        cams_layout.addWidget(self.label_side, 1)

        metrics_controls = QtWidgets.QVBoxLayout()
        metrics_controls.addWidget(self.metrics_label)
        metrics_controls.addWidget(self.metrics_text, 3)

        controls_row = QtWidgets.QHBoxLayout()
        controls_row.addWidget(QtWidgets.QLabel("Material:"))
        controls_row.addWidget(self.material_combo)
        controls_row.addStretch(1)
        controls_row.addWidget(self.btn_capture)

        metrics_controls.addLayout(controls_row)
        metrics_controls.addWidget(self.status_label)

        bottom_layout = QtWidgets.QHBoxLayout()
        bottom_layout.addWidget(self.label_outline, 3)
        bottom_layout.addLayout(metrics_controls, 2)

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addLayout(top_bar)
        main_layout.addSpacing(10)
        main_layout.addLayout(cams_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(bottom_layout)

        # Timer to refresh camera frames
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(33)  # ~30 FPS

        # Thread / worker handles
        self.worker_thread = None
        self.worker = None

    # ------------------------------------------------------------------
    # Camera handling
    # ------------------------------------------------------------------

    def update_frames(self):
        self._update_single_cam(self.cam_top, self.label_top, "Top Camera", is_top=True)
        self._update_single_cam(self.cam_side, self.label_side, "Side Camera", is_top=False)

    def _update_single_cam(self, cam, label, placeholder, is_top: bool):
        if cam is None or not cam.isOpened():
            return
        ret, frame = cam.read()
        if not ret:
            return

        if is_top:
            self.last_frame_top = frame
        else:
            self.last_frame_side = frame

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        pix = pix.scaled(label.width(), label.height(),
                         QtCore.Qt.KeepAspectRatio,
                         QtCore.Qt.SmoothTransformation)
        label.setPixmap(pix)

    # ------------------------------------------------------------------
    # Capture + measurement
    # ------------------------------------------------------------------

    def capture_and_measure(self):
        if self.last_frame_top is None:
            self.status_label.setText("No frame from top camera yet.")
            return

        # Save image into ./images and pass only filename to outline.measure()
        timestamp = int(time.time())
        filename = f"specimen_{timestamp}.jpg"
        img_path = os.path.join(IMAGES_DIR, filename)

        try:
            # OpenCV expects BGR
            cv2.imwrite(img_path, self.last_frame_top)
        except Exception as e:
            self.status_label.setText(f"Failed to save image: {e}")
            return

        # Decide material enum from combo box
        material_text = self.material_combo.currentText().lower()
        material_enum = Material.METAL if "metal" in material_text else Material.PLASTIC

        self.status_label.setText("Measuring...")
        self.btn_capture.setEnabled(False)

        # Launch worker thread
        self.worker_thread = QtCore.QThread()
        self.worker = MeasureWorker(filename, material_enum)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.on_measure_finished)
        self.worker.failed.connect(self.on_measure_failed)

        # Clean up
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.failed.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.failed.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()

    @QtCore.pyqtSlot(dict, str)
    def on_measure_finished(self, metrics: dict, outline_path: str):
        # Show outline image
        if os.path.exists(outline_path):
            pix = QtGui.QPixmap(outline_path)
            pix = pix.scaled(self.label_outline.width(),
                             self.label_outline.height(),
                             QtCore.Qt.KeepAspectRatio,
                             QtCore.Qt.SmoothTransformation)
            self.label_outline.setPixmap(pix)
        else:
            self.label_outline.setText("Outline image not found.")

        # Dump metrics nicely
        lines = []
        for k, v in metrics.items():
            lines.append(f"{k:20s}: {v}")
        self.metrics_text.setPlainText("\n".join(lines))

        self.status_label.setText("Measurement complete.")
        self.btn_capture.setEnabled(True)

    @QtCore.pyqtSlot(str)
    def on_measure_failed(self, msg: str):
        self.status_label.setText(f"Error during measurement: {msg}")
        self.btn_capture.setEnabled(True)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        if self.cam_top is not None:
            self.cam_top.release()
        if self.cam_side is not None:
            self.cam_side.release()
        return super().closeEvent(event)


# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = TiniusGui()
    gui.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
