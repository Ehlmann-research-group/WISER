from PySide2.QtCore import Qt, QTimer, QRect, QPoint
from PySide2.QtGui import QColor, QPainter, QPen
from PySide2.QtWidgets import QWidget, QVBoxLayout, QLabel

class LoadingOverlay(QWidget):
    def __init__(self, target: QWidget, text: str = "Loading…"):
        super().__init__(target)
        self._target = target
        self._angle = 0
        self._timer = QTimer(self, interval=16, timeout=self._tick)  # ~60 FPS
        self._text = QLabel(text, self)
        self._text.setStyleSheet("color: white; font-weight: 600;")
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.hide()
        target.installEventFilter(self)

    def _apply_geometry(self):
        parent = self.parentWidget()
        if not parent:
            return
        self.setGeometry(parent.rect())
        self.raise_()

    # public API
    def start(self):
        self._apply_geometry()
        self.show()
        self.raise_()
        self._timer.start()

    def stop(self):
        self._timer.stop()
        self.hide()

    # keep overlay sized with target
    def eventFilter(self, obj, ev):
        if obj is self.parentWidget() and ev.type() in (ev.Resize, ev.Move, ev.Show):
            self.resize(obj.size())
        return super().eventFilter(obj, ev)

    def _tick(self):
        self._angle = (self._angle + 6) % 360
        self.update()

    # block interactions underneath
    def mousePressEvent(self, e):  e.accept()
    def mouseReleaseEvent(self, e): e.accept()
    def mouseMoveEvent(self, e):    e.accept()

    # draw spinner + text
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120)) 
        # centered spinner
        r = 28
        center = self.rect().center()
        spinnerRect = QRect(0, 0, r*2, r*2)
        spinnerRect.moveCenter(center - QPoint(0, 12))
        # track
        pen = QPen(QColor(0,0,0,60), 4)
        p.setPen(pen); p.drawEllipse(spinnerRect)
        # arc
        pen = QPen(QColor(0,0,0), 4)
        p.setPen(pen)
        p.drawArc(spinnerRect, int(-self._angle*16), 120*16)
        # text
        self._text.adjustSize()
        self._text.move(center.x() - self._text.width()//2, spinnerRect.bottom() + 8)
