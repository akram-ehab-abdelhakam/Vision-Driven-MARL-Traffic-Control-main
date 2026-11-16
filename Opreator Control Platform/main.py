import sys, random, threading, time as _time, json
from datetime import datetime, timedelta

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QPushButton, QFrame, QTextEdit,
    QTableWidget, QTableWidgetItem, QHeaderView, QRadioButton,
    QButtonGroup, QGroupBox, QScrollArea, QSizePolicy, QProgressBar,
    QCheckBox, QLineEdit,
)
from PyQt5.QtCore import (
    Qt, QTimer, QRectF, QPointF, pyqtSignal, pyqtSlot, QObject, QThread,
)
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QPen, QBrush,
    QPainterPath, QRadialGradient, QLinearGradient, QPalette,
)
import numpy as np

try:
    import requests as _requests
    HAS_REQ = True
except ImportError:
    HAS_REQ = False
API_URL = "http://127.0.0.1:8000" # FastAPI base URL (no trailing /state)
MODE_RL, MODE_FIXED, MODE_MANUAL = 0, 1, 2
MODE_NAMES = {0: "RL", 1: "FIXED-TIME"}
SCEN_NORMAL, SCEN_EMERG = 0, 2
SCEN_NAMES = {0: "Normal", 2: "Emergency Vehicle"}
TL_IDS = ["TL1", "TL2", "TL3", "TL4"]

WEBSTER = {
    "TL1": {"queue": 101.97, "delay": 78.53, "phases": 10, "edges": "E7, E9"},
    "TL2": {"queue": 25.25, "delay": 66.81, "phases": 9, "edges": "E1, E2, E4, E6"},
    "TL3": {"queue": 12.67, "delay": 35.45, "phases": 6, "edges": "E3, E5"},
    "TL4": {"queue": 1.53, "delay": 25.36, "phases": 4, "edges": "E8, E10"},
}
RL_CONVERGED = {
    "TL1": {"queue": 1.73, "delay": 0.77},
    "TL2": {"queue": 1.43, "delay": 7.40},
    "TL3": {"queue": 0.57, "delay": 2.16},
    "TL4": {"queue": 1.04, "delay": 2.24},
}
THROUGHPUT_WEBSTER = 1995
THROUGHPUT_RL = 3891

EMISSIONS = {
    "co2_rl_mg": 795_641_940,
    "co2_ft_mg": 1_334_797_891,
    "co2_saving_pct": 40.39,
    "nox_rl_mg": 326_167,
    "nox_ft_mg": 596_819,
    "nox_saving_pct": 45.35,
    "fuel_rl_ml": 253_781_084,
    "fuel_ft_ml": 425_768_734,
    "fuel_saving_pct": 40.39,
}
CO2_PER_VEH_RL = EMISSIONS["co2_rl_mg"] / THROUGHPUT_RL # ~204,467 mg
CO2_PER_VEH_FT = EMISSIONS["co2_ft_mg"] / THROUGHPUT_WEBSTER # ~669,072 mg
CO2_PER_VEH_SAVING_PCT = (1 - CO2_PER_VEH_RL / CO2_PER_VEH_FT) * 100 # ~69.4%

TL_ROLES = {"TL1": "Heavy Corridor", "TL2": "Standard Junction",
            "TL3": "Edge Node", "TL4": "Low Volume"}
TL_AGENT_CLR = {"TL1": "#3b82f6", "TL2": "#22c55e",
                "TL3": "#eab308", "TL4": "#a855f7"}

DETECTORS = [
    {"id": "e2_E7_0", "tl": "TL1", "edge": "E7"},
    {"id": "e2_E7_1", "tl": "TL1", "edge": "E7"},
    {"id": "e2_E9_0", "tl": "TL1", "edge": "E9"},
    {"id": "e2_E9_1", "tl": "TL1", "edge": "E9"},
    {"id": "e2_E1_0", "tl": "TL2", "edge": "E1"},
    {"id": "e2_E2_0", "tl": "TL2", "edge": "E2"},
    {"id": "e2_E4_0", "tl": "TL2", "edge": "E4"},
    {"id": "e2_E6_0", "tl": "TL2", "edge": "E6"},
    {"id": "e2_E3_0", "tl": "TL3", "edge": "E3"},
    {"id": "e2_E3_1", "tl": "TL3", "edge": "E3"},
    {"id": "e2_E5_0", "tl": "TL3", "edge": "E5"},
    {"id": "e2_E5_1", "tl": "TL3", "edge": "E5"},
    {"id": "e2_E8_0", "tl": "TL4", "edge": "E8"},
    {"id": "e2_E8_1", "tl": "TL4", "edge": "E8"},
    {"id": "e2_E10_0", "tl": "TL4", "edge": "E10"},
    {"id": "e2_E10_1", "tl": "TL4", "edge": "E10"},
]

LOG_COLORS = {
    "OVERRIDE": "#f97316", "MODE": "#eab308",
    "SCENARIO": "#a855f7", "AGENT": "#64748b", "SYSTEM": "#3b82f6",
}
C = dict(
    bg="#0f1420", panel="#161d2e", card="#1e293b", border="#2d3a4d",
    accent="#5b9cf6", green="#34d399", yellow="#facc15", red="#f87171",
    orange="#fb923c", text="#d4dce8", muted="#8898ae", dark="#64748b",
    purple="#a78bfa",
)
PHASE_COLORS = [C["green"], C["red"], C["yellow"]]
PHASE_LABELS = ["GREEN", "RED", "YELLOW"]
STYLESHEET = f"""
QMainWindow, #central {{ background:{C['bg']}; }}
QWidget {{ color:{C['text']}; font-family:'Segoe UI',Arial,sans-serif; font-size:13px; }}
QTabWidget::pane {{ border:1px solid {C['border']}; background:{C['bg']}; border-radius:4px; }}
QTabBar::tab {{
    background:{C['panel']}; color:{C['muted']}; padding:11px 26px;
    border:1px solid {C['border']}; border-bottom:none;
    border-top-left-radius:8px; border-top-right-radius:8px;
    margin-right:3px; font-size:13px; font-weight:600;
}}
QTabBar::tab:selected {{
    background:{C['card']}; color:{C['accent']};
    border-bottom:2px solid {C['accent']};
}}
QTabBar::tab:hover {{ background:{C['card']}; color:{C['text']}; }}
QLabel {{ background:transparent; }}
QPushButton {{
    background:{C['accent']}; color:white; border:none; padding:8px 16px;
    border-radius:6px; font-weight:600; font-size:12px;
}}
QPushButton:hover {{ background:#4a8ae6; }}
QPushButton:pressed {{ background:#3a7ad6; }}
QTextEdit {{
    background:#131a2a; color:#a7f3d0; border:1px solid {C['border']};
    border-radius:8px; padding:10px;
    font-family:'Consolas','Courier New',monospace; font-size:12px;
    selection-background-color:#2a4a7a;
}}
QTableWidget {{
    background:{C['panel']}; color:{C['text']}; gridline-color:#252f40;
    border:1px solid {C['border']}; border-radius:6px; font-size:12px;
    alternate-background-color:#1a2435;
}}
QTableWidget::item {{ padding:6px 8px; }}
QTableWidget::item:selected {{ background:#2a4a7a; }}
QHeaderView::section {{
    background:{C['card']}; color:{C['accent']}; padding:8px 6px;
    border:none; border-bottom:2px solid {C['border']};
    font-weight:700; font-size:11px;
}}
QRadioButton {{ spacing:10px; font-size:13px; padding:3px 0; }}
QRadioButton::indicator {{ width:18px; height:18px; border-radius:9px; }}
QRadioButton::indicator:checked {{ background:{C['accent']}; border:2px solid #7ab4fa; }}
QRadioButton::indicator:unchecked {{ background:{C['border']}; border:2px solid #475569; }}
QGroupBox {{
    background:{C['card']}; border:1px solid {C['border']}; border-radius:10px;
    margin-top:16px; padding:22px 14px 14px 14px; font-weight:700;
}}
QGroupBox::title {{
    subcontrol-origin:margin; left:14px; padding:0 8px;
    color:{C['muted']};
}}
QScrollArea {{ border:none; background:transparent; }}
QScrollBar:vertical {{
    background:{C['panel']}; width:8px; border:none; border-radius:4px;
    margin:2px 0;
}}
QScrollBar::handle:vertical {{
    background:{C['border']}; border-radius:4px; min-height:30px;
}}
QScrollBar::handle:vertical:hover {{ background:#475569; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QScrollBar:horizontal {{
    background:{C['panel']}; height:8px; border:none; border-radius:4px;
    margin:0 2px;
}}
QScrollBar::handle:horizontal {{
    background:{C['border']}; border-radius:4px; min-width:30px;
}}
QScrollBar::handle:horizontal:hover {{ background:#475569; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width:0; }}
QCheckBox {{ spacing:8px; font-size:12px; padding:2px 0; }}
QCheckBox::indicator {{ width:16px; height:16px; border-radius:4px; border:1px solid {C['border']}; }}
QCheckBox::indicator:checked {{ background:{C['accent']}; border:1px solid {C['accent']}; }}
QCheckBox::indicator:unchecked:hover {{ border:1px solid {C['accent']}; }}
QLineEdit {{
    background:#131a2a; color:{C['text']}; border:1px solid {C['border']};
    border-radius:6px; padding:8px 12px; font-size:12px;
    selection-background-color:#2a4a7a;
}}
QLineEdit:focus {{ border:1px solid {C['accent']}; }}
QToolTip {{
    background:{C['card']}; color:{C['text']}; border:1px solid {C['border']};
    border-radius:4px; padding:6px 10px; font-size:11px;
}}
"""
_BTN = "border:none;border-radius:6px;font-weight:600;font-size:12px;padding:8px 16px;"
BTN_DANGER = f"background:{C['red']};color:white;{_BTN}"
BTN_SEC = f"background:{C['border']};color:{C['text']};{_BTN}"
BTN_GREEN = f"background:{C['green']};color:white;{_BTN}"
def _lbl(text, color=None, size=None, bold=False, w=None, wrap=False):
    l = QLabel(str(text))
    parts = []
    if color: parts.append(f"color:{color}")
    if size: parts.append(f"font-size:{size}px")
    if bold: parts.append("font-weight:700")
    if parts: l.setStyleSheet(";".join(parts))
    if w: l.setFixedWidth(w)
    if wrap: l.setWordWrap(True)
    return l
class EventBus(QObject):
    logged = pyqtSignal(str, str)
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def log(self, msg, tag="SYSTEM"):
        ts = datetime.now().strftime("%H:%M:%S")
        self.logged.emit(f"[{ts}] {msg}", tag)


class BackendWorker(QObject):
    state_ready = pyqtSignal(dict)
    poll_error = pyqtSignal(str)
    checkpoint_finished = pyqtSignal(bool, str)
    # Fired when /episode/{ep} fetch completes (read-only, never restarts sim)
    episode_metrics_ready = pyqtSignal(bool, str, int, dict)

    def __init__(self, base_url):
        super().__init__()
        self._base_url = base_url.rstrip("/")
        self._session = _requests.Session() if HAS_REQ else None
        self._poll_lock = threading.Lock()
        self._checkpoint_lock = threading.Lock()

    @pyqtSlot(str)
    def set_base_url(self, url):
        url = (url or "").strip().rstrip("/")
        if url:
            self._base_url = url

    @pyqtSlot()
    def poll_state(self):
        if self._session is None or not self._poll_lock.acquire(False):
            return
        try:
            resp = self._session.get(
                f"{self._base_url}/state",
                timeout=(1.0, 1.5),
            )
            resp.raise_for_status()
            self.state_ready.emit(resp.json())
        except Exception as exc:
            self.poll_error.emit(type(exc).__name__)
        finally:
            self._poll_lock.release()

    @pyqtSlot(int)
    def load_checkpoint(self, episode):
        if self._session is None:
            self.checkpoint_finished.emit(
                False, "requests library not installed"
            )
            return
        if not self._checkpoint_lock.acquire(False):
            self.checkpoint_finished.emit(
                False, "Checkpoint load already in progress"
            )
            return
        try:
            resp = self._session.post(
                f"{self._base_url}/load_checkpoint",
                json={"episode": episode},
                timeout=(2.0, 15.0),
            )
            resp.raise_for_status()
            data = resp.json()
            self.checkpoint_finished.emit(
                True,
                data.get("message") or f"Checkpoint ep={episode} loaded",
            )
        except Exception as exc:
            self.checkpoint_finished.emit(
                False, f"{type(exc).__name__}: {exc}"
            )
        finally:
            self._checkpoint_lock.release()

    @pyqtSlot(int)
    def fetch_episode_metrics(self, episode):
        """Read-only fetch of /episode/{ep} - does NOT restart the simulation."""
        if self._session is None:
            self.episode_metrics_ready.emit(False, "requests library not installed", episode, {})
            return
        try:
            resp = self._session.get(
                f"{self._base_url}/episode/{episode}",
                timeout=(2.0, 8.0),
            )
            resp.raise_for_status()
            data = resp.json()
            metrics = data.get("metrics") or {}
            self.episode_metrics_ready.emit(True, f"Episode {episode} loaded", episode, metrics)
        except Exception as exc:
            self.episode_metrics_ready.emit(False, f"{type(exc).__name__}: {exc}", episode, {})

class SimState:
    def __init__(self):
        self.mock_mode = False
        self.manual_mock = False
        self.backend_connected = False
        self.backend_error = ""
        self.api_url = API_URL.rstrip("/")
        self.warmed_up = False
        self.mode = MODE_RL
        self.scenario = SCEN_NORMAL
        self.session_start = _time.time()
        self.vehicles_cleared = 0
        self._veh_acc = 0.0
        self._delay_sum = 0.0
        self._delay_n = 0
        self.tick = 0
        self.phase_history = []
        self.tl = {}
        for t in TL_IDS:
            self.tl[t] = dict(
                phase=random.randint(0, 2),
                timer=random.randint(12, 30),
                queue=RL_CONVERGED[t]["queue"],
                delay=RL_CONVERGED[t]["delay"],
                override=None,
                emergency=False,
                phase_dur=0,
                speed=50.0,
                reward=0.0,
                intent_mag=0.0,
                last_switch=datetime.now().strftime("%H:%M:%S"),
                last_decision="Initialized awaiting first phase cycle",
            )
        self.throughput = float(THROUGHPUT_RL)
        self.co2_session = 0.0

    def session_elapsed(self):
        return str(timedelta(seconds=int(_time.time() - self.session_start)))

    def session_avg_delay(self):
        return self._delay_sum / self._delay_n if self._delay_n else 0.0

    def overridden_ids(self):
        return [t for t in TL_IDS if self.tl[t]["override"]]

    def mock_detector(self, det):
        q = self.tl[det["tl"]]["queue"]
        mx = max(WEBSTER[det["tl"]]["queue"], 1)
        r = q / mx
        cnt = max(0, int(q * 0.15 + random.gauss(0, 1.5)))
        occ = round(min(100.0, r * 80 + random.gauss(0, 4)), 1)
        spd = round(max(5.0, 50 - r * 35 + random.gauss(0, 2.5)), 1)
        return cnt, occ, spd

    def step(self):
        self.tick += 1
        scen_mult = 1.0

        for t in TL_IDS:
            s = self.tl[t]
            s["phase_dur"] += 1
            s["timer"] -= 1

            if s["timer"] <= 0:
                if s["override"]:
                    s["timer"] = 30
                else:
                    old = s["phase"]
                    s["phase"] = (old + 1) % 3
                    s["timer"] = random.randint(3, 5) if s["phase"] == 2 else random.randint(15, 35)
                    if s["phase"] != old:
                        dur_str = f"{s['phase_dur']}s"
                        ev = f"{PHASE_LABELS[old]} -> {PHASE_LABELS[s['phase']]}"
                        s["last_switch"] = datetime.now().strftime("%H:%M:%S")
                        s["phase_dur"] = 0
                        if s["phase"] == 0:
                            s["last_decision"] = (
                                f"Back-pressure on {WEBSTER[t]['edges']} approach "
                                "exceeded threshold clearing primary corridor"
                            )
                        else:
                            s["last_decision"] = (
                                "Green time expired switching to serve cross traffic"
                            )
                        self.phase_history.append(dict(
                            time=datetime.now().strftime("%H:%M:%S"),
                            tl=t, event=ev, dur=dur_str,
                            queue=f"{s['queue']:.0f}v",
                        ))
                        if len(self.phase_history) > 20:
                            self.phase_history.pop(0)
                        EventBus.get().log(
                            f"{t} -> Phase {PHASE_LABELS[s['phase']]}", "AGENT"
                        )
            if self.mode == MODE_RL:
                bq, bd = RL_CONVERGED[t]["queue"], RL_CONVERGED[t]["delay"]
            else:
                bq, bd = WEBSTER[t]["queue"], WEBSTER[t]["delay"]
            s["queue"] = max(0.0, bq * scen_mult + random.gauss(0, bq * 0.05))
            s["delay"] = max(0.0, bd * scen_mult + random.gauss(0, bd * 0.03))
        base_tp = THROUGHPUT_RL if self.mode == MODE_RL else THROUGHPUT_WEBSTER
        self.throughput = base_tp * scen_mult + random.gauss(0, 30)
        self._veh_acc += self.throughput / 3600.0
        cleared = int(self._veh_acc)
        self._veh_acc -= cleared
        self.vehicles_cleared += cleared
        avg_d = np.mean([self.tl[t]["delay"] for t in TL_IDS])
        self._delay_sum += avg_d
        self._delay_n += 1
        if self.mode == MODE_RL:
            self.co2_session += EMISSIONS["co2_rl_mg"] / 3600.0
        else:
            self.co2_session += EMISSIONS["co2_ft_mg"] / 3600.0
class KPICard(QFrame):
    def __init__(self, title, value, unit, subtitle="", color=C["accent"],
                 parent=None):
        super().__init__(parent)
        self.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            "border-radius:12px;"
        )
        self.setMinimumWidth(165)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 12)
        lay.setSpacing(3)
        lay.addWidget(_lbl(title, C["muted"], 10, bold=True))
        self.val = _lbl(str(value), color, 24, bold=True)
        lay.addWidget(self.val)
        lay.addWidget(_lbl(unit, C["dark"], 10))
        lay.addSpacing(2)
        self.sub = _lbl(subtitle, C["green"], 10)
        lay.addWidget(self.sub)
        lay.addSpacing(4)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(50)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(3)
        self.bar.setStyleSheet(
            f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
            f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
        )
        lay.addWidget(self.bar)

    def set(self, value, subtitle=None):
        self.val.setText(str(value))
        if subtitle is not None:
            self.sub.setText(subtitle)

    def set_bar(self, pct, color=None):
        self.bar.setValue(max(0, min(100, int(pct))))
        if color:
            self.bar.setStyleSheet(
                f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
                f"QProgressBar::chunk{{background:{color};border-radius:2px;}}"
            )


class ModeBanner(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 18, 0)
        self.lbl = QLabel()
        self.lbl.setStyleSheet("font-size:13px;font-weight:700;")
        lay.addWidget(self.lbl)
        self.set_mode(MODE_RL)

    def set_mode(self, m):
        cfg = {
            MODE_RL: (C["accent"], "#172a45",
                      "RL MODE: ACTIVE", "All agents autonomous"),
            MODE_FIXED: (C["yellow"], "#2e2a10",
                         "FIXED-TIME MODE", "Webster baseline active"),
        }.get(m, (C["accent"], "#172a45", "RL MODE: ACTIVE", "All agents autonomous"))
        self.setStyleSheet(
            f"background:{cfg[1]};border:1px solid {cfg[0]};border-radius:8px;"
        )
        self.lbl.setText(
            f"<span style='color:{cfg[0]};'>{cfg[2]}</span>"
            f"<span style='color:{C['muted']};'> {cfg[3]}</span>"
        )


class ConnectionBar(QFrame):
    """Persistent 28px status strip always visible."""

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setFixedHeight(38)
        self.setStyleSheet(
            f"background:#111827;border-bottom:1px solid {C['border']};"
        )
        lay = QHBoxLayout(self)
        lay.setContentsMargins(18, 0, 18, 0)
        lay.setSpacing(16)

        self.sumo_lbl = _lbl("", C["muted"], 10)
        self.traci_lbl = _lbl("", C["muted"], 10)
        self.model_lbl = _lbl("", C["muted"], 10)
        for w in (self.sumo_lbl, self.traci_lbl, self.model_lbl):
            lay.addWidget(w)
        lay.addWidget(_lbl("|", C["border"], 10))
        self.tick_l = _lbl("TICK: 0", C["yellow"], 18, bold=True)
        self.tick_l.setMinimumWidth(130)
        self.tick_l.setAlignment(Qt.AlignCenter)
        self.tick_l.setStyleSheet(
            f"color:{C['yellow']};font-size:18px;font-weight:800;"
            f"background:#1f2937;border:1px solid {C['yellow']};"
            "border-radius:6px;padding:3px 10px;"
        )
        lay.addWidget(self.tick_l)
        lay.addWidget(_lbl("|", C["border"], 10))
        self.mode_l = _lbl("MODE: RL", C["accent"], 10, bold=True)
        self.scen_l = _lbl("SCENARIO: Normal", C["text"], 10)
        self.sess_l = _lbl("SESSION: 00:00:00", C["muted"], 10)
        lay.addWidget(self.mode_l)
        lay.addWidget(self.scen_l)
        lay.addWidget(self.sess_l)
        lay.addStretch()

    def refresh(self):
        s = self.state
        if s.manual_mock:
            dot = f"<span style='color:{C['yellow']}'></span>"
            self.sumo_lbl.setText(f"{dot} SUMO: Simulated")
            self.traci_lbl.setText(f"{dot} TraCI: Mock")
            self.model_lbl.setText(f"{dot} Model: Mock Inference")
        elif s.backend_connected:
            if s.warmed_up:
                self.show_connected()
            else:
                self.show_warming_up()
        elif s.mock_mode:
            self.show_disconnected()
        self.mode_l.setText(f"MODE: {MODE_NAMES[s.mode]}")
        self.scen_l.setText(
            f"SCENARIO: {SCEN_NAMES.get(s.scenario, 'Unknown')}"
        )
        self.sess_l.setText(f"SESSION: {s.session_elapsed()}")
        self.tick_l.setText(f"TICK: {int(getattr(s, 'tick', 0))}")

    def show_warming_up(self, step=None, total=None):
        dot = f"<span style='color:{C['yellow']}'></span>"
        extra = ""
        if step is not None and total:
            extra = f" ({step}/{total})"
        self.sumo_lbl.setText(f"{dot} SUMO: Warming up{extra}")
        self.traci_lbl.setText(f"{dot} TraCI: Port 8813 (silent)")
        self.model_lbl.setText(f"{dot} Model: Loaded warming")

    def show_connected(self):
        dot = f"<span style='color:{C['green']}'></span>"
        self.sumo_lbl.setText(f"{dot} SUMO: Running")
        self.traci_lbl.setText(f"{dot} TraCI: Port 8813")
        self.model_lbl.setText(f"{dot} Model: Loaded (ONNX)")

    def show_disconnected(self):
        dot = f"<span style='color:{C['red']}'></span>"
        self.sumo_lbl.setText(f"{dot} Backend Disconnected")
        self.traci_lbl.setText(f"{dot} Status: Mock Mode")
        self.model_lbl.setText(f"{dot} Model: Internal Simulation")
class Sparkline(QWidget):
    def __init__(self, color=C["accent"], parent=None):
        super().__init__(parent)
        self._c = QColor(color)
        self._data = []
        self.setFixedHeight(65)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def push(self, v):
        self._data.append(float(v))
        self.update()

    def paintEvent(self, _):
        if len(self._data) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        d = self._data
        mx = max(max(d), 1)
        def px(i): return int(i / (len(d) - 1) * (w - 2)) + 1
        def py(v): return int(h - 4 - (v / mx) * (h - 10))
        path = QPainterPath()
        path.moveTo(px(0), py(d[0]))
        for i, v in enumerate(d[1:], 1):
            path.lineTo(px(i), py(v))
        path.lineTo(px(len(d) - 1), h)
        path.lineTo(px(0), h)
        path.closeSubpath()
        g = QLinearGradient(0, 0, 0, h)
        c1 = QColor(self._c); c1.setAlpha(55)
        c2 = QColor(self._c); c2.setAlpha(5)
        g.setColorAt(0, c1); g.setColorAt(1, c2)
        p.fillPath(path, g)
        lp = QPainterPath()
        lp.moveTo(px(0), py(d[0]))
        for i, v in enumerate(d[1:], 1):
            lp.lineTo(px(i), py(v))
        pen = QPen(self._c, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawPath(lp)
        lx, ly = px(len(d) - 1), py(d[-1])
        p.setBrush(QBrush(self._c))
        p.setPen(Qt.NoPen)
        p.drawEllipse(lx - 3, ly - 3, 6, 6)
        p.end()


class TrendLine(QWidget):
    def __init__(self, color=C["green"], label="", parent=None):
        super().__init__(parent)
        self._c = QColor(color)
        self._label = label
        self._data = []
        self.setFixedHeight(80)

    def push(self, v):
        self._data.append(float(v))
        self.update()

    def paintEvent(self, _):
        if len(self._data) < 2:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        d = self._data
        mn, mx = min(d), max(d)
        rng = max(mx - mn, 1)
        def px(i): return int(i / (len(d) - 1) * (w - 4)) + 2
        def py(v): return int(h - 6 - ((v - mn) / rng) * (h - 14))
        p.fillRect(self.rect(), QColor(C["panel"]))
        path = QPainterPath()
        path.moveTo(px(0), py(d[0]))
        for i, v in enumerate(d[1:], 1):
            path.lineTo(px(i), py(v))
        pen = QPen(self._c, 1.5)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawPath(path)
        p.setPen(QColor(self._c))
        p.setFont(QFont("Segoe UI", 10, QFont.Bold))
        p.drawText(6, 16, f"{self._label}: {d[-1]:.0f}")
        p.end()
class TopologyWidget(QWidget):
    intersection_clicked = pyqtSignal(str)
    POS = {
        "TL1": (0.25, 0.28), "TL4": (0.75, 0.28),
        "TL2": (0.25, 0.72), "TL3": (0.75, 0.72),
    }
    ROADS = [("TL1", "TL4"), ("TL2", "TL3"), ("TL1", "TL2"), ("TL4", "TL3")]

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumSize(500, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _cx_cy(self, tl):
        rx, ry = self.POS[tl]
        return rx * self.width(), ry * self.height()

    def _rect(self, tl, sz=62):
        cx, cy = self._cx_cy(tl)
        return QRectF(cx - sz / 2, cy - sz / 2, sz, sz)

    def mousePressEvent(self, ev):
        for t in TL_IDS:
            r = self._rect(t).adjusted(-10, -10, 10, 10)
            if r.contains(ev.pos().x(), ev.pos().y()):
                self.intersection_clicked.emit(t)
                break

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(C["panel"]))

        p.setPen(QColor(C["muted"]))
        p.setFont(QFont("Segoe UI", 11, QFont.Bold))
        p.drawText(14, 24, "OM Network Assiut, Egypt")
        rp = QPen(QColor("#253044"), 22)
        rp.setCapStyle(Qt.RoundCap)
        p.setPen(rp)
        for a, b in self.ROADS:
            ax, ay = self._cx_cy(a)
            bx, by = self._cx_cy(b)
            p.drawLine(int(ax), int(ay), int(bx), int(by))
        p.setPen(QPen(QColor(C["yellow"]), 1, Qt.DashLine))
        for a, b in self.ROADS:
            ax, ay = self._cx_cy(a)
            bx, by = self._cx_cy(b)
            p.drawLine(int(ax), int(ay), int(bx), int(by))
        stub = QPen(QColor("#253044"), 18)
        stub.setCapStyle(Qt.RoundCap)
        p.setPen(stub)
        for tl, dirs in [
            ("TL1", [(-1, 0), (0, -1)]), ("TL4", [(1, 0), (0, -1)]),
            ("TL2", [(-1, 0), (0, 1)]), ("TL3", [(1, 0), (0, 1)]),
        ]:
            cx, cy = self._cx_cy(tl)
            for dx, dy in dirs:
                p.drawLine(int(cx), int(cy),
                           int(cx + dx * 100), int(cy + dy * 75))
        p.setPen(QColor(C["muted"]))
        p.setFont(QFont("Segoe UI", 10))
        x1, y1 = self._cx_cy("TL1")
        x4, y4 = self._cx_cy("TL4")
        x2, y2 = self._cx_cy("TL2")
        x3, y3 = self._cx_cy("TL3")
        p.drawText(int((x1 + x4) / 2 - 50), int(y1 - 28), "Bridge Corridor 1")
        p.drawText(int((x2 + x3) / 2 - 50), int(y2 + 38), "Bridge Corridor 2")
        for t in TL_IDS:
            rect = self._rect(t)
            cx, cy = self._cx_cy(t)
            tl = self.state.tl[t]
            pc = QColor(PHASE_COLORS[tl["phase"]])
            qr = tl["queue"] / max(WEBSTER[t]["queue"], 1)
            gc = QColor(
                C["red"] if qr > 0.7
                else C["yellow"] if qr > 0.4
                else C["green"]
            )
            gc.setAlpha(40)
            gr = QRadialGradient(cx, cy, 60)
            gr.setColorAt(0, gc)
            gr.setColorAt(1, QColor(0, 0, 0, 0))
            p.setBrush(QBrush(gr))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QRectF(cx - 60, cy - 60, 120, 120))

            p.setBrush(QColor("#1a2536"))
            p.setPen(QPen(pc, 2.5))
            p.drawRoundedRect(rect, 10, 10)

            p.setPen(QColor(C["text"]))
            p.setFont(QFont("Segoe UI", 13, QFont.Bold))
            p.drawText(rect, Qt.AlignHCenter | Qt.AlignTop, f"\n{t}")

            p.setPen(QColor(C["text"]))
            p.setFont(QFont("Consolas", 11, QFont.Bold))
            p.drawText(int(cx - 14), int(cy + rect.height() / 2 + 20),
                       f"{tl['timer']}s")

            bar_x = cx + rect.width() / 2 + 10
            bar_h = min(50, tl["queue"] / 3.5)
            bc = (C["red"] if qr > 0.7
                  else C["yellow"] if qr > 0.4
                  else C["green"])
            p.setBrush(QColor(bc))
            p.setPen(Qt.NoPen)
            p.drawRoundedRect(
                QRectF(bar_x, cy - bar_h / 2, 8, max(bar_h, 3)), 4, 4
            )
            p.setPen(QColor(C["muted"]))
            p.setFont(QFont("Segoe UI", 10))
            p.drawText(int(bar_x + 14), int(cy + 5), f"Q:{tl['queue']:.0f}")

            if tl["override"]:
                p.setPen(QColor(C["orange"]))
                p.setFont(QFont("Segoe UI", 10, QFont.Bold))
                p.drawText(int(cx - 32), int(cy - rect.height() / 2 - 8),
                           "OVERRIDE")
        p.end()
class DashboardTab(QScrollArea):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWidgetResizable(True)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setSpacing(12)
        lay.setContentsMargins(16, 14, 16, 14)

        self.banner = ModeBanner()
        lay.addWidget(self.banner)
        kr = QHBoxLayout()
        kr.setSpacing(10)
        self.kpi_tick = KPICard("SIM TICK", "0", "current step",
                                "live simulation clock", C["yellow"])
        self.kpi_delay = KPICard("AVG. DELAY", "", "seconds / vehicle",
                                  "", C["accent"])
        self.kpi_reward = KPICard("AVG. REWARD", "", "per agent / tick",
                                   "", C["green"])
        self.kpi_queue = KPICard("AVG. QUEUE", "", "vehicles waiting",
                                  "", C["orange"])
        self.kpi_loss = KPICard("ACTOR LOSS", "", "last tick",
                                 "", C["purple"])
        self.kpi_improvement = KPICard("IMPROVEMENT", "",
                                        "wait-time vs Webster", "", C["green"])
        for k in (self.kpi_tick, self.kpi_delay, self.kpi_reward,
                  self.kpi_queue, self.kpi_loss, self.kpi_improvement):
            kr.addWidget(k)
        lay.addLayout(kr)
        lay.addSpacing(4)

        # -- Per-Agent Queue & Reward panel -----------------------------------------------------
        lay.addWidget(_lbl("Per-Agent Queue & Reward", C["text"], 15, bold=True))
        per_agent_panel = QFrame()
        per_agent_panel.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            "border-radius:10px;"
        )
        pa_grid = QGridLayout(per_agent_panel)
        pa_grid.setContentsMargins(14, 12, 14, 12)
        pa_grid.setSpacing(10)
        self._per_tl_cards = {}
        for idx2, tid in enumerate(["TL1", "TL2", "TL3", "TL4"]):
            col = TL_AGENT_CLR[tid]
            cell = QFrame()
            cell.setStyleSheet(
                f"background:{C['panel']};border:1px solid {col};"
                "border-radius:8px;"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(14, 10, 14, 10)
            cl.setSpacing(5)
            hrow = QHBoxLayout()
            hrow.addWidget(_lbl(tid, col, 14, bold=True))
            hrow.addStretch()
            hrow.addWidget(_lbl(TL_ROLES[tid], C["muted"], 9))
            cl.addLayout(hrow)
            sep = QFrame(); sep.setFixedHeight(1)
            sep.setStyleSheet(f"background:{C['border']};")
            cl.addWidget(sep)
            qrow = QHBoxLayout()
            qrow.addWidget(_lbl("Queue", C["muted"], 10))
            qrow.addStretch()
            q_val = _lbl("", C["orange"], 20, bold=True)
            q_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            qrow.addWidget(q_val)
            qrow.addWidget(_lbl("veh", C["dark"], 9))
            cl.addLayout(qrow)
            q_bar = QProgressBar()
            q_bar.setRange(0, 100); q_bar.setValue(0)
            q_bar.setTextVisible(False); q_bar.setFixedHeight(4)
            q_bar.setStyleSheet(
                f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
                f"QProgressBar::chunk{{background:{C['orange']};border-radius:2px;}}"
            )
            cl.addWidget(q_bar)
            rrow = QHBoxLayout()
            rrow.addWidget(_lbl("Reward", C["muted"], 10))
            rrow.addStretch()
            r_val = _lbl("", C["green"], 20, bold=True)
            r_val.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            rrow.addWidget(r_val)
            rrow.addWidget(_lbl("/tick", C["dark"], 9))
            cl.addLayout(rrow)
            r_bar = QProgressBar()
            r_bar.setRange(0, 100); r_bar.setValue(0)
            r_bar.setTextVisible(False); r_bar.setFixedHeight(4)
            r_bar.setStyleSheet(
                f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
                f"QProgressBar::chunk{{background:{C['green']};border-radius:2px;}}"
            )
            cl.addWidget(r_bar)
            src_lbl = _lbl("", C["dark"], 9)
            cl.addWidget(src_lbl)
            pa_grid.addWidget(cell, 0, idx2)
            self._per_tl_cards[tid] = dict(
                q_val=q_val, q_bar=q_bar,
                r_val=r_val, r_bar=r_bar,
                src_lbl=src_lbl,
            )
        lay.addWidget(per_agent_panel)
        lay.addSpacing(4)

        lay.addWidget(_lbl("OM Network Live Topology", C["text"], 15, bold=True))
        self.topo = TopologyWidget(state)
        self.topo.setMinimumHeight(420)
        self.topo.setFixedHeight(420)
        lay.addWidget(self.topo)
        lay.addSpacing(4)
        lay.addWidget(_lbl("Queue Trends — Full Run History", C["text"], 15, bold=True))
        spark_panel = QFrame()
        spark_panel.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            "border-radius:10px;"
        )
        sg = QGridLayout(spark_panel)
        sg.setContentsMargins(14, 12, 14, 12)
        sg.setSpacing(10)
        self._sparks = {}
        self._spark_lbl = {}
        for idx, (tid, col) in enumerate([
            ("TL1", TL_AGENT_CLR["TL1"]),
            ("TL2", TL_AGENT_CLR["TL2"]),
            ("TL3", TL_AGENT_CLR["TL3"]),
            ("TL4", TL_AGENT_CLR["TL4"]),
        ]):
            cell = QFrame()
            cell.setStyleSheet(
                f"background:{C['panel']};border:1px solid {C['border']};"
                "border-radius:6px;"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(10, 6, 10, 6)
            cl.setSpacing(3)
            header = QHBoxLayout()
            header.addWidget(_lbl(tid, col, 11, bold=True))
            header.addStretch()
            ql = _lbl("0v", C["text"], 11)
            header.addWidget(ql)
            self._spark_lbl[tid] = ql
            cl.addLayout(header)
            sk = Sparkline(col)
            sk.setFixedHeight(65)
            self._sparks[tid] = sk
            cl.addWidget(sk)
            sg.addWidget(cell, idx // 2, idx % 2)
        lay.addWidget(spark_panel)
        lay.addSpacing(4)
        lay.addWidget(_lbl("Intersection Status Live", C["text"], 15, bold=True))
        self.strip = QTableWidget(4, 5)
        self.strip.setHorizontalHeaderLabels(
            ["Intersection", "Queue", "Phase", "Time Left", "Status"]
        )
        self.strip.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.strip.horizontalHeader().setFixedHeight(42)
        self.strip.horizontalHeader().setStyleSheet(
            f"QHeaderView::section{{"
            f"background:{C['card']};color:{C['accent']};"
            f"border:none;border-bottom:2px solid {C['border']};"
            f"font-weight:700;font-size:14px;padding:10px 6px;}}"
        )
        self.strip.verticalHeader().setVisible(False)
        self.strip.setEditTriggers(QTableWidget.NoEditTriggers)
        self.strip.setAlternatingRowColors(True)
        self.strip.setStyleSheet(
            f"QTableWidget{{"
            f"background:{C['panel']};color:{C['text']};"
            f"gridline-color:#252f40;border:1px solid {C['border']};"
            f"border-radius:8px;font-size:14px;font-weight:600;"
            f"alternate-background-color:#1a2435;}}"
            f"QTableWidget::item{{padding:12px 14px;}}"
            f"QTableWidget::item:selected{{background:#2a4a7a;}}"
        )
        self.strip.verticalHeader().setDefaultSectionSize(54)
        self.strip.setFixedHeight(262)
        for i in range(4):
            for j in range(5):
                self.strip.setItem(i, j, QTableWidgetItem())
                self.strip.item(i, j).setTextAlignment(Qt.AlignCenter)
        lay.addWidget(self.strip)

        lay.addStretch()
        self.setWidget(box)

    def refresh(self):
        s = self.state
        be = getattr(self.window(), "_backend", {}) or {}
        live = be.get("live_metrics", {}) or {}
        kpi = be.get("kpi", {}) or {}

        self.kpi_tick.set(str(int(s.tick)), "live simulation clock")
        self.kpi_tick.set_bar(int(s.tick) % 100, C["yellow"])

        avg_d_val = kpi.get("avg_delay", live.get("avg_delay"))
        w_d = float(np.mean([WEBSTER[t]["delay"] for t in TL_IDS]))
        if avg_d_val not in (None, ""):
            avg_d = float(avg_d_val)
            pct_d = (w_d - avg_d) / max(w_d, 1) * 100
            arrow = "\u2193" if pct_d > 0 else "\u2191"
            self.kpi_delay.set(f"{avg_d:.1f}", f"{arrow} {abs(pct_d):.0f}% vs Webster (live)")
            bar_color = C["green"] if pct_d > 20 else C["yellow"] if pct_d > 0 else C["red"]
            self.kpi_delay.set_bar(min(100, abs(pct_d) * 1.5), bar_color)
        else:
            self.kpi_delay.set("\u2014", "waiting for live data")
            self.kpi_delay.set_bar(0, C["muted"])

        avg_r_val = kpi.get("avg_reward", live.get("avg_reward"))
        if avg_r_val not in (None, ""):
            avg_r = float(avg_r_val)
            r_color = C["green"] if avg_r >= 0 else C["red"]
            arrow_r = "\u2191" if avg_r >= 0 else "\u2193"
            self.kpi_reward.set(f"{avg_r:.3f}", f"{arrow_r} live reward")
            self.kpi_reward.set_bar(min(100, abs(avg_r) * 20), r_color)
        else:
            self.kpi_reward.set("\u2014", "waiting for live data")
            self.kpi_reward.set_bar(0, C["muted"])

        avg_q_val = kpi.get("avg_queue", live.get("avg_queue"))
        if avg_q_val not in (None, ""):
            avg_q = float(avg_q_val)
            w_q = float(np.mean([WEBSTER[t]["queue"] for t in TL_IDS]))
            pct_q = (w_q - avg_q) / max(w_q, 1) * 100
            arrow_q = "\u2193" if pct_q > 0 else "\u2191"
            q_color = C["green"] if pct_q > 20 else C["yellow"] if pct_q > 0 else C["red"]
            self.kpi_queue.set(f"{avg_q:.1f}", f"{arrow_q} {abs(pct_q):.0f}% vs Webster")
            self.kpi_queue.set_bar(min(100, abs(pct_q) * 1.5), q_color)
        else:
            self.kpi_queue.set("\u2014", "waiting for live data")
            self.kpi_queue.set_bar(0, C["muted"])

        loss_val = kpi.get("avg_loss", live.get("actor_loss"))
        if loss_val not in (None, ""):
            loss = float(loss_val)
            l_color = C["green"] if loss < 0.05 else C["yellow"] if loss < 0.2 else C["red"]
            self.kpi_loss.set(f"{loss:.4f}", "actor loss (live)")
            self.kpi_loss.set_bar(min(100, loss * 500), l_color)
        else:
            self.kpi_loss.set("\u2014", "waiting for live data")
            self.kpi_loss.set_bar(0, C["muted"])

        imp_val = kpi.get("improvement_wait")
        if imp_val not in (None, ""):
            imp = float(imp_val)
            imp_color = C["green"] if imp > 20 else C["yellow"] if imp > 0 else C["red"]
            arrow_i = "\u2193" if imp > 0 else "\u2191"
            self.kpi_improvement.set(f"{imp:.1f}%", f"{arrow_i} wait-time vs Webster")
            self.kpi_improvement.set_bar(min(100, abs(imp) * 1.5), imp_color)
        else:
            self.kpi_improvement.set("\u2014", "waiting for live data")
            self.kpi_improvement.set_bar(0, C["muted"])

        per_tl_queue = live.get("per_tl_queue") or kpi.get("per_tl_queue") or {}
        per_tl_reward = live.get("per_tl_reward") or kpi.get("per_tl_reward") or {}
        for tid in TL_IDS:
            card = self._per_tl_cards[tid]
            w_q = float(WEBSTER[tid]["queue"])
            q_src = per_tl_queue.get(tid) if per_tl_queue.get(tid) not in (None, "") else s.tl[tid]["queue"]
            r_src = per_tl_reward.get(tid) if per_tl_reward.get(tid) not in (None, "") else s.tl[tid].get("reward", 0.0)
            try:
                q_f = float(q_src)
                card["q_val"].setText(f"{q_f:.1f}")
                q_pct = min(100, int(q_f / max(w_q, 1) * 100))
                q_color = C["red"] if q_pct > 70 else C["yellow"] if q_pct > 40 else C["orange"]
                card["q_bar"].setValue(q_pct)
                card["q_bar"].setStyleSheet(
                    f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
                    f"QProgressBar::chunk{{background:{q_color};border-radius:2px;}}"
                )
                card["q_val"].setStyleSheet(f"color:{q_color};font-size:20px;font-weight:700;")
            except (TypeError, ValueError):
                card["q_val"].setText("\u2014")
            try:
                r_f = float(r_src)
                r_color = C["green"] if r_f >= 0 else C["red"]
                card["r_val"].setText(f"{r_f:.3f}")
                card["r_bar"].setValue(min(100, int(abs(r_f) * 10)))
                card["r_bar"].setStyleSheet(
                    f"QProgressBar{{background:#1a2435;border:none;border-radius:2px;}}"
                    f"QProgressBar::chunk{{background:{r_color};border-radius:2px;}}"
                )
                card["r_val"].setStyleSheet(f"color:{r_color};font-size:20px;font-weight:700;")
            except (TypeError, ValueError):
                card["r_val"].setText("\u2014")
            card["src_lbl"].setText("source: live")

        self.banner.set_mode(s.mode)
        self.topo.update()
        for tid in TL_IDS:
            q = s.tl[tid]["queue"]
            self._sparks[tid].push(q)
            self._spark_lbl[tid].setText(f"{q:.0f}v")

        for i, t in enumerate(TL_IDS):
            tl = s.tl[t]
            is_ov = tl["override"]
            status = "Override" if is_ov else "Normal"
            sc = C["orange"] if is_ov else C["green"]
            vals = [
                t,
                f"{tl['queue']:.0f} veh",
                PHASE_LABELS[tl["phase"]],
                f"{tl['timer']}s",
                f"\u2b24 {status}",
            ]
            for j, v in enumerate(vals):
                item = self.strip.item(i, j)
                item.setText(v)
                item.setFont(QFont("Segoe UI", 12, QFont.Bold if j in (0, 2) else QFont.DemiBold))
                if j == 2:
                    item.setForeground(QColor(PHASE_COLORS[tl["phase"]]))
                if j == 4:
                    item.setForeground(QColor(sc))
class AgentsTab(QScrollArea):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWidgetResizable(True)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setSpacing(14)
        lay.setContentsMargins(16, 16, 16, 16)

        lay.addWidget(_lbl(
            "Agent State Inspector MAPPO-PIS Inference Mode",
            C["text"], 17, bold=True,
        ))
        lay.addWidget(_lbl(
            "Displaying live agent observations and decisions. "
            "Training converged at episode 141 \u2014 best result at episode 112.",
            C["muted"], 11,
        ))

        grid = QGridLayout()
        grid.setSpacing(12)
        self.cards = {}
        for i, t in enumerate(TL_IDS):
            card, refs = self._card(t)
            self.cards[t] = refs
            grid.addWidget(card, i // 2, i % 2)
        lay.addLayout(grid)
        gat = QGroupBox("Runtime GAT Coordination")
        gl = QVBoxLayout(gat)
        gl.addWidget(_lbl(
            "Each agent encodes a 32-dimensional intent vector broadcast to "
            "neighbouring agents and aggregated by the Hierarchical Graph "
            "Attention Network (GAT).\n\n"
            "Local Level: GATConv layers with residual normalisation exchange "
            "vectors within each cluster.\n"
            "Global Level: A second GAT layer aggregates cluster-level "
            "representations for system-wide coordination.\n\n"
            "At episode 141 (converged), all four agent pairs show "
            "positive Pearson reward correlation (r = 0.72\u20130.91, "
            "episode 141 converged). Intent "
            "vectors are broadcast every 5 simulation steps "
            "(10 seconds real-time equivalent).",
            C["muted"], 11, wrap=True,
        ))
        lay.addWidget(gat)
        lay.addStretch()
        self.setWidget(box)

    def _card(self, t):
        f = QFrame()
        ac = TL_AGENT_CLR[t]
        f.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            f"border-left:4px solid {ac};border-radius:12px;"
        )
        lay = QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        h = QHBoxLayout()
        h.addWidget(_lbl(
            f"<b style='font-size:14px;color:{ac};'>{t} Agent</b>"
        ))
        badge = _lbl("INFERENCE MODE", C["green"], 10, bold=True)
        badge.setAlignment(Qt.AlignRight)
        h.addWidget(badge)
        lay.addLayout(h)
        for k, v in [
            ("Role", TL_ROLES[t]),
            ("Edges", WEBSTER[t]["edges"]),
            ("Policy", "MAPPO-PIS (ONNX opset 17, CPU)"),
            ("Phases", str(WEBSTER[t]["phases"])),
        ]:
            r = QHBoxLayout()
            r.addWidget(_lbl(k, C["muted"], 10))
            vl = _lbl(v, C["text"], 10)
            vl.setAlignment(Qt.AlignRight)
            r.addWidget(vl)
            lay.addLayout(r)

        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background:{C['border']};")
        lay.addWidget(sep)
        refs = {}
        for k in [
            "Current Phase", "Time in Phase",
            "Queue (primary)", "Waiting Time", "GAT Intent",
        ]:
            r = QHBoxLayout()
            r.addWidget(_lbl(k, C["muted"], 10))
            vl = _lbl("", C["text"], 10, bold=True)
            vl.setAlignment(Qt.AlignRight)
            r.addWidget(vl)
            lay.addLayout(r)
            refs[k] = vl
            if k == "Current Phase":
                vl.setTextFormat(Qt.RichText)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background:{C['border']};")
        lay.addWidget(sep2)

        lay.addWidget(_lbl("Last Decision", C["muted"], 9, bold=True))
        refs["decision"] = _lbl("", C["text"], 10, wrap=True)
        lay.addWidget(refs["decision"])
        refs["dec_time"] = _lbl("", C["dark"], 9)
        lay.addWidget(refs["dec_time"])

        neighbours = {
            "TL1": "TL2, TL4", "TL2": "TL1, TL3",
            "TL3": "TL2, TL4", "TL4": "TL1, TL3",
        }
        refs["GAT Intent"].setText(f"32-dim vector from {neighbours[t]}")
        return f, refs

    def refresh(self):
        for t in TL_IDS:
            s = self.state.tl[t]
            r = self.cards[t]
            pc = PHASE_COLORS[s["phase"]]
            r["Current Phase"].setText(
                f"<span style='color:{pc}'>{PHASE_LABELS[s['phase']]}</span>"
            )
            r["Time in Phase"].setText(f"{s['phase_dur']}s")
            edges = WEBSTER[t]["edges"].split(", ")
            if len(edges) >= 2:
                q1 = s["queue"] * 0.6
                q2 = s["queue"] * 0.4
                r["Queue (primary)"].setText(
                    f"{edges[0]}: {q1:.0f}v | {edges[1]}: {q2:.0f}v"
                )
            else:
                r["Queue (primary)"].setText(f"{s['queue']:.0f}v")
            r["Waiting Time"].setText(f"{s['delay']:.1f}s")
            r["decision"].setText(s["last_decision"])
            r["dec_time"].setText(f"Last switch: {s['last_switch']}")
class ControlTab(QScrollArea):
    mode_changed = pyqtSignal(int)
    scenario_changed = pyqtSignal(int)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWidgetResizable(True)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setSpacing(14)
        lay.setContentsMargins(16, 16, 16, 16)
        mb = QGroupBox("Global Control Mode")
        ml = QHBoxLayout(mb)
        self.mbg = QButtonGroup(self)
        for mid, txt in [
            (MODE_RL, "RL Mode (MAPPO-PIS)"),
        ]:
            rb = QRadioButton(txt)
            if mid == MODE_RL:
                rb.setChecked(True)
            self.mbg.addButton(rb, mid)
            ml.addWidget(rb)
        self.mbg.buttonClicked.connect(self._mode_click)
        lay.addWidget(mb)
        sb = QGroupBox("Scenario")
        sl = QHBoxLayout(sb)
        self.sbg = QButtonGroup(self)
        for sid, txt in [
            (SCEN_NORMAL, "Normal Traffic"),
        ]:
            rb = QRadioButton(txt)
            if sid == SCEN_NORMAL:
                rb.setChecked(True)
            self.sbg.addButton(rb, sid)
            sl.addWidget(rb)
        self.sbg.buttonClicked.connect(self._scen_click)
        lay.addWidget(sb)
        phase_grp = QGroupBox("Manual Phase Selection Force Phase on All Intersections")
        phase_lay = QVBoxLayout(phase_grp)

        info_row = QHBoxLayout()
        info_row.addWidget(_lbl(
            "Force all traffic lights to a specific phase index. "
            "Phase -1 restores full AI Auto (MAPPO) control.",
            C["muted"], 10, wrap=True,
        ))
        phase_lay.addLayout(info_row)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self.btn_p0 = QPushButton("Phase 0 (GREEN)")
        self.btn_p0.setStyleSheet(BTN_GREEN)
        self.btn_p0.setFixedWidth(130)
        self.btn_p0.clicked.connect(lambda: self._set_phase(0))
        btn_row.addWidget(self.btn_p0)

        self.btn_p1 = QPushButton("Phase 1 (RED)")
        self.btn_p1.setStyleSheet(BTN_DANGER)
        self.btn_p1.setFixedWidth(130)
        self.btn_p1.clicked.connect(lambda: self._set_phase(1))
        btn_row.addWidget(self.btn_p1)

        self.btn_p2 = QPushButton("Phase 2 (YELLOW)")
        self.btn_p2.setStyleSheet(f"background:{C['yellow']};color:#000;{_BTN}")
        self.btn_p2.setFixedWidth(130)
        self.btn_p2.clicked.connect(lambda: self._set_phase(2))
        btn_row.addWidget(self.btn_p2)



        self.btn_auto = QPushButton("Return to AI Auto")
        self.btn_auto.setStyleSheet(f"background:{C['accent']};color:white;{_BTN}")
        self.btn_auto.setFixedWidth(150)
        self.btn_auto.clicked.connect(lambda: self._set_phase(-1))
        btn_row.addWidget(self.btn_auto)

        btn_row.addStretch()
        phase_lay.addLayout(btn_row)

        self.phase_status_lbl = _lbl("Mode: AI Auto (MAPPO)", C["green"], 11, bold=True)
        phase_lay.addWidget(self.phase_status_lbl)

        lay.addWidget(phase_grp)
        lay.addWidget(_lbl("Per-Intersection Controls",
                           C["text"], 15, bold=True))
        grid = QGridLayout()
        grid.setSpacing(12)
        self.panels = {}
        for i, t in enumerate(TL_IDS):
            panel = self._build_panel(t)
            self.panels[t] = panel
            grid.addWidget(panel["frame"], i // 2, i % 2)
        lay.addLayout(grid)
        lay.addStretch()
        self.setWidget(box)

    def _build_panel(self, t):
        f = QFrame()
        f.setObjectName(f"panel_{t}")
        f.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            "border-radius:12px;"
        )
        lay = QVBoxLayout(f)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(8)

        lay.addWidget(_lbl(
            f"<b style='font-size:14px;'>{t}</b> "
            f"{WEBSTER[t]['phases']} phases | Edges: {WEBSTER[t]['edges']}"
        ))
        pr = QHBoxLayout()
        dot = QLabel()
        dot.setFixedSize(18, 18)
        plbl = _lbl("", C["text"], 12, bold=True)
        pr.addWidget(dot)
        pr.addWidget(plbl)
        pr.addStretch()
        lay.addLayout(pr)
        tr = QHBoxLayout()
        for d, lb in [(-5, "-5s"), (5, "+5s"), (10, "+10s")]:
            b = QPushButton(lb)
            b.setFixedWidth(56)
            if d < 0:
                b.setStyleSheet(BTN_SEC)
            b.clicked.connect(
                lambda _, tl=t, delta=d: self._extend(tl, delta)
            )
            tr.addWidget(b)
        tr.addStretch()
        lay.addLayout(tr)
        fr = QHBoxLayout()
        bg = QPushButton("Force GREEN")
        bg.setFixedWidth(108)
        bg.clicked.connect(lambda _, tl=t: self._force(tl, "green"))
        br = QPushButton("Force RED")
        br.setFixedWidth(108)
        br.setStyleSheet(BTN_DANGER)
        br.clicked.connect(lambda _, tl=t: self._force(tl, "red"))
        bc = QPushButton("Clear Override")
        bc.setFixedWidth(108)
        bc.setStyleSheet(BTN_SEC)
        bc.clicked.connect(lambda _, tl=t: self._clear(tl))
        fr.addWidget(bg)
        fr.addWidget(br)
        fr.addWidget(bc)
        fr.addStretch()
        lay.addLayout(fr)
        warn = _lbl("", C["yellow"], 10)
        lay.addWidget(warn)
        training_results = {
            "TL1": (-98.3, -99.0),
            "TL2": (-94.3, -88.9),
            "TL3": (-95.5, -93.9),
            "TL4": (-32.1, -91.2),
        }
        qp, dp = training_results[t]
        lay.addWidget(_lbl(
            f"Trained result: {qp:.1f}% queue vs Webster | "
            f"{dp:.1f}% delay vs Webster",
            C["dark"], 9,
        ))

        return dict(frame=f, dot=dot, plbl=plbl, warn=warn)

    def _mode_click(self, btn):
        mid = self.mbg.id(btn)
        if mid < 0:
            return
        self._post_mode(mid)

    def _post_mode(self, mid):
        """POST {"mode": mid} to /control/mode; update local state on success."""
        if not HAS_REQ:
            self.state.mode = mid
            EventBus.get().log(f"Mode changed -> {MODE_NAMES[mid]} (local only)", "MODE")
            self.mode_changed.emit(mid)
            return
        try:
            r = _requests.post(
                f"{self.state.api_url}/control/mode",
                json={"mode": mid},
                timeout=2,
            )
            r.raise_for_status()
            data = r.json()
            confirmed_mode = int(data.get("mode", mid))
            self.state.mode = confirmed_mode
            EventBus.get().log(
                f"Mode changed -> {MODE_NAMES[confirmed_mode]} (backend confirmed)",
                "MODE",
            )
            self.mode_changed.emit(confirmed_mode)
        except Exception as exc:
            self.state.mode = mid
            EventBus.get().log(
                f"Mode changed -> {MODE_NAMES[mid]} (backend unreachable: {type(exc).__name__})",
                "MODE",
            )
            self.mode_changed.emit(mid)

    def _scen_click(self, btn):
        sid = self.sbg.id(btn)
        if sid >= 0:
            old = self.state.scenario
            self.state.scenario = sid
            EventBus.get().log(
                f"Scenario changed -> {SCEN_NAMES[sid]}", "SCENARIO"
            )
            self.scenario_changed.emit(sid)

    def _extend(self, t, d):
        self.state.tl[t]["timer"] = max(1, self.state.tl[t]["timer"] + d)
        EventBus.get().log(
            f"{t} -> Timer {'extended' if d > 0 else 'reduced'} by {d}s",
            "OVERRIDE",
        )
        if HAS_REQ:
            try:
                _requests.post(
                    f"{self.state.api_url}/extend",
                    json={"tl_id": t, "delta": d},
                    timeout=1,
                )
            except Exception:
                pass

    def _force(self, t, color):
        s = self.state.tl[t]
        s["phase"] = 0 if color == "green" else 1
        s["override"] = color
        s["timer"] = 999
        s["last_decision"] = f"Operator forced {color.upper()}"
        EventBus.get().log(f"{t} -> Force {color.upper()}", "OVERRIDE")
        if HAS_REQ:
            try:
                _requests.post(
                    f"{self.state.api_url}/control/signal",
                    json={"tl_id": t, "action": color},
                    timeout=2,
                )
            except Exception as exc:
                EventBus.get().log(
                    f"{t} signal POST failed ({type(exc).__name__}) local only",
                    "OVERRIDE",
                )

    def _clear(self, t):
        s = self.state.tl[t]
        s["override"] = None
        s["timer"] = random.randint(15, 30)
        s["last_decision"] = (
            "Override cleared autonomous control resumed"
        )
        EventBus.get().log(
            f"{t} -> Override cleared, returning to auto", "OVERRIDE"
        )
        if HAS_REQ:
            try:
                _requests.post(
                    f"{self.state.api_url}/control/signal",
                    json={"tl_id": t, "action": "clear"},
                    timeout=2,
                )
            except Exception as exc:
                EventBus.get().log(
                    f"{t} clear POST failed ({type(exc).__name__}) local only",
                    "OVERRIDE",
                )

    def _set_phase(self, phase_id: int):
        """POST /set_phase/{phase_id} to backend, update UI status label."""
        if not HAS_REQ:
            EventBus.get().log(
                "requests library not available phase override not sent", "OVERRIDE"
            )
            return
        try:
            r = _requests.post(
                f"{self.state.api_url}/set_phase/{phase_id}",
                timeout=2,
            )
            r.raise_for_status()
            data = r.json()
            if phase_id == -1:
                label = "Mode: AI Auto (MAPPO)"
                color = C["green"]
                log_msg = "Manual override cleared MAPPO AI Auto mode restored"
            else:
                phase_label = data.get("phase_label", f"Phase {phase_id}")
                label = f"Mode: Manual Override {phase_label}"
                color = C["orange"]
                log_msg = f"All TLs forced to Phase {phase_id} ({phase_label})"
            if hasattr(self, 'phase_status_lbl'):
                self.phase_status_lbl.setText(label)
                self.phase_status_lbl.setStyleSheet(f"color:{color};font-size:11px;font-weight:700;")
            EventBus.get().log(log_msg, "OVERRIDE")
        except Exception as exc:
            EventBus.get().log(
                f"Phase override POST failed ({type(exc).__name__}) backend unreachable",
                "OVERRIDE",
            )

    def scroll_to(self, tl_id):
        if tl_id in self.panels:
            f = self.panels[tl_id]["frame"]
            self.ensureWidgetVisible(f, 50, 50)
            orig = f.styleSheet()
            f.setStyleSheet(
                f"background:{C['card']};border:2px solid {C['accent']};"
                "border-radius:10px;"
            )
            QTimer.singleShot(1500, lambda: f.setStyleSheet(orig))

    def refresh(self):
        be = {}
        try:
            win = self.window()
            be = getattr(win, "_backend", {}) or {}
        except Exception:
            pass
        override_phase = be.get("manual_override_phase")
        if hasattr(self, 'phase_status_lbl'):
            if override_phase is None:
                self.phase_status_lbl.setText("Mode: AI Auto (MAPPO)")
                self.phase_status_lbl.setStyleSheet(
                    f"color:{C['green']};font-size:11px;font-weight:700;"
                )
            else:
                _pl = ["GREEN", "RED", "YELLOW", "ALT"]
                pl = _pl[override_phase % len(_pl)]
                self.phase_status_lbl.setText(
                    f"Mode: Manual Override Phase {override_phase} ({pl})"
                )
                self.phase_status_lbl.setStyleSheet(
                    f"color:{C['orange']};font-size:11px;font-weight:700;"
                )

        for t in TL_IDS:
            s = self.state.tl[t]
            p = self.panels[t]
            col = PHASE_COLORS[s["phase"]]
            p["dot"].setStyleSheet(
                f"background:{col};border-radius:9px;border:none;"
            )
            tag = " [OVERRIDE]" if s["override"] else ""
            p["plbl"].setText(
                f"{PHASE_LABELS[s['phase']]} {s['timer']}s{tag}"
            )
            p["plbl"].setStyleSheet(
                f"color:{col};font-size:12px;font-weight:600;"
            )
            if s["override"]:
                p["warn"].setText(
                    "Agent suppressed operator override active"
                )
                p["warn"].setStyleSheet(
                    f"color:{C['orange']};font-size:10px;"
                )
            elif self.state.mode == MODE_RL:
                p["warn"].setText(
                    "Override will suppress agent output for this "
                    "intersection"
                )
                p["warn"].setStyleSheet(
                    f"color:{C['yellow']};font-size:10px;"
                )
            else:
                p["warn"].setText("")
class EventLogTab(QWidget):
    CAT_MAP = {
        "OVERRIDE": "Operator", "MODE": "Operator",
        "SCENARIO": "Operator",
        "AGENT": "Agent", "SYSTEM": "System",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.all_entries = []
        self.filters = {
            "Agent": True, "Operator": True,
            "System": True,
        }
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        h = QHBoxLayout()
        h.addWidget(_lbl("System Event Log", C["text"], 15, bold=True))
        h.addStretch()
        cb = QPushButton("Clear Log")
        cb.setStyleSheet(BTN_SEC)
        cb.clicked.connect(self._clear_log)
        h.addWidget(cb)
        lay.addLayout(h)
        fr = QHBoxLayout()
        fr.setSpacing(12)
        self.checks = {}
        for cat in ["Agent", "Operator", "System"]:
            chk = QCheckBox(cat)
            chk.setChecked(True)
            chk.stateChanged.connect(self._filter_changed)
            fr.addWidget(chk)
            self.checks[cat] = chk
        lay.addLayout(fr)

        self.view = QTextEdit()
        self.view.setReadOnly(True)
        lay.addWidget(self.view)

        EventBus.get().logged.connect(self._on_log)

    def _on_log(self, msg, tag):
        cat = self.CAT_MAP.get(tag, "System")
        color = LOG_COLORS.get(tag, C["accent"])
        html = (
            f"<span style='color:{C['dark']}'>{msg[:10]}</span>"
            f"<span style='color:{color}'>{msg[10:]}</span>"
        )
        self.all_entries.append((html, cat))
        if self.filters.get(cat, True):
            self.view.append(html)
            self.view.verticalScrollBar().setValue(
                self.view.verticalScrollBar().maximum()
            )

    def _filter_changed(self):
        for cat, chk in self.checks.items():
            self.filters[cat] = chk.isChecked()
        self._rerender()

    def _rerender(self):
        self.view.clear()
        for html, cat in self.all_entries:
            if self.filters.get(cat, True):
                self.view.append(html)

    def _clear_log(self):
        self.all_entries.clear()
        self.view.clear()
        EventBus.get().log("Log cleared", "SYSTEM")
class SettingsTab(QScrollArea):
    endpoint_changed = pyqtSignal(str)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setWidgetResizable(True)
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setSpacing(16)
        lay.setContentsMargins(16, 16, 16, 16)
        cc = QGroupBox("Connection Configuration")
        ccl = QVBoxLayout(cc)
        er = QHBoxLayout()
        er.addWidget(_lbl("FastAPI Endpoint:", C["muted"], 12, w=120))
        self.endpoint = QLineEdit(self.state.api_url)
        self.endpoint.setMinimumWidth(250)
        self.endpoint.editingFinished.connect(self._apply_endpoint)
        er.addWidget(self.endpoint)
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.setFixedWidth(130)
        self.test_btn.clicked.connect(self._test_conn)
        er.addWidget(self.test_btn)
        self.conn_result = _lbl("", C["muted"], 10)
        er.addWidget(self.conn_result)
        er.addStretch()
        ccl.addLayout(er)

        mr = QHBoxLayout()
        mr.addWidget(_lbl("MOCK MODE:", C["muted"], 12, w=120))
        self.mock_btn = QPushButton("AUTO - Polling FastAPI")
        self.mock_btn.setStyleSheet(BTN_DANGER)
        self.mock_btn.setFixedWidth(260)
        self.mock_btn.clicked.connect(self._toggle_mock)
        mr.addWidget(self.mock_btn)
        mr.addStretch()
        ccl.addLayout(mr)
        lay.addWidget(cc)
        ss = QGroupBox("System Status")
        ssl = QVBoxLayout(ss)
        for lbl, col, st in [
            ("SUMO", C["green"], "Connected (v1.26)"),
            ("TraCI", C["green"], "Running (port 8813)"),
            ("RL Model", C["green"],
             "Loaded (MAPPO-PIS, ONNX opset 17)"),
        ]:
            ssl.addWidget(_lbl(
                f"<span style='color:{col};font-size:16px'>&#9679;</span> "
                f"<b>{lbl}:</b> <span style='color:{col}'>{st}</span>",
                size=12,
            ))
        ssl.addWidget(_lbl(
            "Demand currently simulated (synthetic data)\n"
            "CV integration planned (YOLOv8 + BotSort tracking)",
            C["muted"], 10,
        ))
        lay.addWidget(ss)
        rp = QGroupBox("Runtime Parameters")
        rpl = QVBoxLayout(rp)
        for k, v in [
            ("SUMO step length", "2.0 s"),
            ("TraCI port", "8813"),
            ("GAT inference interval", "every 5 steps"),
            ("ONNX inference device", "CPU"),
            ("V_free", "13.89 m/s"),
            ("K_jam", "0.2"),
        ]:
            r = QHBoxLayout()
            r.addWidget(_lbl(f"<b>{k}:</b>", C["muted"], 11, w=180))
            r.addWidget(_lbl(v, C["text"], 11))
            r.addStretch()
            rpl.addLayout(r)
        lay.addWidget(rp)
        np_box = QGroupBox("Network Parameters - OM Square, Assiut")
        npl = QVBoxLayout(np_box)
        for k, v in [
            ("Intersections", "4 (TL1 TL2 TL3 TL4)"),
            ("E2 Detectors", "16 laneAreaDetectors"),
            ("TL1 Phases", "10 phases - edges E7, E9"),
            ("TL2 Phases", "9 phases - edges E1, E2, E4, E6"),
            ("TL3 Phases", "6 phases - edges E3, E5"),
            ("TL4 Phases", "4 phases - edges E8, E10"),
            ("Demand type", "Synthetic uniform + peak hour profiles"),
            ("Network type", "Urban arterial + bridge crossing corridor"),
            ("Junction model", "SUMO default gap acceptance"),
            ("Simulation horizon", "141 episodes ~350 steps (converged)"),
        ]:
            r = QHBoxLayout()
            r.addWidget(_lbl(f"<b>{k}:</b>", C["muted"], 11, w=180))
            r.addWidget(_lbl(v, C["text"], 11))
            r.addStretch()
            npl.addLayout(r)
        lay.addWidget(np_box)
        cv = QGroupBox("Computer Vision Pipeline")
        cvl = QVBoxLayout(cv)
        for lbl, col, st in [
            ("Demand Estimation", C["accent"], "Simulated (synthetic)"),
        ]:
            cvl.addWidget(_lbl(
                f"<span style='color:{col};font-size:16px'>&#9679;</span> "
                f"<b>{lbl}:</b> <span style='color:{col}'>{st}</span>",
                size=12,
            ))
        cvl.addWidget(_lbl(
            "When integrated, YOLOv8 vehicle counts will replace synthetic "
            "queue values in the agent observation vector automatically "
            "via the DesRUTGe adapter.",
            C["muted"], 10, wrap=True,
        ))
        lay.addWidget(cv)
        si = QGroupBox("System Information")
        sil = QVBoxLayout(si)
        for k, v in [
            ("Project", "SmartTraffic-OM"),
            ("Algorithm", "MAPPO-PIS + Hierarchical GAT"),
            ("Simulation", "SUMO 1.26 + TraCI"),
            ("Network", "OM Network - 4 intersections, 16 detectors"),
            ("Training", "Episode 141 - Converged"),
            ("Deployment", "ONNX opset 17 - CPU edge inference"),
            ("Competition", ""),
            ("University", "Sphinx University, Assiut, Egypt"),
        ]:
            r = QHBoxLayout()
            r.addWidget(_lbl(f"<b>{k}:</b>", C["muted"], 11, w=130))
            r.addWidget(_lbl(v, C["text"], 11))
            r.addStretch()
            sil.addLayout(r)
        lay.addWidget(si)
        lay.addStretch()
        self.setWidget(box)

    def _apply_endpoint(self):
        url = self.endpoint.text().strip().rstrip("/")
        if url:
            self.state.api_url = url
            self.endpoint_changed.emit(url)

    def _test_conn(self):
        url = self.endpoint.text().strip().rstrip("/")
        if not HAS_REQ:
            self.conn_result.setText(
                f"<span style='color:{C['red']}'>"
                "requests library not installed</span>"
            )
            return
        try:
            r = _requests.get(f"{url}/state", timeout=3)
            ms = r.elapsed.total_seconds() * 1000
            self.state.api_url = url
            self.endpoint_changed.emit(url)
            self.conn_result.setText(
                f"<span style='color:{C['green']}'>"
                f"Connected ({ms:.0f}ms)</span>"
            )
        except Exception as e:
            self.conn_result.setText(
                f"<span style='color:{C['red']}'>"
                f"Failed: {type(e).__name__}</span>"
            )

    def _toggle_mock(self):
        self.state.manual_mock = not self.state.manual_mock
        self.state.mock_mode = (
            self.state.manual_mock or not self.state.backend_connected
        )
        if self.state.manual_mock:
            self.mock_btn.setText("ON - Using Internal Simulation")
            self.mock_btn.setStyleSheet(BTN_GREEN)
            self.conn_result.setText(
                f"<span style='color:{C['yellow']}'>Manual mock mode enabled</span>"
            )
        else:
            self.mock_btn.setText("AUTO - Polling FastAPI")
            self.mock_btn.setStyleSheet(BTN_DANGER)
            EventBus.get().log(
                "Manual mock mode disabled - resuming FastAPI polling",
                "SYSTEM",
            )

class LoadEpisodeTab(QScrollArea):
    """Dedicated checkpoint viewer. Historical metrics stay here."""
    checkpoint_requested = pyqtSignal(int)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._loaded_ep = None # episode number last successfully loaded
        self._loaded_data = {} # raw metrics dict from that episode
        self.setWidgetResizable(True)

        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setSpacing(14)
        lay.setContentsMargins(16, 16, 16, 16)

        # -- Header ------------------------------------------------------------
        lay.addWidget(_lbl("Episode Viewer", C["text"], 17, bold=True))
        lay.addWidget(_lbl(
            "Load any saved checkpoint to inspect its per-agent metrics "
            "and complete metrics.json results.",
            C["muted"], 11, wrap=True,
        ))

        # -- Episode Loader ----------------------------------------------------
        loader_box = QGroupBox("Load Checkpoint")
        lb = QHBoxLayout(loader_box)
        lb.setSpacing(10)
        lb.addWidget(_lbl("Episode:", C["muted"], 12, w=60))
        self.ep_input = QLineEdit("20")
        self.ep_input.setFixedWidth(80)
        self.ep_input.setPlaceholderText("e.g. 20")
        lb.addWidget(self.ep_input)
        self.load_btn = QPushButton("Load Episode")
        self.load_btn.setFixedWidth(130)
        self.load_btn.clicked.connect(self._on_load_clicked)
        lb.addWidget(self.load_btn)
        self.load_status = _lbl("", C["muted"], 11)
        lb.addWidget(self.load_status)
        lb.addStretch()
        lay.addWidget(loader_box)

        # -- Loaded Episode Info Banner -------------------------------------
        self.ep_banner = QFrame()
        self.ep_banner.setStyleSheet(
            f"background:#172a45;border:1px solid {C['accent']};"
            "border-radius:10px;"
        )
        eb = QHBoxLayout(self.ep_banner)
        eb.setContentsMargins(18, 10, 18, 10)
        self.ep_banner_lbl = _lbl(
            "No episode loaded yet enter an episode number above and click Load.",
            C["muted"], 12,
        )
        eb.addWidget(self.ep_banner_lbl)
        eb.addStretch()
        self.ep_ts_lbl = _lbl("", C["dark"], 10)
        eb.addWidget(self.ep_ts_lbl)
        lay.addWidget(self.ep_banner)

        # -- Per-TL Metrics Grid (from metrics.json) -----------------------
        lay.addWidget(_lbl("Per-Agent Metrics Loaded Episode", C["text"], 15, bold=True))
        tl_panel = QFrame()
        tl_panel.setStyleSheet(
            f"background:{C['card']};border:1px solid {C['border']};"
            "border-radius:10px;"
        )
        tl_grid = QGridLayout(tl_panel)
        tl_grid.setContentsMargins(14, 12, 14, 12)
        tl_grid.setSpacing(10)
        self._tl_metric_cards = {}
        for idx2, tid in enumerate(["TL1", "TL2", "TL3", "TL4"]):
            col = TL_AGENT_CLR[tid]
            cell = QFrame()
            cell.setStyleSheet(
                f"background:{C['panel']};border:1px solid {col};"
                "border-radius:8px;"
            )
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(14, 12, 14, 12)
            cl.setSpacing(6)
            # TL header
            hr = QHBoxLayout()
            hr.addWidget(_lbl(tid, col, 15, bold=True))
            hr.addStretch()
            hr.addWidget(_lbl(TL_ROLES[tid], C["muted"], 9))
            cl.addLayout(hr)
            sep = QFrame(); sep.setFixedHeight(1)
            sep.setStyleSheet(f"background:{C['border']};")
            cl.addWidget(sep)
            refs = {}
            for metric, label, unit, clr in [
                ("queue", "Avg Queue", "veh", C["orange"]),
                ("wait", "Avg Wait", "s", C["yellow"]),
                ("reward", "Avg Reward", "/tick", C["green"]),
                ("loss", "Actor Loss", "", C["purple"]),
            ]:
                row = QHBoxLayout()
                row.addWidget(_lbl(label, C["muted"], 10))
                row.addStretch()
                val_lbl = _lbl("\u2014", clr, 16, bold=True)
                val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                row.addWidget(val_lbl)
                if unit:
                    row.addWidget(_lbl(unit, C["dark"], 9))
                cl.addLayout(row)
                refs[metric] = val_lbl
            tl_grid.addWidget(cell, 0, idx2)
            self._tl_metric_cards[tid] = refs
        lay.addWidget(tl_panel)

        # -- RL vs Fixed-Time Comparison Table -----------------------------
        lay.addWidget(_lbl("RL vs Fixed-Time Episode Summary", C["text"], 15, bold=True))
        self.cmp_table = QTableWidget(3, 3)
        self.cmp_table.setHorizontalHeaderLabels(["Metric", "RL Agent", "Fixed-Time"])
        self.cmp_table.setVerticalHeaderLabels(["Avg Queue", "Avg Wait", "Avg Reward"])
        self.cmp_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cmp_table.verticalHeader().setDefaultSectionSize(44)
        self.cmp_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.cmp_table.setFixedHeight(175)
        self.cmp_table.setAlternatingRowColors(True)
        for i in range(3):
            for j in range(3):
                item = QTableWidgetItem("\u2014")
                item.setTextAlignment(Qt.AlignCenter)
                self.cmp_table.setItem(i, j, item)
        lay.addWidget(self.cmp_table)

        # -- Improvement Banner --------------------------------------------
        imp_row = QHBoxLayout()
        imp_row.setSpacing(12)
        self.imp_queue_card = KPICard("QUEUE IMPROVEMENT", "\u2014", "% vs fixed-time", "", C["green"])
        self.imp_wait_card = KPICard("WAIT IMPROVEMENT", "\u2014", "% vs fixed-time", "", C["accent"])
        imp_row.addWidget(self.imp_queue_card)
        imp_row.addWidget(self.imp_wait_card)
        imp_row.addStretch()
        lay.addLayout(imp_row)

        lay.addWidget(_lbl("Full metrics.json", C["text"], 15, bold=True))
        self.raw_metrics_view = QTextEdit()
        self.raw_metrics_view.setReadOnly(True)
        self.raw_metrics_view.setMinimumHeight(260)
        self.raw_metrics_view.setText("Load an episode to view the complete metrics.json payload.")
        lay.addWidget(self.raw_metrics_view)
        lay.addStretch()
        self.setWidget(box)

    # -- Signals / slots ---------------------------------------------------
    def _on_load_clicked(self):
        raw = self.ep_input.text().strip() or "20"
        try:
            ep = int(raw)
        except ValueError:
            self.set_status(False, "Episode must be an integer")
            return
        self.load_btn.setEnabled(False)
        self.load_status.setText(
            f"<span style='color:{C['yellow']}'>Loading ep{ep}...</span>"
        )
        self.checkpoint_requested.emit(ep)

    def set_status(self, ok, message):
        self.load_btn.setEnabled(True)
        color = C["green"] if ok else C["red"]
        self.load_status.setText(f"<span style='color:{color}'>{message}</span>")

    def on_checkpoint_loaded(self, ok, message, episode, raw_metrics):
        """Called by MainWindow after checkpoint finishes loading."""
        self.set_status(ok, message)
        if not ok:
            return
        self._loaded_ep = episode
        self._loaded_data = raw_metrics or {}
        self._populate_episode_data()

    def _populate_episode_data(self):
        """Fill all static widgets from self._loaded_data (metrics.json content)."""
        m = self._loaded_data
        ep = self._loaded_ep
        rl = m.get("rl") or {}
        fx = m.get("fixed_time") or {}
        imp = m.get("improvement") or {}
        ts = m.get("timestamp", "")

        # Banner
        self.ep_banner_lbl.setText(
            f"<b style='color:{C['accent']};font-size:14px;'>"
            f"Episode {ep} loaded</b>"
            f" <span style='color:{C['muted']};font-size:11px;'>"
            f"checkpoint: results/checkpoints/ep{ep:04d}</span>"
        )
        self.ep_ts_lbl.setText(ts)
        self.raw_metrics_view.setPlainText(json.dumps(m, indent=2, sort_keys=True))

        # Per-TL metrics
        rl_queue = rl.get("queue") or {}
        rl_wait = rl.get("wait") or {}
        rl_reward = rl.get("reward") or {}
        rl_loss = rl.get("loss") or rl.get("actor_loss") or {}
        for tid in ["TL1", "TL2", "TL3", "TL4"]:
            refs = self._tl_metric_cards[tid]
            def _fmt(d, k, fmt="{:.2f}"):
                v = d.get(k)
                return fmt.format(float(v)) if v is not None else "\u2014"
            refs["queue"].setText(_fmt(rl_queue, tid, "{:.1f}"))
            refs["wait"].setText(_fmt(rl_wait, tid, "{:.1f}"))
            refs["reward"].setText(_fmt(rl_reward, tid, "{:.4f}"))
            refs["loss"].setText(_fmt(rl_loss, tid, "{:.5f}"))

        # Comparison table rows: [label, rl_val, fx_val]
        rows = [
            ("Avg Queue", rl.get("avg_queue"), fx.get("avg_queue")),
            ("Avg Wait", rl.get("avg_wait"), fx.get("avg_wait")),
            ("Avg Reward", rl.get("avg_reward"), fx.get("avg_reward")),
        ]
        for i, (label, rl_v, fx_v) in enumerate(rows):
            def _cell(v, fmt="{:.4f}"):
                return fmt.format(float(v)) if v is not None else "\u2014"
            self.cmp_table.item(i, 0).setText(label)
            rl_item = self.cmp_table.item(i, 1)
            fx_item = self.cmp_table.item(i, 2)
            rl_item.setText(_cell(rl_v))
            fx_item.setText(_cell(fx_v))
            # Color RL better than fixed in green
            if rl_v is not None and fx_v is not None:
                better = float(rl_v) < float(fx_v) if i < 2 else float(rl_v) > float(fx_v)
                rl_item.setForeground(QColor(C["green"] if better else C["red"]))

        # Improvement cards
        q_pct = imp.get("queue_pct")
        w_pct = imp.get("wait_pct")
        if q_pct is not None:
            q_f = float(q_pct)
            arrow = "\u2193" if q_f > 0 else "\u2191"
            self.imp_queue_card.set(
                f"{q_f:.1f}%",
                f"{arrow} queue reduced vs fixed-time",
            )
            self.imp_queue_card.set_bar(min(100, abs(q_f) * 1.2),
                                        C["green"] if q_f > 0 else C["red"])
        if w_pct is not None:
            w_f = float(w_pct)
            arrow = "\u2193" if w_f > 0 else "\u2191"
            self.imp_wait_card.set(
                f"{w_f:.1f}%",
                f"{arrow} wait reduced vs fixed-time",
            )
            self.imp_wait_card.set_bar(min(100, abs(w_f) * 1.2),
                                       C["accent"] if w_f > 0 else C["red"])

    def refresh(self):
        return

class MainWindow(QMainWindow):
    poll_state_requested = pyqtSignal()
    checkpoint_requested = pyqtSignal(int)
    fetch_episode_requested = pyqtSignal(int)  # read-only episode fetch, never restarts sim

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "SmartTraffic-OM - MAPPO-PIS Operator Terminal | "
        )
        self.setMinimumSize(1100, 750)
        self.resize(1300, 850)
        self.state = SimState()
        self._backend = {}
        self._ui_dirty = True
        self._last_ui_signature = None
        self._poll_thread = None
        self._poller = None

        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        hdr = QFrame()
        hdr.setFixedHeight(50)
        hdr.setStyleSheet(
            f"background:{C['panel']};"
            f"border-bottom:1px solid {C['border']};"
        )
        hl = QHBoxLayout(hdr)
        hl.setContentsMargins(20, 0, 20, 0)
        hl.addWidget(_lbl(
            "<b style='color:#3b82f6;font-size:15px;'>SmartTraffic-OM</b>"
            "<span style='color:#94a3b8;font-size:11px;'>"
            " | MAPPO-PIS + GAT | Operator Control Terminal</span>"
        ))
        hl.addStretch()
        self.clock = _lbl("", C["muted"], 11)
        hl.addWidget(self.clock)
        root.addWidget(hdr)

        self.conn_bar = ConnectionBar(self.state)
        root.addWidget(self.conn_bar)

        self.tabs = QTabWidget()
        self.dash = DashboardTab(self.state)
        self.agents = AgentsTab(self.state)
        self.ctrl = ControlTab(self.state)
        self.elog = EventLogTab()
        self.settings = SettingsTab(self.state)
        self.load_ep_tab = LoadEpisodeTab(self.state)
        self.tabs.addTab(self.dash, "Dashboard")
        self.tabs.addTab(self.agents, "RL Agents")
        self.tabs.addTab(self.ctrl, "Control")
        self.tabs.addTab(self.elog, "Event Log")
        self.tabs.addTab(self.load_ep_tab, "Load Episode")
        self.tabs.addTab(self.settings, "System")
        root.addWidget(self.tabs)

        sb = self.statusBar()
        sb.setStyleSheet(
            f"background:{C['panel']};color:{C['muted']};"
            f"font-size:10px;border-top:1px solid {C['border']};"
            "padding:2px 0;"
        )
        sb.showMessage(
            " SmartTraffic-OM | Operator Control Terminal | "
            " | Sphinx University, Assiut"
        )

        self.ctrl.mode_changed.connect(
            lambda m: self.dash.banner.set_mode(m)
        )
        self.dash.topo.intersection_clicked.connect(self._on_node_click)
        self.tabs.currentChanged.connect(
            lambda _: self._refresh_visible_tab(force=True)
        )
        self.load_ep_tab.checkpoint_requested.connect(self._request_checkpoint_load_ep)

        if HAS_REQ:
            self._poll_thread = QThread(self)
            self._poller = BackendWorker(self.state.api_url)
            self._poller.moveToThread(self._poll_thread)
            self.poll_state_requested.connect(self._poller.poll_state)
            self.checkpoint_requested.connect(self._poller.load_checkpoint)
            self.fetch_episode_requested.connect(self._poller.fetch_episode_metrics)
            self.settings.endpoint_changed.connect(self._poller.set_base_url)
            self._poller.state_ready.connect(self._apply_backend_state)
            self._poller.poll_error.connect(self._handle_backend_error)
            self._poller.episode_metrics_ready.connect(self._handle_episode_metrics_result)
            self._poll_thread.start()
        else:
            self.state.mock_mode = True
            self.state.backend_error = "requests library missing"
            self.settings.conn_result.setText(
                f"<span style='color:{C['red']}'>requests library not installed</span>"
            )

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

        bus = EventBus.get()
        bus.log("System initialized SmartTraffic-OM Operator Terminal", "SYSTEM")
        bus.log("Connecting to FastAPI backend at " + self.state.api_url, "SYSTEM")
        self.conn_bar.show_disconnected()
        self.settings.conn_result.setText(
            f"<span style='color:{C["yellow"]}'>Connecting</span>"
        )
        if self._poller is not None:
            self.poll_state_requested.emit()

    def _on_node_click(self, tl_id):
        self.tabs.setCurrentIndex(3)
        self.ctrl.scroll_to(tl_id)

    def _state_signature(self):
        tl_sig = []
        for tl_id in TL_IDS:
            s = self.state.tl[tl_id]
            tl_sig.append((
                s["phase"],
                s["timer"],
                round(s["queue"], 2),
                round(s["delay"], 2),
                s["override"],
                s["last_switch"],
                s["last_decision"],
            ))
        return (
            self.state.tick,
            self.state.mode,
            self.state.scenario,
            self.state.mock_mode,
            self.state.warmed_up,
            self.state.backend_connected,
            self.state.vehicles_cleared,
            round(self.state.throughput, 1),
            tuple(tl_sig),
        )

    def _refresh_visible_tab(self, force=False):
        sig = self._state_signature()
        if not force and not self._ui_dirty and sig == self._last_ui_signature:
            return
        self._last_ui_signature = sig
        self._ui_dirty = False
        cur = self.tabs.currentIndex()
        if cur == 0:
            self.dash.refresh()
        elif cur == 1:
            self.agents.refresh()
        elif cur == 2:
            self.ctrl.refresh()
        elif cur == 4:
            self.load_ep_tab.refresh()

    def _tick(self):
        if self.state.mock_mode:
            self.state.step()
            self._ui_dirty = True
        if self._poller is not None and not self.state.manual_mock:
            self.poll_state_requested.emit()
        self.conn_bar.refresh()
        self.clock.setText(datetime.now().strftime("%H:%M:%S"))
        self._refresh_visible_tab()

    def _apply_backend_state(self, data):
        if self.state.manual_mock:
            return

        self._backend = data
        agents = data.get("agents") or data.get("intersections") or {}
        was_connected = self.state.backend_connected
        self.state.backend_connected = True
        self.state.backend_error = ""
        self.state.mock_mode = False
        self.state.warmed_up = bool(data.get("warmed_up", True))
        self.state.tick = int(data.get("tick", self.state.tick))
        new_mode = int(data.get("mode", self.state.mode))
        self.state.mode = new_mode
        self.state.scenario = int(data.get("scenario", self.state.scenario))
        self.state.throughput = float(
            data.get("throughput", self.state.throughput)
        )
        self.state.vehicles_cleared = int(
            data.get("vehicles_cleared", self.state.vehicles_cleared)
        )
        self.state.phase_history = data.get(
            "phase_history", self.state.phase_history
        )

        rb = self.ctrl.mbg.button(new_mode)
        if rb is not None and not rb.isChecked():
            rb.blockSignals(True)
            rb.setChecked(True)
            rb.blockSignals(False)
        self.dash.banner.set_mode(new_mode)

        if not self.state.warmed_up and not agents:
            # Backend connected but no agent data yet - brief connecting state
            self.conn_bar.show_warming_up()
            self.settings.conn_result.setText(
                f"<span style='color:{C['yellow']}'>Connecting...</span>"
            )
            if not was_connected:
                EventBus.get().log(
                    f"FastAPI connected - waiting for first SUMO tick at {self.state.api_url}",
                    "SYSTEM",
                )
            self._ui_dirty = True
            return

        for tl_id in TL_IDS:
            src = agents.get(tl_id)
            if not src:
                continue
            dst = self.state.tl[tl_id]
            dst["queue"] = float(src.get("queue", dst["queue"]))
            dst["delay"] = float(src.get("delay", dst["delay"]))
            dst["speed"] = float(src.get("speed", dst["speed"]))
            dst["phase"] = int(src.get("phase", dst["phase"]))
            dst["timer"] = int(src.get("timer", dst["timer"]))
            dst["phase_dur"] = int(src.get("phase_dur", dst["phase_dur"]))
            dst["override"] = src.get("override", dst["override"])
            dst["emergency"] = bool(src.get("emergency", dst["emergency"]))
            dst["last_switch"] = src.get("last_switch", dst["last_switch"])
            dst["last_decision"] = (
                src.get("last_decision")
                or dst["last_decision"]
                or "Warming up..."
            )

        avg_d = float(np.mean([self.state.tl[t]["delay"] for t in TL_IDS]))
        self.state._delay_sum += avg_d
        self.state._delay_n += 1
        self.conn_bar.show_connected()
        self.settings.conn_result.setText(
            f"<span style='color:{C['green']}'>Connected</span>"
        )
        if not was_connected:
            EventBus.get().log(
                f"FastAPI connected - live data from {self.state.api_url}",
                "SYSTEM",
            )
        self._ui_dirty = True

    def _handle_backend_error(self, error_name):
        if self.state.manual_mock:
            return
        was_connected = self.state.backend_connected
        self.state.backend_connected = False
        self.state.backend_error = error_name
        self.state.warmed_up = False
        self.state.mock_mode = True
        self.conn_bar.show_disconnected()
        self.settings.conn_result.setText(
            f"<span style='color:{C['red']}'>Backend Disconnected</span>"
        )
        self._ui_dirty = True
        if was_connected:
            EventBus.get().log(
                f"Backend Disconnected ({error_name}) - switched to mock mode",
                "SYSTEM",
            )

    def _request_checkpoint_load_ep(self, episode):
        """Called from the Load Episode tab - read-only fetch, never restarts simulation."""
        if self._poller is None:
            self.load_ep_tab.set_status(False, "requests library not installed")
            return
        if self.state.manual_mock:
            self.load_ep_tab.set_status(False, "Disable manual mock mode first")
            return
        self._last_ckpt_episode = episode
        self.load_ep_tab.load_btn.setEnabled(False)
        _yel = C["yellow"]
        self.load_ep_tab.load_status.setText(
            f"<span style='color:{_yel}'>Loading ep{episode}...</span>"
        )
        # Use read-only /episode/{ep} endpoint - does NOT stop/restart the simulation
        self.fetch_episode_requested.emit(episode)

    def _handle_episode_metrics_result(self, ok, message, episode, raw_metrics):
        """Called after read-only /episode/{ep} fetch from Load Episode tab.
        NEVER touches simulation state or dashboard - only populates the Episode tab."""
        self.load_ep_tab.on_checkpoint_loaded(ok, message, episode, raw_metrics)
        # Dashboard and live simulation are completely unaffected

    def closeEvent(self, event):
        if self._poll_thread is not None:
            self._poll_thread.quit()
            self._poll_thread.wait(1500)
        super().closeEvent(event)
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(C["bg"]))
    pal.setColor(QPalette.WindowText, QColor(C["text"]))
    pal.setColor(QPalette.Base, QColor(C["panel"]))
    pal.setColor(QPalette.AlternateBase, QColor("#1a2435"))
    pal.setColor(QPalette.Text, QColor(C["text"]))
    pal.setColor(QPalette.Button, QColor(C["card"]))
    pal.setColor(QPalette.ButtonText, QColor(C["text"]))
    pal.setColor(QPalette.Highlight, QColor("#2a4a7a"))
    pal.setColor(QPalette.HighlightedText, QColor("#e8edf4"))
    app.setPalette(pal)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()














