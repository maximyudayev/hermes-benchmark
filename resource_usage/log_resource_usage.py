#!/usr/bin/env python3
"""
High-precision unified resource usage logger (Windows, Linux & NVIDIA Jetson).

Records timestamp, per-core CPU load & frequency, RAM, GPU, and Power metrics to CSV.
Fully compatible with the schema parsed by parse_resource_usage.py.

Supported Backends:
    - NVIDIA Jetson (all-in-one via jtop / jetson-stats: CPU, GPU, EMC, RAM, Power)
    - NVIDIA NVML (in-process via pynvml / nvidia-ml-py for Windows & Linux PC/server)
    - Windows Performance Counters & DXGI (PDH via ctypes: GPU Engine, VRAM, Energy Meter RAPL Power)
    - Linux DRM / Sysfs (/sys/class/drm/card*/device/gpu_busy_percent for Linux AMD/Intel)
    - Fallback / Null (records 0.0 when no supported GPU is detected or monitor is disabled)

Usage examples:
    uv run python log_resource_usage.py -o metrics.csv -i 0.5
    python log_resource_usage.py --gpu-backend jetson --interval 0.5
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import warnings

# Suppress pynvml deprecation notice
warnings.filterwarnings("ignore", category=FutureWarning, module=".*nvml.*")

try:
    import psutil
except ImportError:
    print(
        "Error: 'psutil' is required. Install with 'pip install psutil' or 'uv add psutil'.",
        file=sys.stderr,
    )
    sys.exit(1)


# Global flag for graceful termination
stop_logging = False


def handle_sigint(sig, frame):
    global stop_logging
    print("\nStopping logger...")
    stop_logging = True


def is_jetson() -> bool:
    """Detect if running on an NVIDIA Jetson hardware platform."""
    return os.path.exists("/etc/nv_tegra_release") or os.path.exists(
        "/proc/device-tree/nvidia,tegra-arch"
    )


# ==============================================================================
# GPU & Platform Trackers
# ==============================================================================


class BaseGPUTracker:
    """Base class for platform GPU metrics provider."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self.device_name = "Unknown"

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "gpu_load_pct": 0.0,
            "gpu_freq_mhz": 0.0,
            "power_cur_mw": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_tot_mb": 0.0,
        }

    def close(self):
        pass


class JetsonTracker(BaseGPUTracker):
    """NVIDIA Jetson all-in-one resource tracker using jtop (jetson-stats)."""

    def __init__(self, interval: float = 0.2):
        super().__init__(0)
        try:
            # pyrefly: ignore [missing-import]
            from jtop import jtop, JtopException

            self.jtop_cls = jtop
            self.JtopException = JtopException
        except ImportError:
            raise ImportError(
                "NVIDIA Jetson platform detected, but 'jtop' (jetson-stats) is not installed.\n"
                "Install it with: sudo pip3 install jetson-stats"
            )

        self.jetson = self.jtop_cls(interval=interval)
        self.jetson.start()
        if not self.jetson.ok():
            raise RuntimeError(
                "Could not connect to jtop daemon. Is jetson_stats service running?"
            )
        self.device_name = "NVIDIA Jetson (jtop)"

    def is_ok(self) -> bool:
        return self.jetson.ok()

    def get_snapshot(self) -> Dict[str, Any]:
        stats = self.jetson.stats
        cpu_data = self.jetson.cpu
        gpu_data = self.jetson.gpu["gpu"]
        ram_data = self.jetson.memory["RAM"]
        power_data = self.jetson.power["tot"]

        return {
            # Memory
            "ram_used_mb": ram_data["used"] // 1024,
            "ram_tot_mb": ram_data["tot"] // 1024,
            "ram_pct": round(ram_data["used"] / ram_data["tot"] * 100, 2),
            # GPU
            "gpu_load_pct": gpu_data["status"]["load"],
            "gpu_freq_mhz": gpu_data["freq"]["cur"],
            # Memory Controller (EMC)
            "emc_pct": stats["EMC"],
            # Total Module Power (mW)
            "power_cur_mw": power_data["power"],
            "power_avg_mw": power_data["avg"],
            # CPU
            "cpu_cores": cpu_data["cpu"],
        }

    def close(self):
        try:
            self.jetson.close()
        except Exception:
            pass


class NvmlGPUTracker(BaseGPUTracker):
    """NVIDIA GPU monitoring via NVML (pynvml / nvidia-ml-py)."""

    def __init__(self, gpu_index: int = 0):
        super().__init__(gpu_index)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pynvml

        self.pynvml = pynvml
        self.pynvml.nvmlInit()
        count = self.pynvml.nvmlDeviceGetCount()
        if gpu_index >= count:
            raise ValueError(f"Requested GPU index {gpu_index} but only {count} NVIDIA GPU(s) found.")
        self.handle = self.pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        name = self.pynvml.nvmlDeviceGetName(self.handle)
        self.device_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)

    def get_metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {
            "gpu_load_pct": 0.0,
            "gpu_freq_mhz": 0.0,
            "power_cur_mw": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_tot_mb": 0.0,
        }
        try:
            util = self.pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            metrics["gpu_load_pct"] = float(util.gpu)
        except Exception:
            pass

        try:
            clock = self.pynvml.nvmlDeviceGetClockInfo(
                self.handle, self.pynvml.NVML_CLOCK_GRAPHICS
            )
            metrics["gpu_freq_mhz"] = float(clock)
        except Exception:
            pass

        try:
            power = self.pynvml.nvmlDeviceGetPowerUsage(self.handle)
            metrics["power_cur_mw"] = float(power)
        except Exception:
            pass

        try:
            mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            metrics["gpu_mem_used_mb"] = mem.used // (1024 * 1024)
            metrics["gpu_mem_tot_mb"] = mem.total // (1024 * 1024)
        except Exception:
            pass

        return metrics

    def close(self):
        try:
            self.pynvml.nvmlShutdown()
        except Exception:
            pass


class WindowsPdhGPUTracker(BaseGPUTracker):
    """Windows Performance Counters (PDH) GPU tracker for AMD, Intel, and DirectX engines.
    
    Extracts GPU engine load, dedicated VRAM usage, total adapter VRAM (via DXGI),
    and device package power (via Energy Meter / RAPL counters).
    """

    def __init__(self, gpu_index: int = 0):
        super().__init__(gpu_index)
        import ctypes
        from ctypes import wintypes

        self.ctypes = ctypes
        self.wintypes = wintypes
        self.pdh = ctypes.windll.pdh

        class CounterVal(ctypes.Structure):
            _fields_ = [
                ("CStatus", wintypes.DWORD),
                ("doubleValue", ctypes.c_double),
            ]

        class PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
            _fields_ = [("szName", wintypes.LPWSTR), ("FmtValue", CounterVal)]

        self.CounterItemType = PDH_FMT_COUNTERVALUE_ITEM
        self.PDH_FMT_DOUBLE = 0x00000200

        # Discover GPU adapter name and total memory via DXGI
        self.adapter_total_memory_mb, desc_name = self._discover_dxgi_adapter_memory(gpu_index)
        self.device_name = desc_name or "Windows DirectX GPU Engine"

        self.hQuery = wintypes.HANDLE()
        self.hCounterEng = wintypes.HANDLE()
        self.hCounterMem = wintypes.HANDLE()
        self.hCounterPwr = wintypes.HANDLE()

        res = self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.hQuery))
        if res != 0:
            raise RuntimeError(f"PdhOpenQueryW failed with error code: {res}")

        # 1. GPU Engine Utilization
        res = self.pdh.PdhAddEnglishCounterW(
            self.hQuery, r"\GPU Engine(*)\Utilization Percentage", 0, ctypes.byref(self.hCounterEng)
        )
        if res != 0:
            self.pdh.PdhCloseQuery(self.hQuery)
            raise RuntimeError(f"PdhAddEnglishCounterW for GPU Engine failed: {res}")

        # 2. GPU Dedicated VRAM Usage
        res_mem = self.pdh.PdhAddEnglishCounterW(
            self.hQuery, r"\GPU Adapter Memory(*)\Dedicated Usage", 0, ctypes.byref(self.hCounterMem)
        )
        if res_mem != 0:
            self.hCounterMem = None

        # 3. Device Package Power (Energy Meter / RAPL)
        res_pwr = self.pdh.PdhAddEnglishCounterW(
            self.hQuery, r"\Energy Meter(*)\Power", 0, ctypes.byref(self.hCounterPwr)
        )
        if res_pwr != 0:
            self.hCounterPwr = None

        # Prime the query
        self.pdh.PdhCollectQueryData(self.hQuery)

    def _discover_dxgi_adapter_memory(self, target_index: int):
        """Query DXGI for adapter description and total video memory (MB)."""
        try:
            from ctypes import wintypes

            class LUID(self.ctypes.Structure):
                _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

            class DXGI_ADAPTER_DESC1(self.ctypes.Structure):
                _fields_ = [
                    ("Description", wintypes.WCHAR * 128),
                    ("VendorId", wintypes.UINT),
                    ("DeviceId", wintypes.UINT),
                    ("SubSysId", wintypes.UINT),
                    ("Revision", wintypes.UINT),
                    ("DedicatedVideoMemory", self.ctypes.c_size_t),
                    ("DedicatedSystemMemory", self.ctypes.c_size_t),
                    ("SharedSystemMemory", self.ctypes.c_size_t),
                    ("AdapterLuid", LUID),
                    ("Flags", wintypes.UINT),
                ]

            class GUID(self.ctypes.Structure):
                _fields_ = [
                    ("Data1", wintypes.DWORD),
                    ("Data2", wintypes.WORD),
                    ("Data3", wintypes.WORD),
                    ("Data4", self.ctypes.c_ubyte * 8),
                ]

            # IID_IDXGIFactory1: 770aae78-f26f-4dba-a829-253c83d1b387
            IID_IDXGIFactory1 = GUID(
                0x770AAE78,
                0xF26F,
                0x4DBA,
                (self.ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
            )
            dxgi = self.ctypes.windll.dxgi
            pFactory = self.ctypes.c_void_p()
            if dxgi.CreateDXGIFactory1(self.ctypes.byref(IID_IDXGIFactory1), self.ctypes.byref(pFactory)) != 0:
                return 0.0, "Windows DirectX GPU"

            vtbl = self.ctypes.cast(
                self.ctypes.cast(pFactory, self.ctypes.POINTER(self.ctypes.c_void_p))[0],
                self.ctypes.POINTER(self.ctypes.c_void_p),
            )
            EnumAdapters1_proto = self.ctypes.WINFUNCTYPE(
                wintypes.LONG, self.ctypes.c_void_p, wintypes.UINT, self.ctypes.POINTER(self.ctypes.c_void_p)
            )
            EnumAdapters1 = EnumAdapters1_proto(vtbl[12])

            i = 0
            pAdapter = self.ctypes.c_void_p()
            matched_mem = 0.0
            matched_name = "Windows DirectX GPU"
            idx_count = 0

            while EnumAdapters1(pFactory, i, self.ctypes.byref(pAdapter)) == 0:
                adapter_vtbl = self.ctypes.cast(
                    self.ctypes.cast(pAdapter, self.ctypes.POINTER(self.ctypes.c_void_p))[0],
                    self.ctypes.POINTER(self.ctypes.c_void_p),
                )
                GetDesc1_proto = self.ctypes.WINFUNCTYPE(
                    wintypes.LONG, self.ctypes.c_void_p, self.ctypes.POINTER(DXGI_ADAPTER_DESC1)
                )
                GetDesc1 = GetDesc1_proto(adapter_vtbl[10])
                desc = DXGI_ADAPTER_DESC1()
                if GetDesc1(pAdapter, self.ctypes.byref(desc)) == 0:
                    # Skip software renderer (flags & 2)
                    if not (desc.Flags & 2):
                        mem_mb = desc.DedicatedVideoMemory / (1024 * 1024)
                        if mem_mb == 0 and desc.SharedSystemMemory > 0:
                            mem_mb = desc.SharedSystemMemory / (1024 * 1024)
                        if idx_count == target_index or matched_mem == 0.0:
                            matched_mem = round(mem_mb, 2)
                            matched_name = desc.Description
                        idx_count += 1
                Release_proto = self.ctypes.WINFUNCTYPE(wintypes.ULONG, self.ctypes.c_void_p)
                Release = Release_proto(adapter_vtbl[2])
                Release(pAdapter)
                i += 1

            return matched_mem, matched_name
        except Exception:
            return 0.0, "Windows DirectX GPU"

    def _read_counter_items(self, hCounter):
        """Read and format array of values from a wildcard PDH counter handle."""
        if not hCounter:
            return []
        item_buffer_size = self.wintypes.DWORD(0)
        item_count = self.wintypes.DWORD(0)

        self.pdh.PdhGetFormattedCounterArrayW(
            hCounter,
            self.PDH_FMT_DOUBLE,
            self.ctypes.byref(item_buffer_size),
            self.ctypes.byref(item_count),
            None,
        )
        if item_buffer_size.value > 0 and item_count.value > 0:
            buffer = (self.ctypes.c_byte * item_buffer_size.value)()
            res = self.pdh.PdhGetFormattedCounterArrayW(
                hCounter,
                self.PDH_FMT_DOUBLE,
                self.ctypes.byref(item_buffer_size),
                self.ctypes.byref(item_count),
                self.ctypes.byref(buffer),
            )
            if res == 0:
                items = self.ctypes.cast(buffer, self.ctypes.POINTER(self.CounterItemType))
                return [
                    (items[i].szName, items[i].FmtValue.doubleValue)
                    for i in range(item_count.value)
                    if items[i].FmtValue.CStatus == 0
                ]
        return []

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "gpu_load_pct": 0.0,
            "gpu_freq_mhz": 0.0,
            "power_cur_mw": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_tot_mb": self.adapter_total_memory_mb,
        }
        res = self.pdh.PdhCollectQueryData(self.hQuery)
        if res != 0:
            return metrics

        # 1. GPU Engine Utilization
        eng_items = self._read_counter_items(self.hCounterEng)
        if eng_items:
            total_util = sum(val for _, val in eng_items)
            metrics["gpu_load_pct"] = round(max(0.0, min(100.0, total_util)), 2)

        # 2. GPU Dedicated Memory Usage (MB)
        if self.hCounterMem:
            mem_items = self._read_counter_items(self.hCounterMem)
            if mem_items:
                max_vram_bytes = max((val for _, val in mem_items), default=0.0)
                metrics["gpu_mem_used_mb"] = round(max_vram_bytes / (1024 * 1024), 2)

        # 3. Device Package Power (mW)
        if self.hCounterPwr:
            pwr_items = self._read_counter_items(self.hCounterPwr)
            if pwr_items:
                pkg_pwrs = [val for name, val in pwr_items if "pkg" in name.lower()]
                if pkg_pwrs:
                    metrics["power_cur_mw"] = round(sum(pkg_pwrs), 2)
                else:
                    # Fallback to total or sum of active core rails
                    non_zero = [val for name, val in pwr_items if val > 0 and "_total" not in name.lower()]
                    if non_zero:
                        metrics["power_cur_mw"] = round(sum(non_zero), 2)

        return metrics

    def close(self):
        try:
            self.pdh.PdhCloseQuery(self.hQuery)
        except Exception:
            pass


class LinuxSysfsGPUTracker(BaseGPUTracker):
    """Linux sysfs GPU tracker for AMD (/sys/class/drm/card*/device/gpu_busy_percent) and Intel."""

    def __init__(self, gpu_index: int = 0):
        super().__init__(gpu_index)
        self.busy_path = f"/sys/class/drm/card{gpu_index}/device/gpu_busy_percent"
        self.freq_path = f"/sys/class/drm/card{gpu_index}/gt_act_freq_mhz"  # Common on Intel
        if not os.path.exists(self.busy_path):
            alt_path = "/sys/class/drm/card0/device/gpu_busy_percent"
            if os.path.exists(alt_path):
                self.busy_path = alt_path
            else:
                raise FileNotFoundError(f"Sysfs path '{self.busy_path}' not found.")
        self.device_name = f"Linux DRM (card{gpu_index})"

    def get_metrics(self) -> Dict[str, Any]:
        metrics = {
            "gpu_load_pct": 0.0,
            "gpu_freq_mhz": 0.0,
            "power_cur_mw": 0.0,
            "gpu_mem_used_mb": 0.0,
            "gpu_mem_tot_mb": 0.0,
        }
        try:
            with open(self.busy_path, "r") as f:
                metrics["gpu_load_pct"] = float(f.read().strip())
        except Exception:
            pass

        if os.path.exists(self.freq_path):
            try:
                with open(self.freq_path, "r") as f:
                    metrics["gpu_freq_mhz"] = float(f.read().strip())
            except Exception:
                pass

        return metrics


class NullGPUTracker(BaseGPUTracker):
    """Fallback GPU tracker when no GPU backend is available."""

    def __init__(self, reason: str = "No supported GPU monitor detected"):
        super().__init__(0)
        self.device_name = f"None ({reason})"


def init_gpu_tracker(
    backend: str = "auto", gpu_index: int = 0, interval: float = 0.2
) -> BaseGPUTracker:
    """Initialize the most suitable GPU/platform tracker backend."""
    if backend == "none":
        return NullGPUTracker("Disabled by user")

    # 1. Jetson detection (highest priority on Jetson hardware, or if explicitly requested)
    if backend in ("auto", "jetson"):
        if backend == "jetson" or is_jetson():
            try:
                tracker = JetsonTracker(interval=interval)
                print(f"[+] Platform Backend: {tracker.device_name}")
                return tracker
            except Exception as e:
                if backend == "jetson":
                    print(f"Warning: Jetson tracker requested but failed: {e}", file=sys.stderr)
                    return NullGPUTracker(f"Jetson failed: {e}")
                print(f"[!] Jetson detected but jtop initialization failed: {e}", file=sys.stderr)
                print("[!] Falling back to standard Linux trackers (psutil + sysfs/NVML)...")

    # 2. If NVML requested or in auto mode, try NVML
    if backend in ("auto", "nvml"):
        try:
            tracker = NvmlGPUTracker(gpu_index=gpu_index)
            print(f"[+] GPU Backend: NVIDIA NVML ({tracker.device_name})")
            return tracker
        except Exception as e:
            if backend == "nvml":
                print(f"Warning: NVML requested but failed: {e}", file=sys.stderr)
                return NullGPUTracker(f"NVML failed: {e}")

    # 3. Windows PDH backend (DirectX / AMD / Intel)
    if backend in ("auto", "pdh") and sys.platform == "win32":
        try:
            tracker = WindowsPdhGPUTracker(gpu_index=gpu_index)
            print(f"[+] GPU Backend: Windows Performance Counters ({tracker.device_name})")
            return tracker
        except Exception as e:
            if backend == "pdh":
                print(f"Warning: Windows PDH requested but failed: {e}", file=sys.stderr)
                return NullGPUTracker(f"PDH failed: {e}")

    # 4. Linux sysfs backend (AMD / Intel)
    if backend in ("auto", "sysfs") and sys.platform.startswith("linux"):
        try:
            tracker = LinuxSysfsGPUTracker(gpu_index=gpu_index)
            print(f"[+] GPU Backend: Linux DRM Sysfs ({tracker.device_name})")
            return tracker
        except Exception as e:
            if backend == "sysfs":
                print(f"Warning: Linux Sysfs requested but failed: {e}", file=sys.stderr)
                return NullGPUTracker(f"Sysfs failed: {e}")

    print("[-] GPU Backend: None (GPU tracking will report 0.0%)")
    return NullGPUTracker("No supported GPU backend found")


# ==============================================================================
# Logger Engine
# ==============================================================================


def run_logger(
    output_path: str,
    interval: float,
    gpu_backend: str = "auto",
    gpu_index: int = 0,
):
    global stop_logging
    signal.signal(signal.SIGINT, handle_sigint)
    signal.signal(signal.SIGTERM, handle_sigint)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_sigint)

    # Initialize GPU/platform tracker
    gpu_tracker = init_gpu_tracker(backend=gpu_backend, gpu_index=gpu_index, interval=interval)
    is_jetson_tracker = isinstance(gpu_tracker, JetsonTracker)

    if not is_jetson_tracker:
        num_cores = psutil.cpu_count(logical=True) or 1
        print(f"[+] CPU Cores Detected: {num_cores} logical cores")
        # Prime initial psutil CPU counters
        psutil.cpu_times_percent(interval=None, percpu=True)

    print(f"[+] Sampling Interval: {interval}s")
    csv_file = open(output_path, mode="w", newline="", buffering=1)
    writer: Optional[csv.DictWriter] = None

    print(f"[+] Logging metrics to '{output_path}'. Press Ctrl+C to stop.")

    try:
        if is_jetson_tracker:
            # NVIDIA Jetson specialized sampling loop via jtop
            while not stop_logging and gpu_tracker.is_ok():
                t_start = time.time()
                snapshot = gpu_tracker.get_snapshot()

                row: Dict[str, Any] = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch_s": round(t_start, 4),
                    # Memory
                    "ram_used_mb": snapshot["ram_used_mb"],
                    "ram_tot_mb": snapshot["ram_tot_mb"],
                    "ram_pct": snapshot["ram_pct"],
                    # GPU
                    "gpu_load_pct": snapshot["gpu_load_pct"],
                    "gpu_freq_mhz": snapshot["gpu_freq_mhz"],
                    # Memory Controller (EMC)
                    "emc_pct": snapshot["emc_pct"],
                    # Total Module Power (mW)
                    "power_cur_mw": snapshot["power_cur_mw"],
                    "power_avg_mw": snapshot["power_avg_mw"],
                }

                # Dynamic per-core CPU load and frequency from Jetson stats
                for core_id, core_info in enumerate(snapshot["cpu_cores"]):
                    row[f"cpu{core_id}_load_pct_user"] = round(core_info["user"], 2)
                    row[f"cpu{core_id}_load_pct_system"] = round(core_info["system"], 2)
                    row[f"cpu{core_id}_freq_mhz"] = core_info["freq"]["cur"]

                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    writer.writeheader()

                writer.writerow(row)
                csv_file.flush()

                elapsed = time.time() - t_start
                sleep_duration = max(0.0, interval - elapsed)
                time.sleep(sleep_duration)

        else:
            # Generic PC / Server (Windows & Linux)
            while not stop_logging:
                t_start = time.time()

                # 1. Timestamp & Epoch
                row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "epoch_s": round(t_start, 4),
                }

                # 2. System RAM
                vmem = psutil.virtual_memory()
                row["ram_used_mb"] = vmem.used // (1024 * 1024)
                row["ram_tot_mb"] = vmem.total // (1024 * 1024)
                row["ram_pct"] = round(vmem.percent, 2)

                # 3. GPU Metrics
                gpu_metrics = gpu_tracker.get_metrics()
                row["gpu_load_pct"] = gpu_metrics["gpu_load_pct"]
                if "gpu_freq_mhz" in gpu_metrics:
                    row["gpu_freq_mhz"] = gpu_metrics["gpu_freq_mhz"]
                if "power_cur_mw" in gpu_metrics:
                    row["power_cur_mw"] = gpu_metrics["power_cur_mw"]
                if "gpu_mem_used_mb" in gpu_metrics:
                    row["gpu_mem_used_mb"] = gpu_metrics["gpu_mem_used_mb"]
                if "gpu_mem_tot_mb" in gpu_metrics:
                    row["gpu_mem_tot_mb"] = gpu_metrics["gpu_mem_tot_mb"]

                # 4. Per-Core CPU Load and Frequency
                cpu_times = psutil.cpu_times_percent(interval=None, percpu=True)
                freqs = psutil.cpu_freq(percpu=True)
                overall_freq = None
                if freqs and len(freqs) == 1 and freqs[0] and freqs[0].current > 0:
                    overall_freq = freqs[0].current
                else:
                    overall = psutil.cpu_freq()
                    if overall and overall.current > 0:
                        overall_freq = overall.current

                for core_id in range(num_cores):
                    # Core user & system loads
                    if core_id < len(cpu_times):
                        c_time = cpu_times[core_id]
                        row[f"cpu{core_id}_load_pct_user"] = round(c_time.user, 2)
                        row[f"cpu{core_id}_load_pct_system"] = round(c_time.system, 2)
                    else:
                        row[f"cpu{core_id}_load_pct_user"] = 0.0
                        row[f"cpu{core_id}_load_pct_system"] = 0.0

                    # Core frequency (MHz)
                    if freqs and len(freqs) == num_cores and freqs[core_id] and freqs[core_id].current > 0:
                        row[f"cpu{core_id}_freq_mhz"] = round(freqs[core_id].current, 2)
                    elif overall_freq is not None:
                        row[f"cpu{core_id}_freq_mhz"] = round(overall_freq, 2)
                    else:
                        row[f"cpu{core_id}_freq_mhz"] = 0.0

                # Initialize CSV header on first tick
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
                    writer.writeheader()

                writer.writerow(row)
                csv_file.flush()

                # Drift-compensated sleep
                elapsed = time.time() - t_start
                sleep_duration = max(0.0, interval - elapsed)
                time.sleep(sleep_duration)

    finally:
        gpu_tracker.close()
        csv_file.close()
        print(f"[+] Log successfully saved to {output_path}")


# ==============================================================================
# CLI Entrypoint
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Unified high-precision resource usage logger for Windows, Linux, and NVIDIA Jetson."
    )
    parser.add_argument(
        "-o",
        "--output",
        default="system_benchmark_metrics.csv",
        help="CSV output path (default: system_benchmark_metrics.csv)",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=float,
        default=0.5,
        help="Sampling interval in seconds (default: 0.5s)",
    )
    parser.add_argument(
        "--gpu-backend",
        choices=["auto", "jetson", "nvml", "pdh", "sysfs", "none"],
        default="auto",
        help="GPU/Platform monitoring backend (default: auto)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to track on multi-GPU systems (default: 0)",
    )

    args = parser.parse_args()
    run_logger(
        output_path=args.output,
        interval=args.interval,
        gpu_backend=args.gpu_backend,
        gpu_index=args.gpu_index,
    )


if __name__ == "__main__":
    main()
