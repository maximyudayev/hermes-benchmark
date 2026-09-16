import numpy as np
import h5py

from . import DataComponent


class MotorComponent(DataComponent):
    def __init__(
        self,
        unique_id: str,
        hdf5_path: str,
        data_path: str,
        legend_name: str,
        offset: float = 0.0,
    ):
        self._hdf5_path = hdf5_path
        self._data_path = data_path
        self._legend_name = legend_name
        self._offset = offset
        self._data: np.ndarray

        super().__init__(unique_id=unique_id)

    def read_data(self):
        self._read_timestamps()
        self._read_data()

    def _read_timestamps(self):
        with h5py.File(self._hdf5_path, "r") as hdf5:
            toa = hdf5[f"{self._data_path}/timestamp"][:]
            if toa.ndim > 1:
                toa = toa[:, 0]
            toa = toa.astype(np.float64) - self._offset
            valid = toa > 0
            self._toa_s = toa[valid] if np.any(valid) else toa
            self._first_timestamp = float(self._toa_s[0])
            self._last_timestamp = float(self._toa_s[-1])

    def _read_data(self):
        with h5py.File(self._hdf5_path, "r") as hdf5:
            data = hdf5[f"{self._data_path}/position"][:]
            if data.ndim > 1 and data.shape[1] == 1:
                data = data[:, 0]
            if hasattr(self, "_toa_s") and len(data) > len(self._toa_s):
                data = data[:len(self._toa_s)]
            self._data = data

    def get_sync_info(self):
        return {
            "first_timestamp": self._first_timestamp,
            "last_timestamp": self._last_timestamp,
            "timestamps": self._toa_s,
            "data": self._data,
        }
