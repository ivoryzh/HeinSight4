import os
import sys
import threading
import time
from typing import Optional, Dict, Any, List, Union

# Ensure we can import heinsight
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from heinsight import HeinSight, HeinSightConfig
except ImportError:
    # Fallback if running from within the package
    from .heinsight import HeinSight, HeinSightConfig


class HeinsightAPI:
    def __init__(self, address: str = None, source: Union[int, str] = None, fps: int = 10, res: tuple = (1920, 1080)):
        """
        Direct implementation of HeinsightAPI that uses the HeinSight class directly
        instead of making HTTP requests.
        
        :param address: Ignored, kept for compatibility.
        :param source: Video source (int for camera index, str for file path).
        :param fps: Frames per second.
        :param res: Resolution tuple (width, height).
        """
        self.address = address  # Kept for compatibility but unused
        self.source = source
        self.fps = fps
        self.res = res
        self.running = False
        
        # Initialize HeinSight instance
        # Assuming models are in the 'models' directory relative to this script or current working directory
        # We might need to adjust paths if they are not found
        base_dir = os.path.dirname(os.path.abspath(__file__))
        vial_model_path = os.path.join(base_dir, "models", "best_vessel.pt")
        contents_model_path = os.path.join(base_dir, "models", "best_content.pt")
        
        # Check if models exist, otherwise try default relative paths
        if not os.path.exists(vial_model_path):
            vial_model_path = "models/best_vessel.pt"
        if not os.path.exists(contents_model_path):
            contents_model_path = "models/best_content.pt"
            
        self.heinsight = HeinSight(
            vial_model_path=vial_model_path,
            contents_model_path=contents_model_path
        )

    def start(self, source: Union[int, str] = None, res: tuple = None):
        """
        Starts the monitoring process.
        
        :param source: Optional video source override.
        :param res: Optional resolution override.
        """
        if self.running:
            return {"status": "already running"}
        
        if source is not None:
            self.source = source
        if res is not None:
            self.res = res
            
        self.running = True
        # start_monitoring runs in a separate thread
        self.heinsight.start_monitoring(self.source, fps=self.fps, res=self.res)
        return {"status": "started"}

    def stop(self):
        """Stops the monitoring process."""
        self.running = False
        self.heinsight.stop_monitor()
        return {"status": "stopped"}

    def data(self):
        """Returns the accumulated data."""
        # The original API returns a list of data points
        # heinsight.output is a list of dicts
        return {"data": self.heinsight.output}

    def get_current_status(self):
        """Returns the current status and latest data point."""
        # Construct response similar to the API
        latest_data = self.heinsight.output[-1] if self.heinsight.output else {}
        
        return {
            "status": self.heinsight.status,
            "data": latest_data
        }

    def homo(self):
        return self._get_status("Homo")

    def hetero(self):
        return self._get_status("Hetero")

    def empty(self):
        return self._get_status("Empty")

    def residue(self):
        return self._get_status("Residue")

    def solid(self):
        return self._get_status("Solid")

    def turbidity(self, rolling_average: int = 1):
        return self._get_data("turbidity", rolling_average)

    def turbidity_1(self, rolling_average: int = 1):
        return self._get_data("turbidity_1", rolling_average)

    def turbidity_2(self, rolling_average: int = 1):
        return self._get_data("turbidity_2", rolling_average)

    def volume_1(self, rolling_average: int = 1):
        return self._get_data("volume_1", rolling_average)

    def volume_2(self, rolling_average: int = 1):
        return self._get_data("volume_2", rolling_average)

    def _get_data(self, data_class, rolling_average):
        if rolling_average == 1 or rolling_average == 0 or rolling_average is None or rolling_average is False:
            # Get latest data
            if not self.heinsight.output:
                return None
            data = self.heinsight.output[-1]
            return data.get(data_class, None)
        else:
            # Get rolling average
            if not self.heinsight.output:
                return None
            
            # The original API gets 'hsdata' from 'rolling_data' endpoint
            # Here we access self.heinsight.output directly
            data = self.heinsight.output
            last_data = data[-rolling_average:] if len(data) > rolling_average else data
            
            data_queue = []
            for i in last_data:
                val = i.get(data_class, False)
                if val is not False and val is not None:
                    data_queue.append(val)
            
            if len(data_queue) == 0:
                return None
            else:
                return sum(data_queue) / len(data_queue)

    def _get_status(self, hs_class):
        # self.heinsight.status is a dict {class_name: boolean}
        return self.heinsight.status.get(hs_class, False)


