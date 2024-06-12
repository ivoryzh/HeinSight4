import logging
import os, sys
import random
import time

sys.path.append(os.getcwd())

# from component.plc import PLC
from heinsight.heinsight4 import HeinSight

logger = logging.getLogger('gui_loggoer')


class HeinSightRobot:
    DEBUG = True
    CAMERA_PORT = 0
    AVERAGE_VISION = 5
    REACTOR_VOLUME = 10

    def __init__(self):

        # self.plc = PseudoPLC()
        self.hein_sight = HeinSight(vial_model_path=r"C:\Users\User\PycharmProjects\heinsight4.0\heinsight\models\labpic.pt",
                                    contents_model_path=r"C:\Users\User\PycharmProjects\heinsight4.0\heinsight\models\best_train5_yolov8_ez_20240402.pt", )
        self.PARAM = self.hein_sight.output_header
        # self.hein_sight.start_monitoring(r"C:\Users\User\PycharmProjects\heinsight4.0\solid-liq-mixing.mp4")

    def monitor(self, camera_port: int = 0):
        self.hein_sight.start_monitoring(camera_port)

    def monitor_stop(self):
        self.hein_sight.stop_monitor()

    def wait(self, seconds: int):
        logger.info(f"Waiting for {seconds} seconds...")
        time.sleep(seconds)

    def pump(self, speed: float, duration: int = None):
        if True:
            print("Start pumping...")
            if duration:
                time.sleep(duration)
                self.pump_stop()

    def stir(self, speed: float, duration: int = None):
        pass

    def stir_stop(self, ):
        pass

    def vacuum(self, ):
        pass

    def vacuum_stop(self, ):
        pass

    def pump_stop(self):
        print("Stop pumping")


    def heat(self, target_temp: int, duration: int = None):
        print(f'Heating to {target_temp}...')
        if True:
            pass
            if duration:
                time.sleep(duration)

    def heat_stop(self):
        print("Stop heating")
        if True:
            # self.plc.heat_stop()
            pass

    def _do_until(self, action, args={}, statement={}, all_conditions: bool = False):
        """

        :param action:
        :param args:
        :param statement:
        :param all_conditions: meet all conditions or one condition
        :return:
        """
        # print(action.__module__, action.__class__)
        action_name = action.__name__
        stop_cue = action_name + "_stop"
        try:
            stop_action = getattr(self, stop_cue)
        except Exception:
            raise ValueError("Cannot schedule 'do_until': no stop action.")
        action(**args)
        while not self._is_complete(statement, all_conditions):
            time.sleep(1)
            print('condition is not satisfied')
        stop_action()

    def _is_complete(self, param: dict, all_conditions: bool = True):
        """
        :param param: see param_example for dictionary template
        :return: if conditions are satisfied
        """

        sat = {}
        for key, target_value in param.items():
            # target_volume = param['volume1']
            current_value = self.hein_sight.output[-1].get(key, None) if self.hein_sight.output else None
            sat[key] = eval(str(current_value * self.REACTOR_VOLUME) + target_value) if current_value else False
            logger.info(f"current {key}: {current_value}")
        return list(sat.values()).count(False) == 0 if all_conditions else list(sat.values()).count(True) > 0

    def _get_current_temp(self):
        """
        placeholder for getting temperature info from PLC
        :return: float: current temperature
        """
        return 300

    def _get_stir_rate(self):
        """
        placeholder for getting stir rate info from PLC
        :return: float: current stir rate
        """
        return 300


if __name__ == "__main__":
    hein_sight_robot = HeinSightRobot()
    print(hein_sight_robot)
    from ivoryos_vision.app import start_gui

    start_gui(hein_sight_robot)
    # example usage of do_until scheduler,
    # pump until random generated volume is >= 9
    # hein_sight._do_until(hein_sight.pump, 5, statement={"volume": ">= 8"})
    # hein_sight.add
