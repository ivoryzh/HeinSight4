import asyncio
import datetime
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any

import yaml

sys.path.append(os.path.dirname(__file__))

import threading
from itertools import chain
from random import randint
import torch
from torchvision.ops import box_iou
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS
matplotlib.use('Agg')

if __package__ is None or __package__ == "":
    import utils
else:
    from . import utils

class HeinSightConfig:

    """Configuration for the HeinSight system."""
    NUM_ROWS = -1
    VISUALIZE = True
    INCLUDE_BB = True
    SAVE_PLOT_VIDEO = True
    READ_EVERY = 1
    UPDATE_EVERY = 5
    LIQUID_CONTENT = ["Homo", "Hetero"]
    CAP_RATIO = 0
    STATUS_RULE = 0.7
    NMS_RULES = {
        ("Homo", "Hetero"): 0.2,
        ("Hetero", "Residue"): 0.2,
        ("Solid", "Residue"): 0.2,
        ("Empty", "Residue"): 0.2,
    }
    DEFAULT_VIAL_LOCATION = None
    DEFAULT_VIAL_HEIGHT = None
    DEFAULT_FPS = 30
    DEFAULT_RESOLUTION = (1920, 1080)

    DEFAULT_OUTPUT_DIR = './heinsight_output'
    DEFAULT_OUTPUT_NAME = 'output'
    STREAM_DATA_SIZE = 100


    def load_from_file(self, config_path: str):
        """Load configuration from JSON or YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        if config_path.suffix.lower() == '.json':
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        elif config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")

        self.update_from_dict(config_data)

    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary."""
        # Map configuration keys to attributes
        config_mapping = {
            'num_rows': 'NUM_ROWS',
            'visualize': 'VISUALIZE',
            'include_bb': 'INCLUDE_BB',
            'save_plot_video': 'SAVE_PLOT_VIDEO',
            'read_every': 'READ_EVERY',
            'update_every': 'UPDATE_EVERY',
            'liquid_content': 'LIQUID_CONTENT',
            'cap_ratio': 'CAP_RATIO',
            'status_rule': 'STATUS_RULE',
            'default_vial_location': 'DEFAULT_VIAL_LOCATION',
            'default_vial_height': 'DEFAULT_VIAL_HEIGHT',
            'default_fps': 'DEFAULT_FPS',
            'default_resolution': 'DEFAULT_RESOLUTION',
            'default_output_dir': 'DEFAULT_OUTPUT_DIR',
            'default_output_name': 'DEFAULT_OUTPUT_NAME',
            'stream_data_size': 'STREAM_DATA_SIZE',
        }

        for key, value in config_dict.items():
            if key in config_mapping:
                setattr(self, config_mapping[key], value)
            elif key == 'nms_rules':
                # Handle NMS rules which have tuple keys
                nms_rules = {}
                for rule_key, rule_value in value.items():
                    if isinstance(rule_key, str):
                        # Convert string representation to tuple
                        tuple_key = eval(rule_key)  # e.g., "('Homo', 'Hetero')" -> ('Homo', 'Hetero')
                        nms_rules[tuple_key] = rule_value
                    else:
                        nms_rules[rule_key] = rule_value
                self.NMS_RULES = nms_rules
            else:
                print(f"Warning: Unknown configuration key: {key}")

class HeinSight:
    """
    The core of the HeinSight system, responsible for computer vision and analysis.
    """

    def __init__(self, vial_model_path: str, contents_model_path: str, config: HeinSightConfig = HeinSightConfig()):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[2, 1], constrained_layout=True)
        self._set_axes()
        self.config = config
        self.vial_model = YOLO(vial_model_path)
        self.contents_model = YOLO(contents_model_path)
        self.color_palette = self._register_colors([self.vial_model, self.contents_model])
        self._thread = None
        self._running = False
        self.clear_cache()

    def _set_axes(self):
        """creating plot axes"""
        ax0, ax1, ax2, ax3 = self.axs.flat
        ax0.set_position([0.21, 0.45, 0.22, 0.43])  # [left, bottom, width, height]

        ax1.set_position([0.47, 0.45, 0.45, 0.43])  # [left, bottom, width, height]
        ax2.set_position([0.12, 0.12, 0.35, 0.27])
        ax3.set_position([0.56, 0.12, 0.35, 0.27])
        self.fig.canvas.draw_idle()

    def clear_cache(self):
        """Resets the state of the HeinSight system."""
        self.vial_location = self.config.DEFAULT_VIAL_LOCATION.copy() if self.config.DEFAULT_VIAL_LOCATION else None
        self.cap_rows = 0
        self.vial_heigh = self.config.DEFAULT_VIAL_HEIGHT
        self.vial_size = []
        self.content_info = None
        self.x_time = []
        self.turbidity_2d = []
        self.average_colors = []
        self.average_turbidity = []
        self.output = []
        self.stream_output = []
        self.status = {}
        self.status_queue = []
        self.output_dataframe = pd.DataFrame()
        self.output_frame = None
        self.turbidity = []

    @staticmethod
    def _register_colors(model_list):
        """
        register default colors for models
        :param model_list: YOLO models list
        """
        name_color_dict = {
            "Empty": (19, 69, 139),  # Brown
            "Residue": (0, 165, 255),  # Orange
            "Hetero": (255, 0, 255),  # purple
            "Homo": (0, 0, 255),  # Red
            "Solid": (255, 0, 0),  # Blue
        }
        names = set(chain.from_iterable(model.names.values() for model in model_list if model))
        for name in names:
            if name not in name_color_dict:
                name_color_dict[name] = (randint(0, 255), randint(0, 255), randint(0, 255))
        return name_color_dict

    def find_vial(self, frame):
        """
        Detect the vial in video frame with YOLOv8
        :param frame: raw input frame
        :return result: np.ndarray or None: Detected vial bounding box or None if no vial is found.
        """
        # vial location is not defined, use vial model to detect
        if not self.vial_location:
            results = self.vial_model(frame, conf=0.2, max_det=1)
            boxes = results[0].boxes.data.cpu().numpy()
            if boxes.size > 0:
                self.vial_location = [int(x) for x in boxes[0, :4]]
        if self.vial_location:
            self.cap_rows = int((self.vial_location[3] - self.vial_location[1]) * self.config.CAP_RATIO)
        return self.vial_location is not None

    def crop_rectangle(self, image, vial_location):
        """
        crop and resize the image
        :param image: raw image capture
        :param vial_location:
        :return: cropped and resized vial frame
        """
        x1, y1, x2, y2 = vial_location
        y1 = int(self.config.CAP_RATIO * (y2 - y1)) + y1
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image

    def content_detection(self, vial_frame):
        """
        Detect content in a vial frame.
        :param vial_frame: (np.ndarray) Cropped vial frame.
        :return tuple: Bounding boxes, liquid boxes, and detected class titles.
        """
        results = self.contents_model(vial_frame, max_det=4, agnostic_nms=False, conf=0.25, iou=0.25, verbose=False)
        bboxes = self.custom_nms(results[0].boxes.data.cpu().numpy())
        pred_classes = bboxes[:, 5]
        self.average_status(pred_classes)
        title = " ".join([self.contents_model.names[int(x)] for x in pred_classes])
        liquid_boxes = [bboxes[i][:4] for i, cls in enumerate(pred_classes) if
                        self.contents_model.names[int(cls)] in self.config.LIQUID_CONTENT]
        return bboxes, sorted(liquid_boxes, key=lambda x: x[1], reverse=True), title

    def custom_nms(self, bboxes):
        """
        Apply custom NMS based on class overlap rules.

        :param bboxes: Detected bounding boxes (numpy array: [x1, y1, x2, y2, conf, class_id]).
        :return: Filtered bounding boxes.
        """
        keep_indices = []
        bboxes = torch.tensor(bboxes)
        classes = [self.contents_model.names[int(idx)] for idx in bboxes[:, 5]]

        confidences = bboxes[:, 4]

        for i, bbox in enumerate(bboxes):

            suppress = False
            for j, other_bbox in enumerate(bboxes):
                if i == j:
                    continue

                iou = box_iou(bbox[:4].unsqueeze(0), other_bbox[:4].unsqueeze(0)).item()
                iou_thresholds = self.config.NMS_RULES.get((classes[i], classes[j]), None)
                if iou_thresholds and iou > iou_thresholds:
                    suppress = confidences[i] < confidences[j]

                if suppress:
                    break

            if not suppress:
                keep_indices.append(i)

        return bboxes[keep_indices].numpy()

    def process_vial_frame(self, vial_frame, update_od: bool = False):
        """
        process single vial frame, detect content, draw bounding box and calculate turbidity and color
        :param vial_frame: vial frame image
        :param update_od: update object detection, True: run YOLO for this frame, False: use previous YOLO results
        """
        if update_od or self.content_info is None:
            self.content_info = self.content_detection(vial_frame)
        bboxes, liquid_boxes, title = self.content_info
        phase_data, raw_turbidity = self.calculate_value_color(vial_frame, liquid_boxes)
        frame_image = self.draw_bounding_boxes(vial_frame, bboxes, self.contents_model.names, text_right=False)

        if self.config.SAVE_PLOT_VIDEO:
            self.display_frame(raw_turbidity, frame_image, title)
            self.fig.canvas.draw()
            frame_image = np.array(self.fig.canvas.renderer.buffer_rgba())
            frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
        return frame_image, bboxes, raw_turbidity, phase_data

    def calculate_value_color(self, vial_frame, liquid_boxes):
        """
        Calculate the value and color for a given vial image and bounding boxes
        :param vial_frame: the vial image
        :param liquid_boxes: the liquid boxes (["Homo", "Hetero"])
        :return: the output dict and raw turbidity per row
        """
        height, _, _ = vial_frame.shape
        hsv_image = cv2.cvtColor(vial_frame, cv2.COLOR_BGR2HSV)
        output = {
            'time': self.x_time[-1],
            'color': np.mean(hsv_image[:, :, 0]),
            'turbidity': np.mean(hsv_image[:, :, 2])
        }
        raw_value = np.mean(hsv_image[:, :, 2], axis=1)
        for i, bbox in enumerate(liquid_boxes):
            _, top, _, bottom = map(int, bbox)
            roi = hsv_image[top:bottom, :]
            output[f'volume_{i + 1}'] = (bottom - top) / height
            output[f'color_{i + 1}'] = np.mean(roi[:, :, 0])
            output[f'turbidity_{i + 1}'] = np.mean(roi[:, :, 2])
        self.average_colors.append(output['color'])
        self.average_turbidity.append(output['turbidity'])
        return output, raw_value

    @staticmethod
    def _get_dynamic_font_params(img_height, base_height=200, base_font_scale=1, base_thickness=2):
        scale_factor = img_height / base_height
        font_scale = base_font_scale * scale_factor
        text_thickness = max(2, int(base_thickness * scale_factor))
        return font_scale, text_thickness

    def draw_bounding_boxes(self, image, bboxes, class_names, thickness=None, text_right=False, on_raw=False):
        """Draws bounding boxes on the image."""
        output_image = image.copy()
        height = image.shape[1]
        font_scale, text_thickness = self._get_dynamic_font_params(height)
        margin = 2
        thickness = thickness or max(2, int(height / 200))
        for rect in bboxes:
            x1, y1, x2, y2, _, class_id = map(int, rect)
            class_name = class_names[class_id]
            color = self.color_palette.get(class_name, (255, 255, 255))
            if on_raw and self.vial_location:
                x1, y1 = x1 + self.vial_location[0], y1 + self.vial_location[1] + self.cap_rows
                x2, y2 = x2 + self.vial_location[0], y2 + self.vial_location[1] + self.cap_rows
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                                                                  text_thickness)
            text_location = (
                x2 - text_width - margin if text_right ^ (class_name == "Solid") else x1 + margin,
                y1 + text_height + margin
            )
            cv2.putText(output_image, class_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color,
                        text_thickness)
        return output_image

    def display_frame(self, y_values, image, title=None):
        """
        Display the image (top-left) and its turbidity values per row (top-right)
        turbidity over time (bottom-left) and color over time (bottom-right)
        :param y_values: the turbidity value per row
        :param image: vial image frame to display
        :param title: title of the image frame
        """
        # init plot
        for ax in self.axs.flat:
            ax.clear()
        ax0, ax1, ax2, ax3 = self.axs.flat

        # top left - vial frame and bounding boxes
        image_copy = image.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
        ax0.imshow(np.flipud(image_copy), origin='lower')
        if title:
            ax0.set_title(title)

        # use fill between to optimize the speed 154.9857677 -> 68.15193
        x_values = np.arange(len(y_values))
        ax1.fill_betweenx(x_values, 0, y_values[::-1], color='green', alpha=0.5)
        ax1.set_ylim(0, len(y_values))
        ax1.set_xlim(0, 255)
        ax1.xaxis.set_label_position('top')
        ax1.set_xlabel('Turbidity per row')

        realtime_tick_label = None

        # bottom left - turbidity
        ax2.set_ylabel('Turbidity')
        ax2.set_xlabel('Time / min')
        ax2.plot(self.x_time, self.average_turbidity)
        ax2.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)



        # bottom right - color
        ax3.set_ylabel('Color (hue)')
        ax3.set_xlabel('Time / min')
        ax3.plot(self.x_time, self.average_colors)
        ax3.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)


    def average_status(self, pred_classes):
        """Averages the status of the vial over time."""
        current_status = {name: (i in pred_classes) for i, name in self.contents_model.names.items()}
        self.status_queue.append(current_status)
        if len(self.status_queue) > 10:
            self.status_queue.pop(0)
        true_counts = defaultdict(int)
        for status_item in self.status_queue:
            for key, value in status_item.items():
                if value:
                    true_counts[key] += 1
        self.status = {key: (count / len(self.status_queue)) >= self.config.STATUS_RULE for key, count in
                       true_counts.items()}

    def start_monitoring(self, video_source, **kwargs):
        """
        Starts monitoring for the given video source in a separate thread.

        :param video_source: Specifies the source of the video input. It can either be a file path
            or an index representing a camera stream.
        :param kwargs: Optional keyword arguments providing additional configurations required by the
            `run` method.
        :return: None
        """
        if self._thread and self._thread.is_alive():
            print("Monitoring is already running.")
            return
        self._running = True
        self._thread = threading.Thread(target=self.run, args=(video_source,), kwargs=kwargs)
        self._thread.daemon = True
        self._thread.start()
        print("Monitoring started.")

    def stop_monitor(self):
        """
        heinsight GUI function: stop monitoring
        :return: None
        """
        if self._thread and self._thread.is_alive():
            self._running = False
            self._thread.join()
            print("Monitoring stopped.")
        else:
            print("Background task is not running.")

    async def generate_frame(self):
        while True:
            try:
                frame = self.output_frame
                if frame is None:
                    break

                # Convert to RGB! Encode frame as JPEG
                # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Yield frame bytes with correct header
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                await asyncio.sleep(0.01)  # Small delay to avoid high CPU usage
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Client disconnected: {e}")
                break  # Exit the loop when the client disconnects


    def run(self, source, save_directory=None, output_name=None, fps=30,
            res=(1920, 1080), live_save: bool = False):
        """
        Main function to perform vial monitoring. Captures video frames from a camera or video file,
        Workflow:
        Image analysis mode:
        1. Load image 2. Detect the vial 3. Detect content 4 Save output frame as image.
        Video analysis mode:
        1. Initialize video capture (Pi Camera, webcam, or video/image file).
        2. Initialize output video write and optionally initialize video writers for saving raw frames.
        3. Detect the vial in the first frame or as needed.
        4. Process each frame:
            - Crop the vial area.
            - Perform content detection and calculate vial properties (turbidity, phase data).
            - Save processed frames and plots to video.
        5. Optionally display processed frames in real-time.
        6. Save the output data to .CSV files.
        7. Handle cleanup and resource release on completion or interruption.
        Raises:
            KeyboardInterrupt: Stops the monitoring loop when manually interrupted.

        :param source: image/video/capture
        :param save_directory: output directory, defaults to "./heinsight_output"
        :param output_name: output name, defaults to "output"
        :param fps: FPS, defaults to 5
        :param res: (realtime capturing) resolution, defaults to (1920, 1080)
        :param live_save: whether to save csv data after every frame
        :return: output over time dictionary
        """
        # ensure proper naming
        self.clear_cache()

        save_directory = save_directory or self.config.DEFAULT_OUTPUT_DIR
        output_name = output_name or self.config.DEFAULT_OUTPUT_NAME
        os.makedirs(save_directory, exist_ok=True)
        output_filename = os.path.join(save_directory, output_name)

        image_mode = isinstance(source, str) and source.lower().endswith(tuple(IMG_FORMATS))
        if image_mode:

            frame = cv2.imread(source)
            if self.find_vial(frame):
                vial_frame = self.crop_rectangle(frame, self.vial_location)
                self.x_time.append(0)
                frame_image, bboxes, _, phase_data = self.process_vial_frame(vial_frame)
                bboxes_on_raw = self.draw_bounding_boxes(frame, bboxes, self.contents_model.names, on_raw=True)
                cv2.imwrite(f"{output_filename}.png", frame_image)
                cv2.imwrite(f"{output_filename}_raw.png", bboxes_on_raw)
                print(phase_data)
            else:
                print("No vial found.")
            return

        realtime_cap = type(source) is int or source == "picam"
        video, res, fps = utils.init_camera(source, res, fps, realtime_cap)

        # 2. Setup video writers for saving outputs
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        res = (800, 600) if self.config.SAVE_PLOT_VIDEO else res or self.config.DEFAULT_RESOLUTION
        video_writer = cv2.VideoWriter(f"{output_filename}.mkv", fourcc, fps, res)
        if realtime_cap:
            raw_video_writer = cv2.VideoWriter(f"{output_filename}_raw.mkv", fourcc, 30, res)

        # video capturing and analysis
        frame_count = 0
        self._running = True
        while self._running:
            ret, frame = utils.get_camera_frame(video, source)
            if not ret:
                break

            if realtime_cap:
                raw_video_writer.write(frame)

            if frame_count % self.config.READ_EVERY != 0:
                frame_count += 1
                continue

            # 3. Detect the vial in the first frame or as needed.
            if not self.vial_location:
                self.find_vial(frame)

            while self._running and not self.vial_location:
                ret, frame = utils.get_camera_frame(video, source)
                if not ret:
                    break
                self.find_vial(frame)
                if self.vial_location:
                    cv2.destroyAllWindows()
                elif self.config.VISUALIZE:
                    cv2.putText(frame, "Searching for vessel...",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow('Detection', frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("broke loop by pressing q")
                        self._running = False
                        break
                else:
                    print("Waiting for vial...")


            # 4. Process each frame
            if self.vial_location:
                vial_frame = self.crop_rectangle(frame, self.vial_location)
                current_time = datetime.datetime.now().isoformat()  # .strftime('%Y-%m-%d %H:%M:%S')
                self.x_time.append(current_time if realtime_cap else round(frame_count * self.config.READ_EVERY / fps / 60, 5))
                update_od = frame_count % self.config.UPDATE_EVERY == 0
                frame_image, bboxes, raw_turbidity, phase_data = self.process_vial_frame(vial_frame, update_od)
                self.output.append(phase_data)
                self.turbidity_2d.append(raw_turbidity)
                self.output_frame = frame_image
                self.optimize_stream_output()
                raw_frame = self.draw_bounding_boxes(frame, bboxes, self.contents_model.names, on_raw=True)


                # 5. Optionally display processed frames in real-time.
                if self.config.VISUALIZE:
                    cv2.imshow("HeinSight", frame_image if self.config.SAVE_PLOT_VIDEO else raw_frame)
                    # 5.1. Record keystrokes during the analysis, in case of manual real time logging
                    key = cv2.waitKey(1) & 0xFF  # Get key pressed
                    if key == ord('q'):
                        self.stop_monitor()
                        print("broke loop by pressing q")
                        break
                    phase_data["key pressed"] = '' if key == 255 else chr(key)
                    if key != 255:  # 255 is returned if no key is pressed
                        print(f"Key pressed: {chr(key)}")

                # 6. Save the output data
                # Save the processed frame to video file
                if live_save:
                    self.save_output(output_filename)
                if self.config.SAVE_PLOT_VIDEO:
                    video_writer.write(frame_image)
                else:
                    video_writer.write(raw_frame)
            frame_count += 1


        # 7. Handle cleanup and resource release on completion or interruption.
        video.release()
        video_writer.release()
        cv2.destroyAllWindows()

    def optimize_stream_output(self):
        data_length = len(self.output)
        max_points = self.config.STREAM_DATA_SIZE
        # Determine the indices to sample from
        if data_length > max_points:
            step = data_length // max_points + 1  # Ensure even spacing
            self.stream_output = self.output[::step][:max_points]
        else:
            self.stream_output = self.output  # Use all data if less than 1000

    def save_output(self, filename):
        """Saves the output data to CSV files."""
        pd.DataFrame(self.output).to_csv(f"{filename}_per_phase.csv", index=False)
        np.savetxt(f"{filename}_raw.csv", np.array(self.turbidity_2d), delimiter=',')




if __name__ == "__main__":
    heinsight = HeinSight(vial_model_path=r"models/best_vessel.pt",
                          contents_model_path=r"models/best_content.pt", )
    output = heinsight.run(r"../examples/demo.png")


