import asyncio
import datetime
import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

import threading
from itertools import chain
from random import randint
import torch
from torchvision.ops import box_iou
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS


class HeinSight:
    NUM_ROWS = -1  # number of rows that the vial is split into. -1 means each individual pixel row
    VISUALIZE = True  # watch in real time, if False it will only make a video without showing it
    INCLUDE_BB = True  # show bounding boxes in output video
    READ_EVERY = 5  # only considers every 'READ_EVERY' frame -> faster rendering
    UPDATE_EVERY = 5  # classifies vial contents ever 'UPDATE_EVERY' considered frame
    LIQUID_CONTENT = ["Homo", "Hetero"]
    CAP_RATIO = 0.3  # this is the cap ratio of a HPLC vial
    NMS_RULES = {
        ("Homo", "Hetero"): 0.2,        # Suppress lower confidence when Homo overlaps with Hetero
        ("Hetero", "Residue"): 0.2,     # Suppress lower confidence when Hetero overlaps with Residue
        ("Solid", "Residue"): 0.2,      # Suppress lower confidence when Solid overlaps with Residue
        ("Empty", "Residue"): 0.2,      # Suppress lower confidence when Empty overlaps with Residue
    }

    def __init__(self, vial_model_path, contents_model_path):
        """
        Initialize the HeinSight system.
        """
        self._thread = None
        self._running = True
        self.vial_model = YOLO(vial_model_path)
        self.contents_model = YOLO(contents_model_path)
        self.vial_location = None
        self.vial_size = [80, 200]
        self.content_info = None
        self.color_palette = self._register_colors([self.contents_model])
        self.x_time = []
        self.turbidity_2d = []
        self.average_colors = []
        self.average_turbidity = []
        self.output = []
        self.output_dataframe = pd.DataFrame()
        self.output_frame = None

    def draw_bounding_boxes(self, image, bboxes, class_names, thickness=None, text_right: bool = False):
        """Draws rectangles on the input image."""
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        output_image = image.copy()
        # change thickness and font scale based on the vial frame height
        thickness = thickness or max(1, int(self.vial_size[1] / 200))
        margin = thickness * 2
        text_thickness = max(1, min(int((thickness + 1) / 2), 3))  # (1, 3)
        fontscale = min(0.5 * thickness, 2)  # (0.5, 2)

        for i, rect in enumerate(bboxes):
            class_name = class_names[rect[-1]]
            color = self.color_palette.get(class_name)
            x1, y1, x2, y2 = [int(x) for x in rect[:4]]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            (text_width, text_height), baseline = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, fontscale,
                                                                  text_thickness)
            text_location = (
                x2 - text_width - margin if text_right ^ (class_name == "Solid") else x1 + margin,
                y1 + text_height + margin
            )
            cv2.putText(output_image, class_name, text_location, cv2.FONT_HERSHEY_SIMPLEX, fontscale, color,
                        text_thickness)
        return output_image

    def find_vial(self, frame, ):
        """
        Detect the vial in video frame with YOLOv8
        :param frame: raw input frame
        :return result: np.ndarray or None: Detected vial bounding box or None if no vial is found.
        """
        self.vial_model.conf = 0.5
        self.vial_model.max_det = 1
        result = self.vial_model(frame)
        result = result[0].boxes.data.cpu().numpy()
        if len(result) == 0:
            return None
        else:
            # vial_box = result.pred[0].cpu().numpy()[0, :4]  # if self.vial_model else result[0].cpu().numpy()
            self.vial_location = [x.astype(np.int16) for x in result[0, : 4]]
            self.vial_size = [
                int(self.vial_location[2] - self.vial_location[0]),
                int((self.vial_location[3] - self.vial_location[1]) * (1 - self.CAP_RATIO))
            ]
        return result

    @staticmethod
    def find_liquid(pred_classes, liquid_classes, all_classes):
        """
        find the first id of either 'Homo' or 'Hetero' from the class names
        :param pred_classes: [1, 2, 3]
        :param liquid_classes: ['Homo', 'Hetero']
        :param all_classes: {0: solid, 1: No solid, ....}
        :return: [index]
        """
        liquid_classes_id = [key for key, value in all_classes.items() if value in liquid_classes]
        return [index for index, c in enumerate(pred_classes) if c in liquid_classes_id]

    # NMS function for post procesing suppresion

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
                iou_thresholds = self.NMS_RULES.get((classes[i], classes[j]), None)
                if iou_thresholds and iou > iou_thresholds:
                    suppress = confidences[i] < confidences[j]

                if suppress:
                    break

            if not suppress:
                keep_indices.append(i)

        return bboxes[keep_indices].numpy()



    def content_detection(self, vial_frame):
        """
        Detect content in a vial frame.
        :param vial_frame: (np.ndarray) Cropped vial frame.
        :return tuple: Bounding boxes, liquid boxes, and detected class titles.
        """
        result = self.contents_model(vial_frame, max_det=4, agnostic_nms=False, conf=0.25, iou=0.25)
        bboxes = result[0].boxes.data.cpu().numpy()

        # Apply custom NMS
        bboxes = self.custom_nms(bboxes)

        pred_classes = bboxes[:, 5]  # np.array: [1, 3 ,4]
        title = " ".join([self.contents_model.names[x] for x in pred_classes])

        index = self.find_liquid(pred_classes, self.LIQUID_CONTENT, self.contents_model.names)  # [1, 3]
        liquid_boxes = [bboxes[i][:4] for i in index]
        liquid_boxes = sorted(liquid_boxes, key=lambda x: x[1], reverse=True)
        return bboxes, liquid_boxes, title

    def process_vial_frame(self,
                           vial_frame,
                           update_od: bool = False,
                           ):
        """
        process single vial frame, detect content, draw bounding box and calculate turbidity and color
        :param vial_frame: vial frame image
        :param update_od: update object detection, True: run YOLO for this frame, False: use previous YOLO results
        """
        if update_od:
            self.content_info = self.content_detection(vial_frame)
        bboxes, liquid_boxes, title = self.content_info

        phase_data, raw_turbidity = self.calculate_value_color(vial_frame=vial_frame, liquid_boxes=liquid_boxes)

        # this part gets ugly when there is more than 1 l_bbox but for now good enough
        if self.INCLUDE_BB:
            frame = self.draw_bounding_boxes(vial_frame, bboxes, self.contents_model.names, text_right=False)
        # self.frame = frame
        fig = self.display_frame(y_values=raw_turbidity, image=frame, title=title)

        fig.canvas.draw()
        frame_image = np.array(fig.canvas.renderer.buffer_rgba())
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)

        # print(frame_image.shape) # this is 600x800
        return frame_image, raw_turbidity, phase_data

    def start_monitoring(self, video_source, save_directory=None, output_name=None, fps=5, res=(1920, 1080)):
        """
        heinsight GUI function: starting monitoring, same param as run()
        :return: None
        """
        self.VISUALIZE = False
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.run, args=(video_source, save_directory, output_name, fps, res))
            self._thread.daemon = True
            self._thread.start()
            print("Background task started.")
        else:
            print("Background task is already running.")

    def stop_monitor(self):
        """
        heinsight GUI function: stop monitoring
        :return: None
        """
        if self._thread is not None and self._thread.is_alive():
            self._running = False
            self._thread.join()
            print("Background task stopped.")
        else:
            print("Background task is not running.")

    async def generate_frame(self):
        while True:
            try:
                frame = self.output_frame
                if frame is None:
                    break

                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                # Yield frame bytes with correct header
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                await asyncio.sleep(0.01)  # Small delay to avoid high CPU usage
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"Client disconnected: {e}")
                break  # Exit the loop when the client disconnects

    def display_frame(self, y_values, image, title=None):
        """
        Display the image (top-left) and its turbidity values per row (top-right)
        turbidity over time (bottom-left) and color over time (bottom-right)
        :param y_values: the turbidity value per row
        :param image: vial image frame to display
        :param title: title of the image frame
        """
        # create grid for different subplots
        plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[2, 1], constrained_layout=True)
        ax0, ax1, ax2, ax3 = axs.flat

        # top left - vial frame and bounding boxes
        ax0.imshow(np.flipud(image), origin='lower')
        if title:
            ax0.set_title(title)
        ax0.set_position([0.21, 0.45, 0.22, 0.43])  # [left, bottom, width, height]

        # top right - Turbidity per row
        bar_width = 1
        x_values = range(len(y_values))
        ax1.barh(x_values, np.flip(y_values), orientation='horizontal', height=bar_width, color='green', alpha=0.5)
        ax1.set_ylim(0, len(y_values))
        ax1.set_xlim(0, 255)
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        ax1.set_xlabel('Turbidity per row')
        ax1.set_position([0.47, 0.45, 0.45, 0.43])

        realtime_tick_label = None

        # bottom left - turbidity
        ax2.set_ylabel('Turbidity')
        ax2.set_xlabel('Time / min')
        ax2.plot(self.x_time, self.average_turbidity)
        ax2.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)
        ax2.set_position([0.12, 0.12, 0.35, 0.27])

        # bottom right - color
        ax3.set_ylabel('Color (hue)')
        ax3.set_xlabel('Time / min')
        ax3.plot(self.x_time, self.average_colors)
        ax3.set_xticks([self.x_time[0], self.x_time[-1]], realtime_tick_label)
        ax3.set_position([0.56, 0.12, 0.35, 0.27])
        return fig

    def calculate_value_color(self, vial_frame, liquid_boxes):
        """
        Calculate the value and color for a given vial image and bounding boxes
        :param vial_frame: the vial image
        :param liquid_boxes: the liquid boxes (["Homo", "Hetero"])
        :return: the output dict and raw turbidity per row
        """
        raw_value = []
        height, width, _ = vial_frame.shape
        hsv_image = cv2.cvtColor(vial_frame, cv2.COLOR_RGB2HSV)
        average_color = np.mean(hsv_image[:, :, 0])
        average_value = np.mean(hsv_image[:, :, 2])
        self.average_colors.append(average_color)
        self.average_turbidity.append(average_value)
        output = dict(time=self.x_time[-1], color=average_color, turbidity=average_value)
        for i in range(height):
            # Calculate the starting and ending indices for the row
            row = hsv_image[i, :]
            average_value = np.mean(row[:, 2])
            raw_value.append(average_value)
        for index, bbox in enumerate(liquid_boxes):
            # print(bbox)
            _, liquid_top, _, liquid_bottom = bbox
            start_index = int(liquid_top)
            end_index = int(liquid_bottom)
            row = hsv_image[start_index:end_index, :]
            value = np.mean(row[:, :, 2])
            color = np.mean(row[:, :, 0])
            output[f'volume_{index + 1}'] = (liquid_bottom - liquid_top) / height
            output[f'color_{index + 1}'] = color
            output[f'turbidity_{index + 1}'] = value
            # output.append(output_per_box)
        return output, raw_value

    def save_output(self, filename):
        """
        Save the output to _per_phase.csv and _raw.csv
        :param filename: base filename
        :return: None
        """
        self.output_dataframe = pd.DataFrame(self.output)
        self.output_dataframe = self.output_dataframe.fillna(0)
        # combined_df = pd.concat([average_data, phase_data], axis=1)
        self.output_dataframe.to_csv(f"{filename}_per_phase.csv", index=False)

        # saving raw turbidity data
        turbidity_2d = np.array(self.turbidity_2d)
        turbidity_2d.T
        np.savetxt(filename + "_raw.csv", turbidity_2d, delimiter=',', fmt='%d')

    def crop_rectangle(self, image, vial_location):
        """
        crop and resize the image
        :param image: raw image capture
        :param vial_location:
        :return: cropped and resized vial frame
        """
        vial_x1, vial_y1, vial_x2, vial_y2 = vial_location
        vial_y1 = int(self.CAP_RATIO * (vial_y2 - vial_y1)) + vial_y1
        cropped_image = image[vial_y1:vial_y2, vial_x1:vial_x2]
        # cv2.resize(cropped_image, self.vial_size)
        return cv2.resize(cropped_image, self.vial_size)

    def clear_cache(self):
        """
        clear all list when starting new process
        :return: None
        """
        self.x_time = []
        self.average_colors = []
        self.average_turbidity = []
        self.content_info = None
        self.turbidity_2d = []
        self.output = []

    def run(self, source, save_directory=None, output_name=None, fps=5,
            res=(1920, 1080)):
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
        :return: output over time dictionary
        """
        # ensure proper naming
        output_name = output_name or "output"
        save_directory = save_directory or './heinsight_output'
        os.makedirs(save_directory, exist_ok=True)
        output_filename = os.path.join(save_directory, output_name)

        image_mode = type(source) is str and source.split(".")[-1] in IMG_FORMATS
        if image_mode:
            frame = cv2.imread(source)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.find_vial(frame=frame)
            if result is not None:
                vial_frame = self.crop_rectangle(image=frame, vial_location=self.vial_location)
                self.x_time = [0]
                frame_image, _raw_turb, phase_data = self.process_vial_frame(vial_frame=vial_frame, update_od=True)
                print(phase_data)
                cv2.imwrite(f"{output_filename}.png", frame_image)
            else:
                print("No vial found")
            return None
        else:
            realtime_cap = type(source) is int or source == "picam"

            # clear history
            self.clear_cache()

            # 1. Initialize video capture
            # TODO change fps
            if source == "picam":
                from picamera2 import Picamera2
                camera = Picamera2()
                camera.configure(camera.create_video_configuration(main={"size": res}))
                camera.start()
            else:
                video = cv2.VideoCapture(source)
                if realtime_cap:
                    video.set(cv2.CAP_PROP_FPS, fps)
                    video.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
                    video.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
                fps = int(video.get(cv2.CAP_PROP_FPS))
                # print(f"Capture fps: {fps}")

            # 2. Setup video writers for saving outputs
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Choose the appropriate codec
            video_writer = cv2.VideoWriter(output_filename + ".mkv", fourcc, 30, (800, 600))
            if realtime_cap:
                raw_video_writer = cv2.VideoWriter(f"{output_filename}_raw.mkv", fourcc, 30, res)

            # video capturing and analysis
            i = 0
            # try:
            while self._running:
                # Capture and process frames, skip frames if not READ_EVERY = 1
                for _ in range(self.READ_EVERY):
                    if source == "picam":
                        frame = camera.capture_array()
                        if frame is not None:
                            frame = frame[:, :, :3]  # Remove any unnecessary channels
                        else:
                            break
                    else:
                        ret, frame = video.read()
                        if not ret:
                            break
                    if realtime_cap:
                        raw_video_writer.write(frame)

                if frame is None:
                    break

                # 3. Detect the vial in the first frame or as needed.
                if i == 0:
                    while True:
                        result = self.find_vial(frame=frame)
                        if result is not None:
                            break
                        print("No vial found, re-detecting")
                        time.sleep(1)
                        if source == "picam":
                            frame = camera.capture_array()
                            if frame is not None:
                                frame = frame[:, :, :3]  # Remove any unnecessary channels
                        else:
                            ret, frame = video.read()
                            if not ret:
                                break

                # 4. Process each frame
                vial_frame = self.crop_rectangle(image=frame, vial_location=self.vial_location)
                update_od = True if not i % self.UPDATE_EVERY else False  # every _th iteration update
                current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.x_time.append(current_time if realtime_cap else round(i * self.READ_EVERY / fps / 60, 3))
                frame_image, _raw_turb, phase_data = self.process_vial_frame(vial_frame=vial_frame, update_od=update_od)
                self.output_frame = frame_image

                # 5. Optionally display processed frames in real-time.
                if self.VISUALIZE:
                    cv2.imshow("Video", frame_image)

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
                video_writer.write(frame_image)
                self.output.append(phase_data)
                self.turbidity_2d.append(_raw_turb)
                self.save_output(filename=output_filename)
                i += 1

            # 7. Handle cleanup and resource release on completion or interruption.
            # except KeyboardInterrupt:
            #     print("Monitoring stopped manually.")
            # finally:
            if source == "picam":
                camera.stop()
            else:
                video.release()
            if realtime_cap:
                raw_video_writer.release()
            video_writer.release()  # Ensure video is saved
            cv2.destroyAllWindows()
            print(f"Results saved to {output_filename}")
            return self.output

    @staticmethod
    def _register_colors(model_list):
        """
        register default colors for models
        :param model_list: YOLO models list
        """
        name_color_dict = {
            "Empty": (160, 82, 45),  # Brown
            "Residue": (255, 165, 0),  # Orange
            "Hetero": (255, 0, 255),  # purple
            "Homo": (255, 0, 0),  # Red
            "Solid": (0, 0, 255),  # Blue
        }
        names = [model.names.values() for model in model_list if model is not None]
        names = set(chain.from_iterable(names))
        for index, name in enumerate(names):
            if name not in name_color_dict.keys():
                name_color_dict[name] = (randint(0, 255), randint(0, 255), randint(0, 255))
        return name_color_dict


if __name__ == "__main__":
    heinsight = HeinSight(vial_model_path=r"models/best_vial_20250108.pt",
                          contents_model_path=r"models/best_content_200250109.pt", )
    output = heinsight.run(r"../examples/demo.png")
    # heinsight.run("C:\Users\User\Downloads\demo.png")
    # heinsight.run(r"C:\Users\User\Downloads\WIN_20240620_11_28_09_Pro.mp4")

