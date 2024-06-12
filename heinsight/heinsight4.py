import datetime
import os
import threading

import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from heinsight.utils import colors
from ultralytics import YOLO

"""
2024-6-11:
generic HeinSight 
"""


class HeinSight:
    # RESIZE_VIAL = [86, 200]
    NUM_ROWS = -1  # number of rows that the vial is split into. -1 means each individual pixel row
    VISUALIZE = True  # watch in real time, if False it will only make a video without showing it
    INCLUDE_BB = True  # show bounding boxes in output video
    READ_EVERY = 5  # only considers every 'READ_EVERY' frame -> faster rendering
    UPDATE_EVERY = 5  # classifies vial contents ever 'UPDATE_EVERY' considered frame
    # turbidity paremeters, if #percentage of value greater than #threshold, then it is not dissolved
    THRESHOLD = 40
    PERCENTAGE = 50
    LIQUID_CONTENT = ["Homo", "Hetero"]

    def __init__(self, vial_model_path, contents_model_path, max_phase:int=2, solid_model_path=None, use_yolov8=True):
        self.phase_data = {}
        self.frame = None
        self._thread = None
        self._running = True
        self.vial_model = torch.hub.load('ultralytics/yolov5', 'custom', path=vial_model_path)
        self.contents_model = YOLO(contents_model_path)
        self.solid_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                          path=solid_model_path) if solid_model_path else None
        self.use_yolov8 = use_yolov8
        self.max_phase = max_phase
        self.vial_location = None
        self.vial_size = [80, 200]
        self.content_info = None
        self.monitor_range = [0, 1]
        self.settle_monitor_range = [0, 1]
        self.color_palette = colors.register_colors([self.contents_model, self.solid_model])
        self.x_time = []
        self.output_header = self._output_header()
        self.turbidity_2d = []
        self.average_colors = []
        self.average_turbidity = []
        self.output = []
        self.output_dataframe = pd.DataFrame()

    def _output_header(self):
        header = ["turbidity", "color"]
        for i in range(self.max_phase):
            header.extend([f"volume_{i + 1}", f"turbidity_{i + 1}", f"color_{i + 1}"])
        return header

    def draw_bounding_boxes(self, image, bboxes, class_names, thickness=2, text_right: bool = False):
        """Draws rectangles on the input image."""
        output_image = image.copy()
        for i, rect in enumerate(bboxes):
            color = self.color_palette.get(class_names[rect[-1]])
            x1, y1, x2, y2 = [int(x) for x in rect[:4]]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            text_location = (int((x2 - x1) * 0.7) + x1 if text_right else x1, y1 + 15)
            cv2.putText(output_image, class_names[rect[-1]], text_location, cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color, 2)
        return output_image

    def find_vial(self, frame, ):
        """
        find vial in video frame with YOLOv5
        :param frame:
        """
        self.vial_model.conf = 0.5
        self.vial_model.max_det = 1
        result = self.vial_model(frame)

        result = result.pred[0].cpu().numpy()
        if len(result) == 0:
            return None
        else:
            # vial_box = result.pred[0].cpu().numpy()[0, :4]  # if self.vial_model else result[0].cpu().numpy()
            self.vial_location = [x.astype(np.int16) for x in result[0, : 4]]
            self.vial_size = [int(self.vial_location[2] - self.vial_location[0]),
                              int(self.vial_location[3] - self.vial_location[1])]
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

    def content_detection(self, vial_frame):
        """
        detect content from one frame
        :param vial_frame: 
        :return: 
        """
        result = self.contents_model(vial_frame)
        if self.use_yolov8:
            bboxes = result[0].boxes.data.cpu().numpy()
        else:
            bboxes = result.pred[0].cpu().numpy()
        pred_classes = bboxes[:, 5]  # np.array: [1, 3 ,4]
        # pred_classes_names = [self.contents_model.names[x] for x in pred_classes]
        title = " ".join([self.contents_model.names[x] for x in pred_classes])
        # dissolved_from_liquid = "Homo" in pred_classes_names
        # has_residue = "Residue" in pred_classes_names
        index = self.find_liquid(pred_classes, self.LIQUID_CONTENT, self.contents_model.names)  # [1, 3]
        liquid_boxes = [bboxes[i][:4] for i in index]
        liquid_boxes = sorted(liquid_boxes, key=lambda x: x[1], reverse=True)
        return bboxes, liquid_boxes, title

    def process_vial_frame(self,
                           vial_frame,
                           update_od: bool = False,
                           ):
        # frame = self.crop_rectangle(frame, self.vial_location) if cropped else frame
        if update_od:
            self.content_info = self.content_detection(vial_frame)
        bboxes, liquid_boxes, title = self.content_info

        phase_data, raw_turbidity = self.calculate_value_color(seg=vial_frame, liquid_boxes=liquid_boxes)

        # this part gets ugly when theere is more than 1 l_bbox but for now good enough
        if self.INCLUDE_BB:
            frame = self.draw_bounding_boxes(vial_frame, bboxes, self.contents_model.names, text_right=True)
        self.frame = frame
        fig = self.display_frame(y_values=raw_turbidity, image=frame, title=title)
        fig.canvas.draw()
        frame_image = np.array(fig.canvas.renderer.buffer_rgba())
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
        # print(frame_image.shape) # this is 600x800
        return frame_image, raw_turbidity, phase_data

    def start_monitoring(self, video_path,  save_directory=None, output_name=None, fps=5, res=(1920, 1080)):
        if self._thread is None or not self._thread.is_alive():
            self._running = True
            self._thread = threading.Thread(target=self.run, args=(video_path, save_directory, output_name, fps, res))
            self._thread.daemon = True
            self._thread.start()
            print("Background task started.")
        else:
            print("Background task is already running.")

    def stop_monitor(self):
        if self._thread is not None and self._thread.is_alive():
            self._running = False
            self._thread.join()
            print("Background task stopped.")
        else:
            print("Background task is not running.")

    def run(self, video_path, save_directory=None, output_name=None, fps=5,
            res=(1920, 1080)):
        # first_frame = True
        realtime_cap = type(video_path) is int
        self.clear_cache()
        # prev_turbidities = None
        # ensure proper naming
        if output_name is None:
            output_name = "output"
        save_directory = save_directory or './heinsight_output'
        os.makedirs(save_directory, exist_ok=True)
        output_filename = os.path.join(save_directory, output_name)

        video = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Choose the appropriate codec
        if realtime_cap:
            video.set(cv2.CAP_PROP_FPS, fps)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            raw_video_writer = cv2.VideoWriter(f"{output_filename}_raw.mkv", fourcc, 30, res)

        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f"Capture fps: {fps}")

        video_writer = cv2.VideoWriter(output_filename + ".mkv", fourcc, 30, (800, 600))

        i = 0
        while self._running:
            for _ in range(self.READ_EVERY):
                ret, frame = video.read()
                if realtime_cap and ret:
                    raw_video_writer.write(frame)
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i == 0:
                while True:
                    result = self.find_vial(frame=frame)
                    if result is not None:
                        break
                    print("No vial found, re-detecting")
                    ret, frame = video.read()
                    if not ret:
                        break
                # print("using actual vial frame size", self.vial_size)

            vial_frame = self.crop_rectangle(image=frame, vial_location=self.vial_location)

            # every _th iteration update
            update_od = True if not i % self.UPDATE_EVERY else False
            # self.x_time.append(datetime.datetime.now() if realtime_cap else round(i * self.READ_EVERY / fps / 60, 3))
            self.x_time.append(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
            frame_image, _raw_turb, phase_data = self.process_vial_frame(vial_frame=vial_frame, update_od=update_od)
            self.phase_data = phase_data
            self.output.append(phase_data)

            self.turbidity_2d.append(_raw_turb)
            video_writer.write(frame_image)

            self.save_output(filename=output_filename)

            if self.VISUALIZE:
                cv2.imshow("Video", frame_image)
            if cv2.waitKey(1) == ord('q'):
                print("broke loop by pressing q")
                break
            # first_frame = False
            i += 1

        # Release the video file and output video and csv
        video.release()
        video_writer.release()
        if realtime_cap:
            raw_video_writer.release()
        cv2.destroyAllWindows()
        print(f"Results saved to {output_filename}")
        return self.output

    def display_frame(self, y_values, image, title=None):
        """
        Args:
            y_values:
            image:
            title:
        Returns: frame image
        """
        # create grid for different subplots
        liquid_top, liquid_bottom = self.monitor_range
        plt.close()
        fig, axs = plt.subplots(2, 2, figsize=(8, 6), height_ratios=[2, 1], constrained_layout=True)
        ax0, ax1, ax2, ax3 = axs.flat
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
        ax1.fill_between([self.THRESHOLD / 2, self.THRESHOLD], (1 - liquid_top) * len(y_values),
                         (1 - liquid_bottom) * len(y_values),
                         color="gray", alpha=0.3, label="Edge region")
        ax1.set_xlabel('Turbidity per row')
        ax1.set_position([0.47, 0.45, 0.45, 0.43])
        # realtime_tick_label = [self.x_time[0].strftime("%H:%M:%S"), self.x_time[-1].strftime("%H:%M:%S")] if type(
        #     self.x_time[-1]) is not float else None
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

    def calculate_value_color(self, seg, liquid_boxes):
        raw_value = []
        height, width, _ = seg.shape
        hsv_image = cv2.cvtColor(seg, cv2.COLOR_RGB2HSV)
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
        self.output_dataframe = pd.DataFrame(self.output)
        self.output_dataframe = self.output_dataframe.fillna(0)
        # combined_df = pd.concat([average_data, phase_data], axis=1)
        self.output_dataframe.to_csv(f"{filename}_per_phase.csv", index=False)

        # saving raw turbidity data
        turbidity_2d = np.array(self.turbidity_2d)
        turbidity_2d.T
        np.savetxt(filename + "_raw.csv", turbidity_2d, delimiter=',', fmt='%d')

    def crop_rectangle(self, image, vial_location):
        vial_x1, vial_y1, vial_x2, vial_y2 = vial_location
        cropped_image = image[vial_y1:vial_y2, vial_x1:vial_x2]
        # cv2.resize(cropped_image, self.vial_size)
        return cv2.resize(cropped_image, self.vial_size)

    def clear_cache(self):
        """
        clear all list when starting new process
        Returns: Null
        """
        self.x_time = []
        self.average_colors = []
        self.average_turbidity = []
        self.content_info = None
        self.turbidity_2d = []
        self.output = []


if __name__ == "__main__":
    heinsight = HeinSight(vial_model_path=r"models/labpic.pt",
                          contents_model_path=r"models/best_train5_yolov8_ez_20240402.pt", )
    output = heinsight.run(r"C:\Users\User\PycharmProjects\heinsight4.0\solid-liq-mixing.mp4")
    heinsight.run(r"C:\Users\User\PycharmProjects\heinsight4.0\solid-liq-mixing.mp4")
