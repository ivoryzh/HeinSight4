import datetime
import os
import torch
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from heinsight.utils import colors
from ultralytics import YOLO

"""
2024-4-11 update:
1. added decision making for settling
"""


class HeinSight:
    RESIZE_VIAL = [86, 200]
    NUM_ROWS = -1  # number of rows that the vial is split into. -1 means each individual pixel row
    VISUALIZE = True  # watch in real time, if False it will only make a video without showing it
    INCLUDE_BB = True  # show bounding boxes in output video
    READ_EVERY = 5  # only considers every 'READ_EVERY' frame -> faster rendering
    UPDATE_EVERY = 5  # classifies vial contents ever 'UPDATE_EVERY' considered frame
    # turbidity paremeters, if #percentage of value greater than #threshold, then it is not dissolved
    THRESHOLD = 40
    PERCENTAGE = 50
    LIQUID_CONTENT = ["Homo", "Hetero"]

    def __init__(self, vial_model_path, contents_model_path, solid_model_path=None, use_yolov8=True):
        self.vial_model = torch.hub.load('ultralytics/yolov5', 'custom', path=vial_model_path)
        self.contents_model = YOLO(contents_model_path)
        self.solid_model = torch.hub.load('ultralytics/yolov5', 'custom',
                                          path=solid_model_path) if solid_model_path else None
        self.use_yolov8 = use_yolov8
        # self.setting_yolo_parameters()
        self.vial_location = None
        self.content_info = None
        self.monitor_range = [0, 1]
        self.settle_monitor_range = [0, 1]
        self.color_palette = colors.register_colors([self.contents_model, self.solid_model])
        self.x_time = []
        self.turbidity_2d = []
        self.average_colors = []
        self.average_turbidity = []
        self.output = []

    def draw_bounding_boxes(self, image, bboxes, class_names, thickness=2, text_right: bool = False):
        """
        Draws rectangles on the input image.
        Parameters:
            image (numpy.ndarray): The input image.
            bboxes (list): A list of rectangles in the form of [(x1, y1, x2, y2), (x1, y1, x2, y2), ...].
            class_names ():
            thickness (int): The thickness of the rectangle edges. Default is 2.
            text_right (bool):
        Returns:
            numpy.ndarray: The image with rectangles drawn on it.
        """
        output_image = image.copy()
        for i, rect in enumerate(bboxes):
            color = self.color_palette.get(class_names[rect[-1]])
            x1, y1, x2, y2 = [int(x) for x in rect[:4]]
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, thickness)
            text_location = (int((x2 - x1) * 0.7) + x1 if text_right else x1, y1 + 15)
            cv2.putText(output_image, class_names[rect[-1]], text_location, cv2.FONT_HERSHEY_SIMPLEX, 2,
                        color, 2)
        return output_image

    def find_vial(self, frame):
        """
        find vial in video frame
        Args:
            frame: video frame
        Returns:
            vial location: x1, y1, x2, y2
        """
        self.vial_model.conf = 0.5
        self.vial_model.max_det = 1
        result = self.vial_model(frame)
        # vial_box = result.pred[0].cpu().numpy()[0, :4]  # if self.vial_model else result[0].cpu().numpy()
        return result.pred[0].cpu().numpy()

    @staticmethod
    def find_liquid(pred_classes, liquid_classes, all_classes):
        """
        find the first id of either 'Homo' or 'Hetero' from the class names
        :param pred_classes: [1, 2, 3]
        :param liquid_classes: ['Homo', 'Hetero']
        :param all_classes: {0: solid, 1: No solid, ....}
        :return: index
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
        # turbidities, color = self.calculate_average_value(seg=vial_frame, num_rows=self.NUM_ROWS, content_only=False)
        # process liquid contents
        phase_data, avg_turbidity, avg_color, raw_turbidity = self.calculate_value_color(seg=vial_frame,
                                                                                         liquid_boxes=liquid_boxes)
        self.average_colors.append(avg_color)
        self.average_turbidity.append(avg_turbidity)
        # this part gets ugly when theere is more than 1 l_bbox but for now good enough
        if self.INCLUDE_BB:
            frame = self.draw_bounding_boxes(vial_frame, bboxes, self.contents_model.names, thickness=2,
                                             text_right=True)
        fig = self.display_frame(y_values=raw_turbidity, image=frame, title=title)
        fig.canvas.draw()
        frame_image = np.array(fig.canvas.renderer.buffer_rgba())
        frame_image = cv2.cvtColor(frame_image, cv2.COLOR_RGBA2BGR)
        # print(frame_image.shape) # this is 600x800
        return frame_image, raw_turbidity, avg_turbidity, avg_color, phase_data

    def process_one(self, video_path, blank_image_path=None, save_directory=None, output_name=None):
        first_frame = True
        self.clear_cache()
        prev_turbidities = None
        # ensure proper naming
        if not output_name:
            output_name = "output"
        save_directory = save_directory or './heinsight_output'
        os.makedirs(save_directory, exist_ok=True)
        output_filename = f'{save_directory}/{output_name}'

        video = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # Choose the appropriate codec
        if type(video_path) is int:
            video.set(cv2.CAP_PROP_FPS, 1)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            raw_video_writer = cv2.VideoWriter(f"{output_filename}_raw.mkv", fourcc, 30, (1920, 1080))

        fps = int(video.get(cv2.CAP_PROP_FPS))
        print(f"Capture fps: {fps}")

        video_writer = cv2.VideoWriter(output_filename + ".mkv", fourcc, fps, (800, 600))

        i = 0
        while True:
            for _ in range(self.READ_EVERY):
                ret, frame = video.read()
                if type(video_path) is int and ret:
                    raw_video_writer.write(frame)
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if i == 0:
                # get vial_bb for entire run

                result = self.find_vial(frame=frame)
                while len(result) == 0:
                    print("No vial found, re-detecting")
                    ret, frame = video.read()
                    if not ret:
                        break
                    result = self.find_vial(frame=frame)
                # cv2.imwrite("frame.jpg", frame)
                self.vial_location = [x.astype(np.int16) for x in result[0, : 4]]
                self.RESIZE_VIAL = [int(self.vial_location[2] - self.vial_location[0]),
                                    int(self.vial_location[3] - self.vial_location[1])]
                print("using actual vial frame size", self.RESIZE_VIAL)

            vial_frame = self.crop_rectangle(image=frame, vial_location=self.vial_location)
            vial_frame = cv2.resize(vial_frame, self.RESIZE_VIAL)

            # every _th iteration update
            update_od = True if not i % self.UPDATE_EVERY else False
            # self.x_time.append(round(i * self.READ_EVERY / fps / 60, 3))
            self.x_time.append(
                datetime.datetime.now() if type(video_path) is int else round(i * self.READ_EVERY / fps / 60, 3))
            frame_image, turbidities, turbidity_avg, color_avg, output_list = self.process_vial_frame(
                vial_frame=vial_frame,
                update_od=update_od,
            )
            self.output.append(output_list)

            self.turbidity_2d.append(turbidities)
            video_writer.write(frame_image)

            self.save_output(filename=output_filename)

            if self.VISUALIZE:
                cv2.imshow("Video", frame_image)
            if cv2.waitKey(1) == ord('q'):
                print("broke loop by pressing q")
                break
            first_frame = False
            i += 1

        # Release the video file and output video and csv
        video.release()
        video_writer.release()
        if type(video_path) is int:
            raw_video_writer.release()
        cv2.destroyAllWindows()
        print(f"Results saved to {output_filename}")

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
        realtime_tick_label = [self.x_time[0].strftime("%H:%M:%S"), self.x_time[-1].strftime("%H:%M:%S")] if type(
            self.x_time[-1]) is not float else None

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

    @staticmethod
    def calculate_value_color(seg, liquid_boxes):
        raw_value = []
        height, width, _ = seg.shape
        output = []
        hsv_image = cv2.cvtColor(seg, cv2.COLOR_RGB2HSV)
        average_color = np.mean(hsv_image[:, :, 2])
        for i in range(height):
            # Calculate the starting and ending indices for the row
            row = hsv_image[i, :]
            average_value = np.mean(row[:, 2])
            raw_value.append(average_value)
        for bbox in liquid_boxes:
            # print(bbox)
            _, liquid_top, _, liquid_bottom = bbox
            start_index = int(liquid_top)
            end_index = int(liquid_bottom)
            row = hsv_image[start_index:end_index, :]
            value = np.mean(row[:, :, 2])
            color = np.mean(row[:, :, 0])
            output_per_box = dict(volume=(liquid_bottom - liquid_top) / height, color=color, turbidity=value)
            output.append(output_per_box)
        return output, average_value, average_color, raw_value

    def save_output(self, filename):
        max_len = max(len(sublist) for sublist in self.output)
        average_data = pd.DataFrame({
            "Time": self.x_time,
            "color": self.average_colors,
            "turbidity": self.average_turbidity,
        })
        # Create CSV headers based on the maximum length
        headers = ["Time", "Average Color", "Average Turbidity"]
        rows = []
        for i in range(max_len):
            headers.extend([f"volume_{i + 1}", f"turbidity_{i + 1}", f"color_{i + 1}"])
        for sublist in self.output:
            row = {}
            for i, item in enumerate(sublist):
                for key, value in item.items():
                    row[f"{key}_{i + 1}"] = value
            rows.append(row)
        phase_data = pd.DataFrame(rows)
        phase_data = phase_data.fillna(0)
        combined_df = pd.concat([average_data, phase_data], axis=1)
        combined_df.to_csv(f"{filename}_per_phase.csv", index=False)

        # saving raw turbidity data
        turbidity_2d = np.array(self.turbidity_2d)
        turbidity_2d.T
        np.savetxt(filename + "_raw.csv", turbidity_2d, delimiter=',', fmt='%d')

    @staticmethod
    def crop_rectangle(image, vial_location):
        vial_x1, vial_y1, vial_x2, vial_y2 = vial_location
        cropped_image = image[vial_y1:vial_y2, vial_x1:vial_x2]
        return cropped_image

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

    heinsight.process_one(r"C:\Users\User\PycharmProjects\heinsight4.0\solid-liq-mixing.mp4")
