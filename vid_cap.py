import asyncio
# from opcua_rds import ReactorDeviceOPCUA
import cv2
import time
import os

# easy_max = ReactorDeviceOPCUA()
# easy_max.connect()
easy_max = None

REACTOR = 1


def capture_camera_feed(before_stir, stir_time, after_stir, output_filename="output"):
    # Check for existing output file and avoid overwriting
    base_filename = output_filename
    counter = 1
    output_filename = f"{base_filename}_{counter}.avi"
    while os.path.exists(output_filename):
        counter += 1
        output_filename = f"{base_filename}_{counter}.avi"

    # Open the video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 30.0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    end_time = start_time + (before_stir + stir_time + after_stir) * 60
    stop_stir = (before_stir + stir_time) * 60
    start_stir = before_stir * 60
    stirring = False

    print("Starting capture ...")

    while time.time() < end_time:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            print("Error: Unable to read frame.")
            break

        # Check for the first action timing
        if start_stir <= time.time() - start_time < stop_stir and not stirring:
            print("start stirring.")
            stirring = True
            if easy_max:
                asyncio.run(easy_max.change_stir_rate(reactor=REACTOR, rate=1000, duration=1))

        # Check for the second action timing
        if time.time() - start_time >= stop_stir and stirring:
            print("stop stirring.")
            stirring = False
            if easy_max:
                asyncio.run(easy_max.change_stir_rate(reactor=REACTOR, rate=0, duration=1))

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved as {output_filename}")


if __name__ == "__main__":
    # Example usage:
    capture_camera_feed(0.3, 0.3, 0.4)
