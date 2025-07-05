import cv2


def get_camera_frame(camera_obj, source):
    """
    Fetches and processes a single frame from the specified camera object.
    :param camera_obj: The camera object used to capture the frame.
    :type camera_obj: An instance of PiCamera, cv2.VideoCapture, or equivalent
    :param source: The source type from which the image frame is to be retrieved.
                   Default is "webcam". Use "picam" for Pi cameras.
    :type source: str
    :return: A tuple where the first value indicates success (True/False)
             and the second value contains the captured frame or None
             in case of failure.
    :rtype: tuple[bool, Optional[numpy.ndarray]]
    """
    if source == "picam":
        try:
            frame = camera_obj.capture_array()
            if frame is None:
                return False, None

            # Handle different Pi camera formats
            if len(frame.shape) == 3 and frame.shape[2] > 3:
                frame = frame[:, :, :3]  # Remove alpha channel if present

            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            return True, frame
        except Exception:
            return False, None

    else:  # webcam or any cv2.VideoCapture
        try:
            ret, frame = camera_obj.read()
            return ret, frame
        except Exception:
            return False, None


def init_camera(source, res=(1920, 1080), fps=None, realtime_cap=True):
    if source == "picam":
        from picamera2 import Picamera2
        video = Picamera2()
        video.configure(video.create_video_configuration(main={"size": res}))
        video.start()
        return video, res, fps

    else:
        video = cv2.VideoCapture(source)
        fps = fps or video.get(cv2.CAP_PROP_FPS)
        if realtime_cap:
            video.set(cv2.CAP_PROP_FPS, fps)
            video.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            video.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        fps = int(video.get(cv2.CAP_PROP_FPS))
        return video, res, fps