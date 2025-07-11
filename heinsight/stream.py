import argparse
import asyncio
import importlib
from datetime import datetime
from typing import Union, Tuple

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel, Field

if __package__ is None or __package__ == "":
    from heinsight import HeinSight, HeinSightConfig
else:
    from .heinsight import HeinSight, HeinSightConfig


# uvicorn stream:app --host 0.0.0.0 --port 8000

REFRESH_RATE = 20

# Initialize FastAPI app
app = FastAPI()
# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class StartMonitoringRequest(BaseModel):
    """Request model for starting monitoring"""
    video_source: Union[str, int] = Field(..., description="Video source (file path, camera index, or 'picam')")
    frame_rate: int = Field(30, description="Frame rate for processing")
    res: Tuple[int, int] = Field((1920, 1080), description="Video resolution")

# Placeholder for additional data
class FrameData(BaseModel):
    hsdata: list


class StatusData(BaseModel):
    data: dict
    status: dict


is_monitoring = False


@app.on_event("startup")
async def startup():
    print("App started.")


@app.on_event("shutdown")
async def shutdown():
    global heinsight
    if is_monitoring:
        heinsight.stop_monitor()
        print("Camera stopped.")


@app.post("/start")
async def start_monitoring(request: StartMonitoringRequest):
    """Endpoint to start monitoring."""
    global heinsight, is_monitoring, FRAME_RATE

    video_source = request.video_source
    fps = request.frame_rate
    res = request.res

    if video_source is None:
        return JSONResponse(content={"message": "Video source is required."}, status_code=400)
    fps = fps or 20
    if not is_monitoring:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = heinsight.config.DEFAULT_OUTPUT_NAME or f"stream_{video_source}_{current_time}"
        heinsight.start_monitoring(video_source, res=res, fps=fps,
                                   save_directory=heinsight.config.DEFAULT_OUTPUT_DIR, output_name=filename)
        is_monitoring = True
        return JSONResponse(content={"message": "Monitoring started."})
    else:
        return JSONResponse(content={"message": "Monitoring is already running."}, status_code=400)


@app.get("/stop")
async def stop_monitoring():
    """Endpoint to stop monitoring."""
    global heinsight, is_monitoring
    if is_monitoring:
        heinsight.stop_monitor()
        is_monitoring = False
        return JSONResponse(content={"message": "Monitoring stopped."})
    else:
        return JSONResponse(content={"message": "Monitoring is not running."}, status_code=400)


@app.get("/frame")
async def get_frame():
    """Endpoint to stream video frames."""
    if not is_monitoring:
        return JSONResponse(content={"error": "Monitoring is not active."}, status_code=400)
    await asyncio.sleep(1 / REFRESH_RATE)
    return StreamingResponse(heinsight.generate_frame(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/data")
async def get_data():
    """Endpoint to return additional data."""
    if not is_monitoring:
        return JSONResponse(content={"error": "Monitoring is not active."}, status_code=400)
    frame_data = FrameData(hsdata=heinsight.stream_output)
    return JSONResponse(content=frame_data.model_dump())

@app.get("/rolling_data")
async def get_rolling_data():
    """Endpoint to return additional data."""
    if not is_monitoring:
        return JSONResponse(content={"error": "Monitoring is not active."}, status_code=400)
    if len(heinsight.output) < 10:
        frame_data = FrameData(hsdata=heinsight.output)
    else:
        frame_data = FrameData(hsdata=heinsight.output[-10:])
    return JSONResponse(content=frame_data.model_dump())

@app.get("/current_status")
async def get_last_status():
    """Endpoint to return additional data."""
    if not is_monitoring:
        return JSONResponse(content={"error": "Monitoring is not active."}, status_code=400)
    if len(heinsight.output) == 0:
        return JSONResponse(content={"error": "No data available."}, status_code=400)
    status_data = StatusData(status=heinsight.status, data=heinsight.output[-1])
    # print(status_data.dict())
    return JSONResponse(content=status_data.model_dump())

def main():
    """
    Main function to run the FastAPI server.
    Initializes HeinSight with default or user-provided models.
    """
    global heinsight
    parser = argparse.ArgumentParser(description="HeinSight FastAPI Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--vial-model", type=str, default=None, help="Path to a custom vessel detection model.")
    parser.add_argument("--contents-model", type=str, default=None, help="Path to a custom contents detection model.")

    # Configuration file
    parser.add_argument("--config", type=str, default=None, help="Path to configuration file (JSON or YAML)")

    # Individual configuration parameters (override config file)
    parser.add_argument("--save-plot-video", type=bool, default=None, help="Save plot video")
    parser.add_argument("--read-every", type=int, default=None, help="Read every N frames")
    parser.add_argument("--update-every", type=int, default=None, help="Update every N frames")
    parser.add_argument("--cap-ratio", type=float, default=None, help="Cap ratio threshold")
    parser.add_argument("--default-fps", type=int, default=None, help="Default FPS")
    parser.add_argument("--stream-data-size", type=int, default=None, help="Stream data size")
    parser.add_argument("--save-directory", type=str, default=None, help="Path to save videos.")

    args = parser.parse_args()

    vial_model_path = args.vial_model
    contents_model_path = args.contents_model
    config = HeinSightConfig()
    config.VISUALIZE = False

    if args.config:
        print(f"Loading configuration from: {args.config}")
        config.load_from_file(args.config)

        # Override with command line arguments
        if args.save_plot_video is not None:
            config.SAVE_PLOT_VIDEO = args.save_plot_video
        if args.read_every is not None:
            config.READ_EVERY = args.read_every
        if args.update_every is not None:
            config.UPDATE_EVERY = args.update_every
        if args.cap_ratio is not None:
            config.CAP_RATIO = args.cap_ratio
        if args.default_fps is not None:
            config.DEFAULT_FPS = args.default_fps
        if args.stream_data_size is not None:
            config.STREAM_DATA_SIZE = args.stream_data_size
        if args.save_directory:
            config.DEFAULT_OUTPUT_DIR = args.save_directory

    try:
        # If user does not provide a path, load the default model from the package
        if not vial_model_path or not contents_model_path:
            print("Loading default models...")
            if __package__ is None or __package__ == "":
                vial_model_path="models/best_vessel.pt"
                contents_model_path="models/best_content.pt"
            else:
                with importlib.resources.path('heinsight.models', 'best_vessel.pt') as vessel_path, \
                        importlib.resources.path('heinsight.models', 'best_content.pt') as content_path:
                    vial_model_path = vial_model_path or str(vessel_path)
                    contents_model_path = contents_model_path or str(content_path)

        else:
            print("Loading custom models from provided paths...")
        print(f"Initializing with vial model: {vial_model_path}")
        print(f"Initializing with contents model: {contents_model_path}")
        # Initialize HeinSight with the determined model paths
        heinsight = HeinSight(
            vial_model_path=vial_model_path,
            contents_model_path=contents_model_path,
            config=config
        )

        uvicorn.run(app, host=args.host, port=args.port)

    except FileNotFoundError:
        print("Error: A specified model file was not found. Please check the paths.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()