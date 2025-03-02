import asyncio
from datetime import datetime

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from heinsight import HeinSight

# uvicorn stream:app --host 0.0.0.0 --port 8000


source = 0
FRAME_RATE = 5
DIRECTORY = None
FILENAME = None
heinsight = HeinSight(vial_model_path="models/best_vessel.pt",
                      contents_model_path="models/best_content.pt")

# Initialize FastAPI app
app = FastAPI()


# Placeholder for additional data
class FrameData(BaseModel):
    hsdata: list


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


@app.get("/start")
async def start_monitoring():
    """Endpoint to start monitoring."""
    global heinsight, is_monitoring, source
    if not is_monitoring:
        current_time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        filename = FILENAME or f"stream_{current_time}"
        heinsight.start_monitoring(source, save_directory=DIRECTORY, fps=FRAME_RATE, output_name=filename)
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
    await asyncio.sleep(1 / FRAME_RATE)
    return StreamingResponse(heinsight.generate_frame(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/data")
async def get_data():
    """Endpoint to return additional data."""
    if not is_monitoring:
        return JSONResponse(content={"error": "Monitoring is not active."}, status_code=400)
    frame_data = FrameData(hsdata=heinsight.output)
    return JSONResponse(content=frame_data.dict())
