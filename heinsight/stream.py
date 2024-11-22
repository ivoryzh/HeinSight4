import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from rpi_heinsight4 import HeinSight

# Initialize FastAPI app
app = FastAPI()


# Placeholder for additional data
class FrameData(BaseModel):
    hsdata: list

FRAME_RATE = 5
heinsight = HeinSight(vial_model_path="/home/heinsight/heinsight2.5/heinsight/models/labpic.pt",
                      contents_model_path="/home/heinsight/heinsight2.5/heinsight/models/best_train5_yolov8_ez_20240402.pt")


@app.on_event("startup")
async def startup():
    global heinsight
    heinsight.start_monitoring("picam", output_name="steam_test")
    print("Camera started.")

@app.on_event("shutdown")
async def shutdown():
    global heinsight
    heinsight.stop_monitor()
    print("Camera stopped.")

@app.get("/frame")
async def get_frame():
    """Endpoint to stream video frames."""
    await asyncio.sleep(1 / FRAME_RATE)
    return StreamingResponse(heinsight.generate_frame(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/data")
async def get_data():
    """Endpoint to return additional data."""
    # data = heinsight.output
    frame_data = FrameData(hsdata=heinsight.output)
    return JSONResponse(content=frame_data.dict())