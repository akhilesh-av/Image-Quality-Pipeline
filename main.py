from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
import os
import tempfile
import numpy as np
from utils import ImageQualityPipeline

app = FastAPI()

def convert_numpy_types(obj):
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float16, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


@app.post("/upload/")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Create a temporary file to store the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Initialize pipeline with the temporary file path
        pipeline = ImageQualityPipeline(temp_file_path)

        # Run the pipeline
        processed_image, report = pipeline.run_pipeline()

        # Convert NumPy types in the report to native Python types
        report = convert_numpy_types(report)

        # Return the report and the processed image
        return {
            "report": report,
            "processed_image_path": report['output_path'],
            "report_path": report['report_path']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{report_path:path}")
async def get_report(report_path: str):
    return FileResponse(report_path)

@app.get("/image/{image_path:path}")
async def get_image(image_path: str):
    return FileResponse(image_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
