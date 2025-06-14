from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import torch
from typing import List
import os
import sys
import traceback

app = FastAPI(title="Tricoscopio API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Load the model at startup with CPU
try:
    print("Iniciando carregamento do modelo bestv2.pt...")
    model_path = "bestv2.pt"
    
    # Verify model file exists
    if not os.path.exists(model_path):
        print(f"ERRO: Arquivo de modelo {model_path} não encontrado!")
        sys.exit(1)
        
    # Check file size
    file_size = os.path.getsize(model_path)
    print(f"Tamanho do arquivo do modelo: {file_size} bytes")
    
    if file_size < 1000:  # Arquivo muito pequeno
        print("ERRO: Arquivo de modelo parece estar corrompido (muito pequeno)")
        sys.exit(1)
        
    # Load model
    model = YOLO(model_path, task='detect')
    model.to('cpu')  # Explicitly set to CPU
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"ERRO ao carregar o modelo: {str(e)}")
    print(traceback.format_exc())
    sys.exit(1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Make prediction using CPU
        results = model(img, device='cpu')
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                detection = {
                    "class": result.names[int(box.cls[0])],
                    "confidence": float(box.conf[0]),
                    "bbox": box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                }
                detections.append(detection)
        
        # Convert the result image to base64 for visualization
        result_img = results[0].plot()  # Get the plotted image with boxes
        im_pil = Image.fromarray(result_img)
        buffered = BytesIO()
        im_pil.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "image": img_str
        })
        
    except Exception as e:
        print(f"Erro no endpoint predict: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.post("/detect-multiple")
async def detect_multiple(files: List[UploadFile] = File(...)):
    try:
        results_list = []
        total_detections = {"ostios": 0, "simples": 0, "duplo": 0, "triplo": 0, "total": 0}
        
        for index, file in enumerate(files):
            # Read the image file
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Make prediction using CPU
            results = model(img, device='cpu')
            
            # Count detections by class
            file_detections = {"ostios": 0, "simples": 0, "duplo": 0, "triplo": 0, "total": 0}
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    
                    if class_name in file_detections:
                        file_detections[class_name] += 1
                        file_detections["total"] += 1
                        
                        # Add to total counts
                        total_detections[class_name] += 1
                        total_detections["total"] += 1
            
            # Convert the result image to base64 for visualization
            result_img = results[0].plot()  # Get the plotted image with boxes
            im_pil = Image.fromarray(result_img)
            buffered = BytesIO()
            im_pil.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Add to results list
            results_list.append({
                "filename": file.filename,
                "detections": file_detections,
                "image_base64": img_str,
                "index": index
            })
        
        return JSONResponse({
            "total_detections": total_detections,
            "results": results_list
        })
        
    except Exception as e:
        print(f"Erro no endpoint detect-multiple: {str(e)}")
        print(traceback.format_exc())
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)

@app.get("/")
async def root():
    return {"message": "Welcome to Tricoscopio API. Use /predict endpoint to detect objects."}

@app.get("/health")
async def health_check():
    """Endpoint para verificar a saúde da API."""
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7000) 
