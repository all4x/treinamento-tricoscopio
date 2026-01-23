from fastapi import FastAPI, UploadFile, File, HTTPException
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

# Configurações de processamento de imagem
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB máximo por arquivo
MAX_IMAGE_DIMENSION = 2048  # Dimensão máxima (largura ou altura)
JPEG_QUALITY = 85  # Qualidade da imagem de saída (0-100)

def resize_image_if_needed(img: np.ndarray, max_dimension: int = MAX_IMAGE_DIMENSION) -> np.ndarray:
    """
    Redimensiona a imagem se ela for maior que max_dimension, mantendo aspect ratio.
    """
    height, width = img.shape[:2]
    
    if height <= max_dimension and width <= max_dimension:
        return img
    
    # Calcula o fator de escala
    scale = max_dimension / max(height, width)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Redimensiona a imagem
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    print(f"Imagem redimensionada de {width}x{height} para {new_width}x{new_height}")
    
    return resized

def validate_image_file(contents: bytes) -> None:
    """
    Valida o tamanho e tipo do arquivo de imagem.
    """
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Arquivo muito grande. Tamanho máximo permitido: {MAX_FILE_SIZE / (1024*1024):.0f}MB"
        )
    
    # Tenta validar se é uma imagem válida
    try:
        img = Image.open(BytesIO(contents))
        img.verify()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Arquivo inválido. Por favor, envie uma imagem válida (JPEG, PNG, etc.)"
        )

def image_to_base64(img: np.ndarray, quality: int = JPEG_QUALITY) -> str:
    """
    Converte uma imagem numpy array para base64, com compressão otimizada.
    """
    # Converte BGR para RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(img_rgb)
    
    # Redimensiona se necessário para a resposta
    width, height = im_pil.size
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        im_pil = im_pil.resize(new_size, Image.LANCZOS)
    
    # Salva com compressão
    buffered = BytesIO()
    im_pil.save(buffered, format="JPEG", quality=quality, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return img_str

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
        # Read and validate the image file
        contents = await file.read()
        validate_image_file(contents)
        
        # Decode image
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Não foi possível decodificar a imagem")
        
        # Resize if needed
        img = resize_image_if_needed(img)
        
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
        img_str = image_to_base64(result_img)
        
        return JSONResponse({
            "success": True,
            "detections": detections,
            "image": img_str
        })
        
    except HTTPException as he:
        raise he
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
        errors = []
        
        for index, file in enumerate(files):
            try:
                # Read and validate the image file
                contents = await file.read()
                validate_image_file(contents)
                
                # Decode image
                nparr = np.frombuffer(contents, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    errors.append({
                        "filename": file.filename,
                        "error": "Não foi possível decodificar a imagem"
                    })
                    continue
                
                # Resize if needed
                img = resize_image_if_needed(img)
                
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
                img_str = image_to_base64(result_img)
                
                # Add to results list
                results_list.append({
                    "filename": file.filename,
                    "detections": file_detections,
                    "image_base64": img_str,
                    "index": index
                })
                
            except HTTPException as he:
                errors.append({
                    "filename": file.filename,
                    "error": he.detail
                })
            except Exception as e:
                errors.append({
                    "filename": file.filename,
                    "error": str(e)
                })
                print(f"Erro processando {file.filename}: {str(e)}")
        
        response = {
            "total_detections": total_detections,
            "results": results_list,
            "processed": len(results_list),
            "failed": len(errors)
        }
        
        if errors:
            response["errors"] = errors
        
        return JSONResponse(response)
        
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
