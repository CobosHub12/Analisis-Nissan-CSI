from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Permite que herramientas externas (como Stitch) se conecten
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Carga el modelo Multiclase
print("Cargando IA...")
clasificador = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-3")
ETIQUETAS = ["Falla de Producto", "Atención al Cliente", "Servicio Técnico", "Precios", "Satisfecho"]

class Comentario(BaseModel):
    texto: str

@app.post("/analizar")
async def analizar(data: Comentario):
    res = clasificador(data.texto[:512], candidate_labels=ETIQUETAS)
    return {"categoria": res['labels'][0], "confianza": res['scores'][0]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)