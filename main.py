import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List
app = FastAPI()

# Создаем папки
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настраиваем шаблоны
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Получаем список изображений
    images = []
    uploads_dir = "static/uploads"

    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append(f"/static/uploads/{file}")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "images": images
    })


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    for file in files:
        # Сохраняем файл
        file_path = f"static/uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            # Читаем и сохраняем файл
            content = await file.read()
            buffer.write(content)

    # Возвращаем HTML с обновленным списком изображений
    return HTMLResponse(content="""
        <script>
            window.location.href = "/";
        </script>
    """)
def clear_static_files():
    import shutil
    shutil.rmtree('static')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)