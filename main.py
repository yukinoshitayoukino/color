import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
import io
import base64

app = FastAPI()

# Создаем папки
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static/modified", exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Настраиваем шаблоны
templates = Jinja2Templates(directory="templates")


def adjust_color_channels(image_path: str, r_factor: float = 1.0,
                          g_factor: float = 1.0, b_factor: float = 1.0,
                          save_path: Optional[str] = None) -> str:
    """
    Изменяет цветовую карту изображения по каналам RGB.

    Args:
        image_path: Путь к исходному изображению
        r_factor: Коэффициент для красного канала (1.0 - без изменений)
        g_factor: Коэффициент для зеленого канала (1.0 - без изменений)
        b_factor: Коэффициент для синего канала (1.0 - без изменений)
        save_path: Путь для сохранения модифицированного изображения

    Returns:
        Путь к сохраненному изображению или base64 строку
    """
    # Читаем изображение
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Конвертируем из BGR (формат OpenCV) в RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Разделяем на каналы
    r, g, b = cv2.split(image_rgb)

    # Применяем коэффициенты к каждому каналу
    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
    g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)

    # Объединяем каналы обратно
    modified = cv2.merge([r, g, b])

    # Конвертируем обратно в BGR для сохранения через OpenCV
    modified_bgr = cv2.cvtColor(modified, cv2.COLOR_RGB2BGR)

    if save_path:
        # Сохраняем на диск
        cv2.imwrite(save_path, modified_bgr)
        return save_path
    else:
        # Конвертируем в base64 для отображения в браузере
        _, buffer = cv2.imencode('.jpg', modified_bgr)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Получаем список изображений
    images = []
    uploads_dir = "static/uploads"

    if os.path.exists(uploads_dir):
        for file in os.listdir(uploads_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                images.append({
                    "name": file,
                    "url": f"/static/uploads/{file}"
                })

    # Получаем список модифицированных изображений
    modified_images = []
    modified_dir = "static/modified"

    if os.path.exists(modified_dir):
        for file in os.listdir(modified_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                modified_images.append({
                    "name": file,
                    "url": f"/static/modified/{file}"
                })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "images": images,
        "modified_images": modified_images
    })


@app.post("/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    for file in files:
        # Сохраняем файл
        file_path = f"static/uploads/{file.filename}"

        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

    return HTMLResponse(content="""
        <script>
            window.location.href = "/";
        </script>
    """)


@app.post("/adjust-colors")
async def adjust_colors(
        image_name: str = Form(...),
        r_factor: float = Form(1.0),
        g_factor: float = Form(1.0),
        b_factor: float = Form(1.0)
):
    try:
        # Создаем имя для модифицированного файла
        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1]
        modified_name = f"{base_name}_mod_r{r_factor}g{g_factor}b{b_factor}{ext}"

        # Пути к файлам
        source_path = f"static/uploads/{image_name}"
        save_path = f"static/modified/{modified_name}"

        # Изменяем цветовую карту
        adjust_color_channels(source_path, r_factor, g_factor, b_factor, save_path)

        # Возвращаем информацию о модифицированном изображении
        return JSONResponse({
            "success": True,
            "message": "Изображение успешно обработано",
            "image_url": f"/static/modified/{modified_name}"
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при обработке изображения: {str(e)}"
        }, status_code=500)


@app.post("/adjust-preview")
async def adjust_preview(
        image_name: str = Form(...),
        r_factor: float = Form(1.0),
        g_factor: float = Form(1.0),
        b_factor: float = Form(1.0)
):
    """Предпросмотр изменения цветов без сохранения на диск"""
    try:
        source_path = f"static/uploads/{image_name}"
        # Получаем base64 представление модифицированного изображения
        img_base64 = adjust_color_channels(
            source_path, r_factor, g_factor, b_factor, None
        )

        return JSONResponse({
            "success": True,
            "preview_image": img_base64
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при предпросмотре: {str(e)}"
        }, status_code=500)


@app.get("/delete/{folder}/{filename}")
async def delete_image(folder: str, filename: str):
    """Удаление изображения"""
    try:
        file_path = f"static/{folder}/{filename}"
        if os.path.exists(file_path):
            os.remove(file_path)

        return HTMLResponse(content="""
            <script>
                window.location.href = "/";
            </script>
        """)

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при удалении: {str(e)}"
        }, status_code=500)


def clear_static_files():
    import shutil
    shutil.rmtree('static', ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)