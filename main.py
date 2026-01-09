import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict, Tuple
import cv2
import numpy as np
from PIL import Image
import io
import base64
import matplotlib

matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt

app = FastAPI()

# Создаем папки
os.makedirs("static/uploads", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("static/modified", exist_ok=True)
os.makedirs("static/histograms", exist_ok=True)

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


def create_color_histograms(image_path: str, modified_image_path: Optional[str] = None,
                            r_factor: float = 1.0, g_factor: float = 1.0, b_factor: float = 1.0) -> Dict[str, str]:
    """
    Создает гистограммы распределения цветов для оригинального и модифицированного изображений.

    Args:
        image_path: Путь к оригинальному изображению
        modified_image_path: Путь к модифицированному изображению (опционально)
        r_factor, g_factor, b_factor: Коэффициенты для модификации

    Returns:
        Словарь с base64 строками гистограмм
    """
    # Читаем оригинальное изображение
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Конвертируем в RGB
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Получаем модифицированное изображение
    if modified_image_path and os.path.exists(modified_image_path):
        mod_image = cv2.imread(modified_image_path)
        mod_image_rgb = cv2.cvtColor(mod_image, cv2.COLOR_BGR2RGB)
    else:
        # Создаем модифицированное изображение на лету
        mod_image_rgb = np.copy(orig_image_rgb)
        r, g, b = cv2.split(mod_image_rgb)
        r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
        g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
        b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
        mod_image_rgb = cv2.merge([r, g, b])

    # Создаем фигуру с двумя гистограммами
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Распределение цветов по каналам RGB', fontsize=16, fontweight='bold')

    colors = ['red', 'green', 'blue']
    color_names = ['Красный', 'Зеленый', 'Синий']

    # Гистограмма оригинального изображения
    ax1 = axes[0, 0]
    for i, color in enumerate(colors):
        hist = cv2.calcHist([orig_image_rgb], [i], None, [256], [0, 256])
        ax1.plot(hist, color=color, alpha=0.7, label=color_names[i])

    ax1.set_title('Оригинальное изображение', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Интенсивность', fontsize=12)
    ax1.set_ylabel('Количество пикселей', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 255])

    # Гистограмма модифицированного изображения
    ax2 = axes[0, 1]
    for i, color in enumerate(colors):
        hist = cv2.calcHist([mod_image_rgb], [i], None, [256], [0, 256])
        ax2.plot(hist, color=color, alpha=0.7, label=color_names[i])

    ax2.set_title(f'Модифицированное изображение\n(R×{r_factor}, G×{g_factor}, B×{b_factor})',
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Интенсивность', fontsize=12)
    ax2.set_ylabel('Количество пикселей', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 255])

    # Сравнительная гистограмма по каналам
    ax3 = axes[1, 0]
    for i, color in enumerate(colors):
        # Оригинальный канал
        hist_orig = cv2.calcHist([orig_image_rgb], [i], None, [256], [0, 256])
        # Модифицированный канал
        hist_mod = cv2.calcHist([mod_image_rgb], [i], None, [256], [0, 256])

        ax3.plot(hist_orig, color=color, alpha=0.5, linestyle='-', label=f'{color_names[i]} (ориг.)')
        ax3.plot(hist_mod, color=color, alpha=0.8, linestyle='--', label=f'{color_names[i]} (мод.)')

    ax3.set_title('Сравнение оригинальных и модифицированных каналов', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Интенсивность', fontsize=12)
    ax3.set_ylabel('Количество пикселей', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 255])

    # Разностная гистограмма
    ax4 = axes[1, 1]
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([orig_image_rgb], [i], None, [256], [0, 256])
        hist_mod = cv2.calcHist([mod_image_rgb], [i], None, [256], [0, 256])

        # Разность гистограмм
        hist_diff = hist_mod - hist_orig

        # Нормализуем для лучшей визуализации
        if np.max(np.abs(hist_diff)) > 0:
            hist_diff_normalized = hist_diff / np.max(np.abs(hist_diff)) * np.max(
                [hist_orig.max(), hist_mod.max()]) * 0.3
        else:
            hist_diff_normalized = hist_diff

        ax4.plot(hist_diff_normalized, color=color, alpha=0.7, label=color_names[i])
        # Заполняем область разности
        ax4.fill_between(range(256), 0, hist_diff_normalized.flatten(),
                         color=color, alpha=0.2)

    ax4.set_title('Разность гистограмм (модиф. - оригин.)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Интенсивность', fontsize=12)
    ax4.set_ylabel('Разность (норм.)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 255])
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)

    # Настраиваем layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    # Конвертируем в base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)

    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return {
        "histogram": f"data:image/png;base64,{img_str}"
    }


def create_simple_histogram(image_path: str, r_factor: float = 1.0,
                            g_factor: float = 1.0, b_factor: float = 1.0) -> str:
    """
    Создает упрощенную гистограмму для предпросмотра.
    """
    # Читаем оригинальное изображение
    orig_image = cv2.imread(image_path)
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Создаем модифицированное изображение
    mod_image_rgb = np.copy(orig_image_rgb)
    r, g, b = cv2.split(mod_image_rgb)
    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
    g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
    mod_image_rgb = cv2.merge([r, g, b])

    # Создаем фигуру
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    colors = ['red', 'green', 'blue']

    # Оригинальная гистограмма
    for i, color in enumerate(colors):
        hist = cv2.calcHist([orig_image_rgb], [i], None, [256], [0, 256])
        ax1.plot(hist, color=color, alpha=0.7)

    ax1.set_title('Оригинальное изображение')
    ax1.set_xlabel('Интенсивность')
    ax1.set_ylabel('Пиксели')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 255])

    # Модифицированная гистограмма
    for i, color in enumerate(colors):
        hist = cv2.calcHist([mod_image_rgb], [i], None, [256], [0, 256])
        ax2.plot(hist, color=color, alpha=0.7)

    ax2.set_title(f'Модифицированное (R×{r_factor}, G×{g_factor}, B×{b_factor})')
    ax2.set_xlabel('Интенсивность')
    ax2.set_ylabel('Пиксели')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 255])

    plt.tight_layout()

    # Конвертируем в base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
    plt.close(fig)

    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')

    return f"data:image/png;base64,{img_str}"


def get_image_stats(image_path: str) -> Dict[str, Dict[str, float]]:
    """
    Возвращает статистику по изображению.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    stats = {}
    colors = ['r', 'g', 'b']
    color_names = ['red', 'green', 'blue']

    for i, (color, name) in enumerate(zip(colors, color_names)):
        channel = image_rgb[:, :, i]
        stats[name] = {
            'mean': float(np.mean(channel)),
            'std': float(np.std(channel)),
            'min': float(np.min(channel)),
            'max': float(np.max(channel)),
            'median': float(np.median(channel))
        }

    return stats


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

    # Получаем список гистограмм
    histogram_images = []
    histogram_dir = "static/histograms"

    if os.path.exists(histogram_dir):
        for file in os.listdir(histogram_dir):
            if file.lower().endswith('.png'):
                histogram_images.append({
                    "name": file,
                    "url": f"/static/histograms/{file}"
                })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "images": images,
        "modified_images": modified_images,
        "histogram_images": histogram_images
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
        histogram_name = f"{base_name}_hist_r{r_factor}g{g_factor}b{b_factor}.png"

        # Пути к файлам
        source_path = f"static/uploads/{image_name}"
        save_path = f"static/modified/{modified_name}"
        histogram_path = f"static/histograms/{histogram_name}"

        # Изменяем цветовую карту
        adjust_color_channels(source_path, r_factor, g_factor, b_factor, save_path)

        # Создаем гистограммы
        histograms = create_color_histograms(
            source_path, save_path, r_factor, g_factor, b_factor
        )

        # Сохраняем гистограмму на диск
        hist_data = histograms["histogram"].split(",")[1]
        with open(histogram_path, "wb") as f:
            f.write(base64.b64decode(hist_data))

        # Получаем статистику
        orig_stats = get_image_stats(source_path)
        mod_stats = get_image_stats(save_path)

        # Возвращаем информацию о модифицированном изображении
        return JSONResponse({
            "success": True,
            "message": "Изображение успешно обработано",
            "image_url": f"/static/modified/{modified_name}",
            "histogram_url": f"/static/histograms/{histogram_name}",
            "stats": {
                "original": orig_stats,
                "modified": mod_stats
            }
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

        # Создаем упрощенную гистограмму для предпросмотра
        hist_base64 = create_simple_histogram(
            source_path, r_factor, g_factor, b_factor
        )

        # Получаем статистику для отображения
        stats = get_image_stats(source_path)

        return JSONResponse({
            "success": True,
            "preview_image": img_base64,
            "preview_histogram": hist_base64,
            "stats": stats
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при предпросмотре: {str(e)}"
        }, status_code=500)


@app.get("/get-histogram")
async def get_histogram(image_name: str):
    """Получение гистограммы для конкретного изображения"""
    try:
        source_path = f"static/uploads/{image_name}"

        # Создаем базовую гистограмму
        hist_base64 = create_simple_histogram(source_path, 1.0, 1.0, 1.0)

        return JSONResponse({
            "success": True,
            "histogram": hist_base64
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при создании гистограммы: {str(e)}"
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