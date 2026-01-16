import os
import uuid
import io
import base64
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
import asyncio
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import matplotlib

matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt

app = FastAPI(title="RGB Image Editor", version="1.0.0")

# Определяем пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Создаем папки
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "modified"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "histograms"), exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Настраиваем шаблоны
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Пул потоков для параллельной обработки
executor = ThreadPoolExecutor(max_workers=4)


def adjust_color_channels(image_path: str, r_factor: float = 1.0,
                          g_factor: float = 1.0, b_factor: float = 1.0,
                          save_path: Optional[str] = None) -> str:
    """
    Изменяет цветовую карту изображения по каналам RGB.
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


def add_watermark_to_image_sync(
        image_path: str,
        watermark_type: str = "text",
        watermark_text: str = "WATERMARK",
        watermark_file_path: Optional[str] = None,
        font_size: int = 40,
        opacity: float = 0.5,
        color: str = "#FFFFFF"
):
    """Синхронная версия функции добавления водяного знака"""
    # Открываем исходное изображение
    with Image.open(image_path) as img:
        img = img.convert("RGBA")
        width, height = img.size

        # Создаем слой для водяного знака
        watermark = Image.new("RGBA", img.size, (0, 0, 0, 0))

        if watermark_type == "text":
            # Создаем контекст для рисования
            draw = ImageDraw.Draw(watermark)

            # Пытаемся загрузить шрифт
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                try:
                    # Попробуем найти стандартный шрифт
                    font_paths = [
                        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                        "C:/Windows/Fonts/arial.ttf"
                    ]
                    for font_path in font_paths:
                        if os.path.exists(font_path):
                            font = ImageFont.truetype(font_path, font_size)
                            break
                    else:
                        font = ImageFont.load_default()
                except:
                    font = ImageFont.load_default()

            # Рассчитываем размер текста
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Позиция по центру
            position = ((width - text_width) // 2, (height - text_height) // 2)

            # Преобразуем HEX цвет в RGB
            color_rgb = tuple(int(color.lstrip('#')[i:i + 2], 16) for i in (0, 2, 4))

            # Рисуем текст
            draw.text(position, watermark_text, font=font, fill=color_rgb)

        elif watermark_type == "image" and watermark_file_path:
            # Открываем изображение водяного знака
            with Image.open(watermark_file_path) as watermark_img:
                watermark_img = watermark_img.convert("RGBA")

                # Масштабируем водяной знак
                watermark_size = min(width, height) // 5
                watermark_img.thumbnail((watermark_size, watermark_size))

                # Позиция по центру
                wm_width, wm_height = watermark_img.size
                position = ((width - wm_width) // 2, (height - wm_height) // 2)

                # Вставляем водяной знак
                watermark.paste(watermark_img, position, watermark_img)

        # Применяем прозрачность
        if opacity < 1.0:
            alpha = watermark.split()[3]
            alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
            watermark.putalpha(alpha)

        # Накладываем водяной знак
        result = Image.alpha_composite(img, watermark)

        # Сохраняем результат
        output_filename = f"watermarked_{os.path.basename(image_path)}"
        output_path = os.path.join("static", "modified", output_filename)
        result.save(output_path, "PNG")

        return output_path


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


@app.post("/watermark")
async def add_watermark(
        file: UploadFile = File(...),
        watermark_type: str = Form("text"),
        watermark_text: Optional[str] = Form("WATERMARK"),
        watermark_file: Optional[UploadFile] = File(None),
        font_size: int = Form(40),
        opacity: float = Form(0.5),
        color: str = Form("#FFFFFF")
):
    """
    Добавляет водяной знак в центр изображения
    """
    # Проверяем тип файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Файл должен быть изображением")

    # Проверяем тип водяного знака
    if watermark_type not in ["text", "image"]:
        raise HTTPException(status_code=400, detail="Тип водяного знака должен быть 'text' или 'image'")

    if watermark_type == "image" and not watermark_file:
        raise HTTPException(status_code=400, detail="Для типа 'image' требуется файл водяного знака")

    # Сохраняем временные файлы
    temp_files = []

    try:
        # Сохраняем основное изображение
        image_filename = f"temp_{uuid.uuid4().hex}_{file.filename}"
        image_path = os.path.join("static", "uploads", image_filename)

        with open(image_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        temp_files.append(image_path)

        watermark_file_path = None
        if watermark_type == "image" and watermark_file:
            # Сохраняем водяной знак
            watermark_filename = f"watermark_{uuid.uuid4().hex}_{watermark_file.filename}"
            watermark_file_path = os.path.join("static", "uploads", watermark_filename)

            with open(watermark_file_path, "wb") as buffer:
                content = await watermark_file.read()
                buffer.write(content)

            temp_files.append(watermark_file_path)

        # Обрабатываем водяной знак в отдельном потоке
        output_path = await asyncio.get_event_loop().run_in_executor(
            executor,
            add_watermark_to_image_sync,
            image_path,
            watermark_type,
            watermark_text,
            watermark_file_path,
            font_size,
            opacity,
            color
        )

        # Получаем имя файла для ответа
        output_filename = os.path.basename(output_path)

        # Возвращаем ссылку на обработанный файл
        return JSONResponse({
            "success": True,
            "message": "Водяной знак успешно добавлен",
            "image_url": f"/static/modified/{output_filename}",
            "filename": output_filename
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка при добавлении водяного знака: {str(e)}")
    finally:
        # Удаляем временные файлы
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


@app.post("/watermark/batch")
async def add_watermark_batch(
        files: List[UploadFile] = File(...),
        watermark_type: str = Form("text"),
        watermark_text: Optional[str] = Form("WATERMARK"),
        watermark_file: Optional[UploadFile] = File(None),
        font_size: int = Form(40),
        opacity: float = Form(0.5),
        color: str = Form("#FFFFFF")
):
    """
    Пакетное добавление водяного знака на несколько изображений
    """
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Максимум 10 файлов за раз")

    temp_files = []
    processed_files = []

    try:
        # Сохраняем водяной знак если нужно
        watermark_file_path = None
        if watermark_type == "image" and watermark_file:
            watermark_filename = f"batch_watermark_{uuid.uuid4().hex}_{watermark_file.filename}"
            watermark_file_path = os.path.join("static", "uploads", watermark_filename)

            with open(watermark_file_path, "wb") as buffer:
                content = await watermark_file.read()
                buffer.write(content)

            temp_files.append(watermark_file_path)

        # Обрабатываем каждое изображение
        for file in files:
            if not file.content_type.startswith("image/"):
                continue

            # Сохраняем изображение
            image_filename = f"batch_{uuid.uuid4().hex}_{file.filename}"
            image_path = os.path.join("static", "uploads", image_filename)

            with open(image_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            temp_files.append(image_path)

            # Обрабатываем водяной знак
            output_path = await asyncio.get_event_loop().run_in_executor(
                executor,
                add_watermark_to_image_sync,
                image_path,
                watermark_type,
                watermark_text,
                watermark_file_path,
                font_size,
                opacity,
                color
            )

            processed_files.append(os.path.basename(output_path))

        return JSONResponse({
            "success": True,
            "message": f"Обработано {len(processed_files)} изображений",
            "processed_files": processed_files,
            "urls": [f"/static/modified/{filename}" for filename in processed_files]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка пакетной обработки: {str(e)}")
    finally:
        # Удаляем временные файлы
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


# Главная функция запуска
if __name__ == "__main__":
    import uvicorn

    print("Starting server on http://0.0.0.0:8000")
    print("Доступные эндпоинты:")
    print("- GET  / - главная страница")
    print("- POST /upload - загрузка изображений")
    print("- POST /adjust-colors - изменение цветов")
    print("- POST /watermark - добавление водяного знака")
    print("- POST /watermark/batch - пакетное добавление водяных знаков")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )