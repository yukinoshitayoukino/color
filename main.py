import os
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional, Dict
import cv2
import numpy as np
from PIL import Image
import io
import base64
import matplotlib

matplotlib.use('Agg')  # Для работы без GUI
import matplotlib.pyplot as plt
import random
import tempfile

app = FastAPI(title="RGB Image Editor", version="1.0.0")

# Определяем пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

# Создаем папки
os.makedirs(os.path.join(STATIC_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "modified"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "histograms"), exist_ok=True)
os.makedirs(os.path.join(STATIC_DIR, "watermarks"), exist_ok=True)  # Новая папка для водяных знаков
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Настраиваем шаблоны
templates = Jinja2Templates(directory=TEMPLATES_DIR)


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


def add_text_watermark_center(image_path: str, watermark_text: str = "Watermark",
                              opacity: float = 0.5, save_path: Optional[str] = None) -> str:
    """
    Добавляет текстовый водяной знак по центру изображения с использованием OpenCV.
    """
    # Читаем изображение
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    # Создаем копию для водяного знака
    watermarked = image.copy()
    height, width = image.shape[:2]

    # Настройки шрифта OpenCV
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Автоматический расчет размера шрифта в зависимости от размера изображения
    font_scale = min(width, height) / 1000
    font_scale = max(font_scale, 0.5)  # Минимальный размер
    thickness = max(int(font_scale * 2), 2)

    # Получаем размер текста
    text_size, baseline = cv2.getTextSize(watermark_text, font, font_scale, thickness)

    # Позиция по центру
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2

    # Добавляем тень для лучшей читаемости
    # Сначала рисуем черную тень с небольшим смещением
    shadow_offset = 2
    cv2.putText(watermarked, watermark_text,
                (text_x + shadow_offset, text_y + shadow_offset),
                font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    # Создаем маску для основного текста с прозрачностью
    mask = np.zeros_like(image)
    cv2.putText(mask, watermark_text, (text_x, text_y),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    # Смешиваем с прозрачностью
    cv2.addWeighted(mask, opacity, watermarked, 1.0, 0, watermarked)

    if save_path:
        cv2.imwrite(save_path, watermarked)
        return save_path
    else:
        # Конвертируем в base64 для отображения в браузере
        _, buffer = cv2.imencode('.jpg', watermarked)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"


def add_image_watermark_center(background_path: str, watermark_path: str,
                               opacity: float = 0.5, save_path: Optional[str] = None) -> str:
    """
    Добавляет изображение-водяной знак по центру основного изображения с использованием OpenCV.
    """
    # Читаем основное изображение
    background = cv2.imread(background_path)
    if background is None:
        raise ValueError(f"Не удалось загрузить изображение: {background_path}")

    # Читаем водяной знак (с поддержкой альфа-канала)
    watermark = cv2.imread(watermark_path, cv2.IMREAD_UNCHANGED)
    if watermark is None:
        raise ValueError(f"Не удалось загрузить водяной знак: {watermark_path}")

    bg_height, bg_width = background.shape[:2]

    # Масштабируем водяной знак (20% от меньшей стороны основного изображения)
    scale_factor = 0.2
    target_size = min(bg_width, bg_height) * scale_factor

    # Сохраняем пропорции водяного знака
    wm_height, wm_width = watermark.shape[:2]
    wm_aspect = wm_width / wm_height

    if wm_width > wm_height:
        new_width = int(target_size)
        new_height = int(target_size / wm_aspect)
    else:
        new_height = int(target_size)
        new_width = int(target_size * wm_aspect)

    # Масштабируем водяной знак с интерполяцией
    if watermark.shape[2] == 4:  # Если есть альфа-канал
        # Разделяем каналы
        b, g, r, a = cv2.split(watermark)
        rgb_watermark = cv2.merge([b, g, r])

        # Масштабируем RGB и альфа-канал отдельно
        rgb_resized = cv2.resize(rgb_watermark, (new_width, new_height),
                                 interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(a, (new_width, new_height),
                                   interpolation=cv2.INTER_AREA)

        # Собираем обратно
        watermark_resized = cv2.merge([rgb_resized[:, :, 0],
                                       rgb_resized[:, :, 1],
                                       rgb_resized[:, :, 2],
                                       alpha_resized])
    else:
        watermark_resized = cv2.resize(watermark, (new_width, new_height),
                                       interpolation=cv2.INTER_AREA)

    result = background.copy()
    wm_h, wm_w = watermark_resized.shape[:2]

    # Позиция по центру
    x_offset = (bg_width - wm_w) // 2
    y_offset = (bg_height - wm_h) // 2

    # Проверяем, что водяной знак помещается в изображение
    if x_offset >= 0 and y_offset >= 0 and x_offset + wm_w <= bg_width and y_offset + wm_h <= bg_height:

        if watermark_resized.shape[2] == 4:  # С альфа-каналом
            # Нормализуем альфа-канал
            alpha = watermark_resized[:, :, 3] / 255.0 * opacity
            alpha = alpha[:, :, np.newaxis]  # Добавляем измерение для broadcasting

            # Извлекаем RGB каналы
            wm_rgb = watermark_resized[:, :, :3]

            # Область на основном изображении, куда будем накладывать водяной знак
            roi = result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w]

            # Накладываем водяной знак с учетом альфа-канала и прозрачности
            result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w] = \
                roi * (1 - alpha) + wm_rgb * alpha

        else:  # Без альфа-канала
            # Создаем маску с прозрачностью
            roi = result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w]
            result[y_offset:y_offset + wm_h, x_offset:x_offset + wm_w] = \
                cv2.addWeighted(roi, 1, watermark_resized, opacity, 0)

    if save_path:
        cv2.imwrite(save_path, result)
        return save_path
    else:
        _, buffer = cv2.imencode('.jpg', result)
        img_str = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{img_str}"

def create_color_histograms(image_path: str, modified_image_path: Optional[str] = None,
                            r_factor: float = 1.0, g_factor: float = 1.0, b_factor: float = 1.0) -> Dict[str, str]:
    """
    Создает гистограммы распределения цветов.
    """
    # Оригинальное изображение
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"Не удалось загрузить изображение: {image_path}")

    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    # Модифицированное изображение
    if modified_image_path and os.path.exists(modified_image_path):
        mod_image = cv2.imread(modified_image_path)
        mod_image_rgb = cv2.cvtColor(mod_image, cv2.COLOR_BGR2RGB)
    else:
        mod_image_rgb = np.copy(orig_image_rgb)
        r, g, b = cv2.split(mod_image_rgb)
        r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
        g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
        b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
        mod_image_rgb = cv2.merge([r, g, b])

    # Создаем фигуру
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

    # Сравнительная гистограмма
    ax3 = axes[1, 0]
    for i, color in enumerate(colors):
        hist_orig = cv2.calcHist([orig_image_rgb], [i], None, [256], [0, 256])
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

        hist_diff = hist_mod - hist_orig

        if np.max(np.abs(hist_diff)) > 0:
            hist_diff_normalized = hist_diff / np.max(np.abs(hist_diff)) * np.max(
                [hist_orig.max(), hist_mod.max()]) * 0.3
        else:
            hist_diff_normalized = hist_diff

        ax4.plot(hist_diff_normalized, color=color, alpha=0.7, label=color_names[i])
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
    orig_image = cv2.imread(image_path)
    orig_image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

    mod_image_rgb = np.copy(orig_image_rgb)
    r, g, b = cv2.split(mod_image_rgb)
    r = np.clip(r * r_factor, 0, 255).astype(np.uint8)
    g = np.clip(g * g_factor, 0, 255).astype(np.uint8)
    b = np.clip(b * b_factor, 0, 255).astype(np.uint8)
    mod_image_rgb = cv2.merge([r, g, b])

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

    # Получаем список водяных знаков
    watermark_images = []
    watermark_dir = "static/watermarks"

    if os.path.exists(watermark_dir):
        for file in os.listdir(watermark_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                watermark_images.append({
                    "name": file,
                    "url": f"/static/watermarks/{file}"
                })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "images": images,
        "modified_images": modified_images,
        "histogram_images": histogram_images,
        "watermark_images": watermark_images
    })


@app.post("/upload")
async def upload_images(
    files: List[UploadFile] = File(...),
    upload_type: str = Form("images")  # "images" или "watermarks"
):
    for file in files:
        # Определяем папку для сохранения
        if upload_type == "watermarks":
            folder = "watermarks"
        else:
            folder = "uploads"

        file_path = f"static/{folder}/{file.filename}"

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
        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1]
        modified_name = f"{base_name}_mod_r{r_factor}g{g_factor}b{b_factor}{ext}"
        histogram_name = f"{base_name}_hist_r{r_factor}g{g_factor}b{b_factor}.png"

        source_path = f"static/uploads/{image_name}"
        save_path = f"static/modified/{modified_name}"
        histogram_path = f"static/histograms/{histogram_name}"

        adjust_color_channels(source_path, r_factor, g_factor, b_factor, save_path)

        histograms = create_color_histograms(
            source_path, save_path, r_factor, g_factor, b_factor
        )

        hist_data = histograms["histogram"].split(",")[1]
        with open(histogram_path, "wb") as f:
            f.write(base64.b64decode(hist_data))

        orig_stats = get_image_stats(source_path)
        mod_stats = get_image_stats(save_path)

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

        img_base64 = adjust_color_channels(
            source_path, r_factor, g_factor, b_factor, None
        )

        hist_base64 = create_simple_histogram(
            source_path, r_factor, g_factor, b_factor
        )

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


@app.post("/watermark-preview")
async def watermark_preview(
        image_name: str = Form(...),
        r_factor: float = Form(1.0),
        g_factor: float = Form(1.0),
        b_factor: float = Form(1.0),
        watermark_type: str = Form("text"),
        watermark_text: Optional[str] = Form("Watermark"),
        watermark_image: Optional[str] = Form(None),
        opacity: float = Form(0.5)
):
    """Предпросмотр водяного знака без сохранения на диск"""
    try:
        source_path = f"static/uploads/{image_name}"

        # Сначала применяем коррекцию RGB
        temp_path = None
        if r_factor != 1.0 or g_factor != 1.0 or b_factor != 1.0:
            # Создаем временное изображение с коррекцией
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            adjust_color_channels(source_path, r_factor, g_factor, b_factor, temp_path)
            image_to_watermark = temp_path
        else:
            image_to_watermark = source_path

        # Применяем водяной знак
        if watermark_type == "text":
            img_base64 = add_text_watermark_center(
                image_to_watermark,
                watermark_text=watermark_text,
                opacity=opacity,
                save_path=None
            )
        else:
            if not watermark_image:
                # Если изображение не выбрано, возвращаем обычный предпросмотр
                img_base64 = adjust_color_channels(
                    source_path, r_factor, g_factor, b_factor, None
                )
            else:
                watermark_path = f"static/watermarks/{watermark_image}"
                img_base64 = add_image_watermark_center(
                    image_to_watermark,
                    watermark_path,
                    opacity=opacity,
                    save_path=None
                )

        # Удаляем временный файл если он был создан
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

        # Создаем гистограмму для предпросмотра
        hist_base64 = create_simple_histogram(
            source_path, r_factor, g_factor, b_factor
        )

        # Получаем статистику
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
            "message": f"Ошибка при предпросмотре водяного знака: {str(e)}"
        }, status_code=500)


@app.post("/add-watermark")
async def add_watermark_to_image(
        image_name: str = Form(...),
        r_factor: float = Form(1.0),
        g_factor: float = Form(1.0),
        b_factor: float = Form(1.0),
        watermark_type: str = Form(...),
        watermark_text: Optional[str] = Form("Watermark"),
        watermark_image: Optional[str] = Form(None),
        opacity: float = Form(0.5)
):
    """Добавление водяного знака на изображение"""
    try:
        base_name = os.path.splitext(image_name)[0]
        ext = os.path.splitext(image_name)[1]

        random_suffix = random.randint(1000, 9999)

        if watermark_type == "text":
            watermarked_name = f"{base_name}_text_wm_r{r_factor}g{g_factor}b{b_factor}_{random_suffix}{ext}"
        else:
            watermarked_name = f"{base_name}_img_wm_r{r_factor}g{g_factor}b{b_factor}_{random_suffix}{ext}"

        save_path = f"static/modified/{watermarked_name}"
        source_path = f"static/uploads/{image_name}"

        # Сначала применяем коррекцию RGB
        temp_path = None
        if r_factor != 1.0 or g_factor != 1.0 or b_factor != 1.0:
            temp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_path = temp_file.name
            temp_file.close()

            adjust_color_channels(source_path, r_factor, g_factor, b_factor, temp_path)
            image_to_watermark = temp_path
        else:
            image_to_watermark = source_path

        # Применяем водяной знак
        if watermark_type == "text":
            result = add_text_watermark_center(
                image_to_watermark,
                watermark_text=watermark_text,
                opacity=opacity,
                save_path=save_path
            )
        else:
            if not watermark_image:
                return JSONResponse({
                    "success": False,
                    "message": "Выберите изображение для водяного знака"
                }, status_code=400)

            watermark_path = f"static/watermarks/{watermark_image}"
            result = add_image_watermark_center(
                image_to_watermark,
                watermark_path,
                opacity=opacity,
                save_path=save_path
            )

        # Удаляем временный файл если он был создан
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

        # Создаем гистограмму
        histogram_name = f"{base_name}_hist_wm_r{r_factor}g{g_factor}b{b_factor}.png"
        histogram_path = f"static/histograms/{histogram_name}"

        histograms = create_color_histograms(
            source_path, save_path, r_factor, g_factor, b_factor
        )

        hist_data = histograms["histogram"].split(",")[1]
        with open(histogram_path, "wb") as f:
            f.write(base64.b64decode(hist_data))

        # Получаем статистику
        orig_stats = get_image_stats(source_path)
        watermark_stats = get_image_stats(save_path)

        return JSONResponse({
            "success": True,
            "message": "Водяной знак успешно добавлен",
            "image_url": f"/static/modified/{watermarked_name}",
            "histogram_url": f"/static/histograms/{histogram_name}",
            "stats": {
                "original": orig_stats,
                "watermarked": watermark_stats
            }
        })

    except Exception as e:
        return JSONResponse({
            "success": False,
            "message": f"Ошибка при добавлении водяного знака: {str(e)}"
        }, status_code=500)


@app.get("/get-histogram")
async def get_histogram(image_name: str):
    """Получение гистограммы для конкретного изображения"""
    try:
        source_path = f"static/uploads/{image_name}"

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


# Главная функция запуска
if __name__ == "__main__":
    import uvicorn

    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )