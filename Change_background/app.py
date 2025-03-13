from fastapi import FastAPI, File, UploadFile
import gradio as gr
from rembg import remove
from PIL import Image, ImageEnhance, ImageFilter
import io

app = FastAPI()


def resize_and_crop(img, target_size):
    img_ratio = img.width / img.height
    target_ratio = target_size[0] / target_size[1]

    if img_ratio > target_ratio:
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    else:
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    left = (new_width - target_size[0]) // 2
    top = (new_height - target_size[1]) // 2
    right = left + target_size[0]
    bottom = top + target_size[1]

    return img.crop((left, top, right, bottom))


def change_background(foreground, background, blur_radius, brightness, contrast):
    fg_no_bg = remove(foreground).convert("RGBA")
    background = resize_and_crop(background, fg_no_bg.size).convert("RGBA")

    if blur_radius > 0:
        fg_no_bg = fg_no_bg.filter(ImageFilter.GaussianBlur(blur_radius))

    fg_no_bg = ImageEnhance.Brightness(fg_no_bg).enhance(brightness)
    fg_no_bg = ImageEnhance.Contrast(fg_no_bg).enhance(contrast)

    result = background.copy()
    result.paste(fg_no_bg, (0, 0), fg_no_bg)

    return result

@app.post('/change_background')
async def api_background(
        foreground: UploadFile = File(...),
        background: UploadFile = File(...),
        blur_radius: int = 0,
        brightness: float = 1.0,
        contrast: float = 1.0,):
    fg_image = Image.open(io.BytesIO(await foreground.read()))
    bg_image = Image.open(io.BytesIO(await background.read()))

    result = change_background(fg_image, bg_image,blur_radius, brightness, contrast)
    img_io = io.BytesIO()
    result.save(img_io, format='JPEG')
    img_io.seek(0)

    return {'filename': 'output.jpeg', 'content': img_io.getvalue()}


interface = gr.Interface(
    fn=change_background,
    inputs=[
        gr.Image(type="pil", label="Ваше фото"),
        gr.Image(type="pil", label="Фон"),
        gr.Slider(0, 10, value=0, step=1, label="Размытие краёв"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Яркость"),
        gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="Контраст"),
    ],
    outputs=gr.Image(type="pil", label='Результат', show_download_button=True),
    title="Замена фона",
    description="Загрузите ваше фото и фон. Настройте параметры"
)

app = gr.mount_gradio_app(app, interface, path='/gr')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)