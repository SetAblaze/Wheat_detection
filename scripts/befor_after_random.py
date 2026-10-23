from ultralytics import YOLO
from pathlib import Path
import random
import matplotlib.pyplot as plt
from PIL import Image

model = YOLO('models/best.pt')

# Папки
image_folder = Path('data/arvalis_3')
output_folder = Path('results')
output_folder.mkdir(exist_ok=True)

images = list(image_folder.glob('*.png'))

if images:
    test_img = random.choice(images)
    print(f"Тестируем: {test_img.name}")
    
    # Предсказание
    results = model(str(test_img))
    
    # Загружаем оригинальное изображение
    img_original = Image.open(test_img)
    
    # Получаем изображение с рамками от модели
    img_with_boxes = results[0].plot()
    
    # Создаём рисунок с двумя картинками рядом
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Левая картинка - оригинал (ДО)
    ax1.imshow(img_original)
    ax1.set_title(f"ДО ОБРАБОТКИ\n{test_img.name}", fontsize=12)
    ax1.axis('off')
    
    # Правая картинка - с предсказаниями (ПОСЛЕ)
    ax2.imshow(img_with_boxes)
    boxes_count = len(results[0].boxes) if results[0].boxes else 0
    ax2.set_title(f"ПОСЛЕ ОБРАБОТКИ\nНайдено колосьев: {boxes_count}", fontsize=12)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Обнаружено колосьев: {boxes_count}")
else:
    print("Нет фото в папке data/arvalis_3")

    