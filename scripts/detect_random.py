from ultralytics import YOLO
from pathlib import Path
import random

model = YOLO('models/best.pt')

# Папка с фото
image_folder = Path('data/arvalis_3')
images = list(image_folder.glob('*.png'))

if images:
    test_img = random.choice(images)
    print(f"Тестируем: {test_img.name}")
    
    results = model(str(test_img))
    results[0].show()
    
    # Сохраняем в папку results
    results[0].save('results/random_result.jpg')
    
    boxes_count = len(results[0].boxes) if results[0].boxes else 0
    print(f"Обнаружено колосьев: {boxes_count}")
    print("Результат сохранён в results/random_result.jpg")
else:
    print("Нет фото в папке data/arvalis_3")