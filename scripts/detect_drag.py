from ultralytics import YOLO
from pathlib import Path

# Создаём папку results
Path('results').mkdir(exist_ok=True)

model = YOLO('models/best.pt')

print("=" * 50)
print("ДЕТЕКТОР КОЛОСЬЕВ")
print("=" * 50)

# Запрашиваем путь к фото
image_path = input("Введите путь к фото (или перетащите файл в окно): ").strip().strip('"')

if Path(image_path).exists():
    print(f"Обработка: {Path(image_path).name}")
    
    results = model(image_path)
    results[0].show()
    results[0].save('results/drag_result.jpg')
    
    boxes_count = len(results[0].boxes) if results[0].boxes else 0
    print(f"\nОбнаружено колосьев: {boxes_count}")
    print("Результат сохранён в results/drag_result.jpg")
else:
    print(f"Файл не найден: {image_path}")