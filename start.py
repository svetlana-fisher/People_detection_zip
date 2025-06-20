from test import FastYOLOv8WithSahi
import cv2
import argparse
import os

def main():
    # Настройка парсера аргументов
    parser = argparse.ArgumentParser(description='YOLOv8 with SAHI detection')
    parser.add_argument('-i', '--input', required=True, help='Input folder path')
    parser.add_argument('-o', '--output', default='result.jpg', help='Output folder path')
    parser.add_argument('--conf', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Ошибка: папка {args.input} не существует!")
        return

    # Инициализация модели
    model = FastYOLOv8WithSahi(
        model_path="best.pt",
        conf_threshold=args.conf
    )

    # Обработка изображений
    model.process_folder(
        input_folder=args.input,
        output_folder=args.output
    )


if __name__ == "__main__":
    main()