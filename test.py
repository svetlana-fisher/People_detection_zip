from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import cv2
import numpy as np
import tempfile
import os
from tqdm import tqdm


class FastYOLOv8WithSahi:
    def __init__(self, model_path, conf_threshold=0.3, device="cuda:0"):
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=conf_threshold,
            device=device,
        )
        self.slice_size = 832
        self.overlap_ratio = 0.2

    def process_folder(self, input_folder, output_folder):
        os.makedirs(output_folder, exist_ok=True)

        # Поддерживаемые форматы изображений
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [f for f in os.listdir(input_folder)
                       if f.lower().endswith(valid_exts)]

        if not image_files:
            print(f"В папке {input_folder} нет изображений!")
            return

        for img_file in tqdm(image_files, desc="Обработка изображений"):
            img_path = os.path.join(input_folder, img_file)
            base_name = os.path.splitext(img_file)[0]
            output_img_path = os.path.join(output_folder, img_file)
            output_txt_path = os.path.join(output_folder, f"{base_name}.txt")

            # Загрузка изображения для получения размеров
            img = cv2.imread(img_path)
            img_height, img_width = img.shape[:2]

            # Детекция
            result = get_sliced_prediction(
                img_path,
                self.model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                verbose=0
            )

            # Сбор аннотаций в YOLO-формате
            yolo_annotations = []
            for pred in result.object_prediction_list:
                bbox = pred.bbox

                # Конвертация в YOLO-формат
                x_center = (bbox.minx + bbox.maxx) / 2 / img_width
                y_center = (bbox.miny + bbox.maxy) / 2 / img_height
                width = (bbox.maxx - bbox.minx) / img_width
                height = (bbox.maxy - bbox.miny) / img_height

                yolo_annotations.append(
                    f"{pred.category.id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                )

                # Отрисовка на изображении
                cv2.rectangle(img,
                              (int(bbox.minx), int(bbox.miny)),
                              (int(bbox.maxx), int(bbox.maxy)),
                              (0, 255, 0), 2)
                cv2.putText(img, f"{pred.category.id}:{pred.score.value:.2f}",
                            (int(bbox.minx), int(bbox.miny) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Сохранение результатов
            cv2.imwrite(output_img_path, img)

            # Сохранение аннотаций
            with open(output_txt_path, 'w') as f:
                f.write("\n".join(yolo_annotations))

        print(f"Результаты сохранены в: {os.path.abspath(output_folder)}")