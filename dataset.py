from datasets import Dataset, DatasetDict
from datasets import load_dataset
from PIL import Image
import torch
import numpy as np

def build_dataset() -> DatasetDict:
    """
    Build the dataset for object detection with unified RGB image mode.

    Returns:
        DatasetDict: The dataset with train, validation, and test splits,
                     where all images are converted to RGB mode.
    """
    # 加载 CPPE-5 数据集
    raw_datasets = load_dataset("cppe-5")

    # 如果没有 validation 分割，从 train 中拆分 15% 作为 validation
    if "validation" not in raw_datasets:
        split = raw_datasets["train"].train_test_split(test_size=0.15, seed=1337)
        raw_datasets["train"] = split["train"]
        raw_datasets["validation"] = split["test"]

    # 确保 test 分割存在
    if "test" not in raw_datasets:
        raw_datasets["test"] = load_dataset("cppe-5", split="test")

    print("loading raw_datasets successfully")
    return raw_datasets


def process_to_rgb(raw_datasets: DatasetDict) -> DatasetDict:
    """
    Process the raw dataset to convert all images to RGB mode using iteration.
    """
    processed_datasets = DatasetDict()

    for split_name in raw_datasets.keys():
        print(f"\nProcessing {split_name} split with {len(raw_datasets[split_name])} samples:")
        processed_data = []

        for idx, example in enumerate(raw_datasets[split_name]):
            image = example["image"]
            image_id = example["image_id"]

            if not isinstance(image, Image.Image):
                try:
                    image = Image.fromarray(image)
                except Exception as e:
                    print(f"Sample {idx} (ID: {image_id}) - Failed to convert to PIL: {e}")
                    continue

            original_mode = image.mode
            if original_mode != "RGB":
                image = image.convert("RGB")

            processed_example = example.copy()
            processed_example["image"] = image
            processed_data.append(processed_example)

        # Corrected the usage of from_dict
        processed_datasets[split_name] = Dataset.from_dict(
            {k: [d[k] for d in processed_data] for k in processed_data[0].keys()}
        )
        print(f"Finished processing {split_name} split.")

    return processed_datasets

def add_preprocessing(raw_dataset, processor) -> DatasetDict:
    """
    Add preprocessing to the dataset using only the processor.

    Args:
        raw_dataset: The dataset to preprocess.
        processor: The DetrImageProcessor to use for preprocessing.

    Returns:
        The preprocessed dataset.
    """
    datasets = process_to_rgb(raw_dataset)

    # Batch transformation function for both train and validation/test splits
    def transform_batch(examples):
        images = []
        targets = []
        for i in range(len(examples["image"])):  # 使用 "image" 字段
            # 将 PIL 图像转换为 NumPy 数组
            image = examples["image"][i]
            if isinstance(image, Image.Image):
                image = torch.tensor(np.array(image)).permute(2, 0, 1)  # 转换为 [C, H, W]

            # 提取标签信息
            objects = examples["objects"][i]
            bboxes = objects["bbox"]  # 从 "objects" 中提取 "bbox"
            category_ids = objects["category"]  # 从 "objects" 中提取 "category"
            areas = objects["area"]  # 从 "objects" 中提取 "area"

            # 将 bbox 从 [x, y, w, h] 转换为 [x0, y0, x1, y1]
            converted_bboxes = []
            for bbox in bboxes:
                x, y, w, h = bbox
                x0, y0, x1, y1 = x, y, x + w, y + h
                converted_bboxes.append([x0, y0, x1, y1])

            # 构建 annotations
            annotations = []
            for j in range(len(converted_bboxes)):
                annotation = {
                    "bbox": converted_bboxes[j],
                    "category_id": category_ids[j],
                    "area": areas[j],
                }
                annotations.append(annotation)

            target = {
                "image_id": examples["image_id"][i],  # 使用 "image_id" 字段
                "annotations": annotations,
            }
            images.append(image)
            targets.append(target)

        # 使用处理器进行预处理
        encoding = processor(images=images, annotations=targets, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "labels": encoding["labels"],
        }

    # Apply the same transformation to all splits (train, validation, test)
    datasets["train"] = datasets["train"].with_transform(transform_batch)
    datasets["validation"] = datasets["validation"].with_transform(transform_batch)
    datasets["test"] = datasets["test"].with_transform(transform_batch)

    return datasets
