import os
import time

from dataset import build_dataset, add_preprocessing
from model import initialize_model, initialize_processor
from trainer import build_trainer
from utils import not_change_test_dataset
from pprint import pprint
from utils import set_random_seeds

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 过滤掉有问题的下标
def filter_problematic_indices(dataset, indices_to_remove):
    return dataset.filter(lambda example, idx: idx not in indices_to_remove, with_indices=True)

def main():
    """
    Main function to execute model training and evaluation.
    """
    # Set seed for reproducibility
    set_random_seeds()

    # Build the dataset
    raw_datasets = build_dataset()
    # 有问题的下标
    problematic_train = [5, 65, 188, 234, 383, 389, 392, 448, 476, 495, 525, 555, 642, 780]
    problematic_validation = [24,25]

    # 处理后的数据集
    filtered_train_dataset = filter_problematic_indices(raw_datasets["train"], problematic_train)
    filtered_validation_dataset = filter_problematic_indices(raw_datasets["validation"], problematic_validation)
    # 更新原始数据集
    raw_datasets["train"] = filtered_train_dataset
    raw_datasets["validation"] = filtered_validation_dataset

    assert not_change_test_dataset(raw_datasets), "You should not change the test dataset"

    # Initialize the image processor
    processor = initialize_processor()

    # Add preprocessing to the dataset
    datasets = add_preprocessing(raw_datasets, processor)

    # Build the object detection model
    model = initialize_model()

    # Build and train the model
    trainer = build_trainer(
        model=model,
        image_processor=processor,
        datasets=datasets,
    )

    start_time = time.time()
    trainer.train()
    end_time = time.time()
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")

    # Evaluate the model on the test dataset
    test_metrics = trainer.evaluate(
        eval_dataset=datasets["test"],
        metric_key_prefix="test",
    )
    pprint(test_metrics)


if __name__ == "__main__":
    main()
