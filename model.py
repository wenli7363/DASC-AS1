from transformers import AutoModelForObjectDetection, AutoImageProcessor
from constants import ID2LABEL as ID_TO_LABEL, LABEL2ID as LABEL_TO_ID

def initialize_model():
    """
    Initialize a model for object detection.

    Returns:
        A model for object detection.

    NOTE: Below is an example of how to initialize a model for object detection.

    from transformers import AutoModelForObjectDetection
    from constants import ID_TO_LABEL, LABEL_TO_ID, MODEL_NAME

    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME,  # specify the model checkpoint
        id2label=ID_TO_LABEL,  # map of label id to label name
        label2id=LABEL_TO_ID,  # map of label name to label id
        ignore_mismatched_sizes=True,  # allow replacing the classification head
    )

    You are free to change this.
    But make sure the model meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """

    model = AutoModelForObjectDetection.from_pretrained(
        pretrained_model_name_or_path="qubvel-hf/detr-resnet-50-finetuned-10k-cppe5",  # specify the model checkpoint
        id2label=ID_TO_LABEL,  # map of label id to label name
        label2id=LABEL_TO_ID,  # map of label name to label id
        ignore_mismatched_sizes=True,  # allow replacing the classification head
    )
    return model


def initialize_processor():
    """
    Initialize a processor for object detection.

    Returns:
        A processor for object detection.

    NOTE: Below is an example of how to initialize a processor for object detection.

    from transformers import AutoImageProcessor
    from constants import MODEL_NAME

    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path=MODEL_NAME
    )

    You are free to change this.
    But make sure the processor meets the requirements of the `transformers.Trainer` API.
    ref: https://huggingface.co/transformers/main_classes/trainer.html#transformers.Trainer
    """
    processor = AutoImageProcessor.from_pretrained(
        pretrained_model_name_or_path="qubvel-hf/detr-resnet-50-finetuned-10k-cppe5",
        use_fast=False
    )
    return processor
