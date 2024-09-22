import torch
import typer
from PIL import Image


def inference(weight_path, image):
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weight_path)

    model.conf = 0.03  # NMS confidence threshold
    model.iou = 0.3  # NMS IoU threshold
    model.agnostic = False  # NMS class-agnostic
    model.multi_label = False  # NMS multiple labels per box
    model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    model.max_det = 1000  # maximum number of detections per image
    model.amp = False  # Automatic Mixed Precision (AMP) inference
    results = model(image, size=640)
    return results


def main(weight_path, image_path):
    image = Image.open(image_path)  # PIL image
    results = inference(weight_path=weight_path, image=image)
    results.save()


if __name__ == "__main__":
    typer.run(main)
