"""주어진 데이터셋에서 필요한 데이터를 선택합니다."""

import random
import warnings
from pathlib import Path
from typing import List

import numpy as np
import supervision as sv
import typer
from tqdm import tqdm

warnings.filterwarnings("ignore")


def main(
    save_dir: Path,
    src_image_dir: Path = "data/KoreanFOOD_Detecting/test/images",
    src_annot_dir: Path = "data/KoreanFOOD_Detecting/test/labels",
    src_datayaml_dir: Path = "data/KoreanFOOD_Detecting/data.yaml",
    data_type: str = "test",
    classes: List[str] = ["jjajangmyeon", "Jjambbong"],
    num_sample: int = 30,
):
    # 원본 데이터 불러오기
    ds = sv.DetectionDataset.from_yolo(
        images_directory_path=src_image_dir,
        annotations_directory_path=src_annot_dir,
        data_yaml_path=src_datayaml_dir,
    )

    # classes 명 확인
    wrong_classes = [cls for cls in classes if cls not in ds.classes]
    if len(wrong_classes) > 0:
        raise ValueError(f"class {wrong_classes} does not exists.")

    # 저장 경로 확인
    if (save_dir / data_type).exists():
        raise ValueError(f"path '{str(save_dir/data_type)}' already exists.")

    # data_type 값 확인
    if data_type not in ["train", "valid", "test"]:
        raise ValueError("data teyp must be one of ['train', 'valid', 'test']")

    # 필터링
    classes.sort()

    images_filtered = {}
    annot_filtered = {}
    for path, image, annotation in tqdm(ds):
        annots_classes = [ds.classes[cid] for cid in annotation.class_id]
        if intersections := set(annots_classes) & set(classes):
            images_filtered[path] = image
            indices = [annots_classes.index(cls) for cls in intersections]
            annot_filtered[path] = sv.Detections(
                xyxy=annotation.xyxy[indices],
                class_id=np.array([classes.index(annots_classes[i]) for i in indices]),
            )

    # 개수 제한 필터링
    if num_sample is not None:
        image_paths_sample = random.sample(list(images_filtered.keys()), num_sample)
        images_filtered = {path: images_filtered[path] for path in image_paths_sample}
        annot_filtered = {path: annot_filtered[path] for path in image_paths_sample}

    # 결과 저장
    ds_filtered = sv.DetectionDataset(
        classes=classes,
        images=images_filtered,
        annotations=annot_filtered,
    )

    ds_filtered.as_yolo(
        images_directory_path=save_dir / data_type / "images",
        annotations_directory_path=save_dir / data_type / "labels",
        data_yaml_path=save_dir / "data.yaml",
    )


if __name__ == "__main__":
    typer.run(main)
