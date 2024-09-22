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
    src_dir: Path = Path("data/KoreanFOOD_Detecting"),
    data_type: str = "train",
    classes: List[str] = ["jjajangmyeon", "Jjambbong"],
    num_sample: int = 100,
):
    """주어진 데이터셋에서 필요한 데이터를 선택하고 필터링하여 저장합니다.

    이 함수는 YOLOv5 형식의 데이터셋에서 특정 클래스에 해당하는 데이터만 필터링하여
    지정한 경로에 저장합니다. 저장된 데이터는 주어진 클래스에 해당하는 이미지와
    라벨(annotation)만 포함합니다.

    Args:
        save_dir (Path): 필터링된 데이터를 저장할 경로입니다.
        src_dir (Path, optional): 원본 데이터가 저장되어 있는 경로입니다.
            이 디렉토리 하위 경로에는 data.yaml, test/ train/ valid/ 가 포함되어야 합니다.
            기본값은 'data/KoreanFOOD_Detecting'입니다.
        data_type (str, optional): 저장할 데이터의 유형을 지정합니다.
            'train', 'valid', 'test' 중 하나여야 합니다. 기본값은 'train'입니다.
        classes (List[str], optional): 필터링할 클래스 목록입니다. 기본값은
            ['jjajangmyeon', 'Jjambbong']입니다.
        num_sample (int, optional): 필터링된 데이터에서 추출할 샘플의 개수입니다.
            기본값은 100입니다.

    Raises:
        ValueError: 잘못된 클래스가 입력되었거나, 이미 경로가 존재하는 경우, 또는
            잘못된 data_type 값이 전달될 경우 발생합니다.

    Example:
        기본 사용법:

        ```sh
        python src/filter_data.py \
            data/filtered_data \
            --src-dir data/KoreanFOOD_Detecting \
            --data-type train \
            --classes jjajangmyeon \
            --classes Jjambbong \
            --num-sample 100
        ```

        위 예시는 'jjajangmyeon'과 'Jjambbong' 클래스에 해당하는 100개의 이미지를
        필터링하여 'filtered_data/train' 경로에 저장합니다.
    """
    # 원본 데이터 불러오기
    ds = sv.DetectionDataset.from_coco(
        images_directory_path=src_dir / data_type,
        annotations_path=src_dir / data_type / "_annotations.coco.json",
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

    ds_filtered.as_coco(
        images_directory_path=save_dir / data_type,
        annotations_path=save_dir / data_type / "_annotations.coco.json",
    )


if __name__ == "__main__":
    typer.run(main)
