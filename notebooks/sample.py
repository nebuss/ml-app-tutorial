# %%
# 강의 진행에 필요한 패키지가 설치되었는지 확인
import os
import cv2
import mlflow
import torch
import typer
import ultralytics
from dotenv import load_dotenv

load_dotenv()


# pythonpath가 제대로 설정되었는지 확인
from src.sample import sample_print

# %%
print("sample notebook")

# %%
sample_print()

# %%

# 환경 변수가 제대로 설정되었는지 확인
print(f"WORKING_DIRECTORY={os.environ.get('WORKING_DIRECTORY')}")
print(f"PYTHONPATH={os.environ.get('PYTHONPATH')}")
print(f"MLFLOW_URI={os.environ.get('MLFLOW_URI')}")
