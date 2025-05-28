from huggingface_hub import upload_folder, login
import os
from dotenv import load_dotenv

# 第一步：登录 Hugging Face
load_dotenv()  # 加载 .env 文件中的环境变量
login(token=os.getenv("HUGGINGFACE_KEY")) 

# 第二步：上传整个文件夹
upload_folder(
    folder_path="record/k_ablation",  # 本地文件夹路径
    repo_id="RealFantasy/IKEA2025",  # 目标仓库 ID
    path_in_repo="k_ablation",  # 仓库内路径，空表示上传到根目录
    repo_type="dataset",  # 也可以是 "model" 或 "space"
)
