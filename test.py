import os
import sys

# 获取项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 设置 ffmpeg 路径（假设你已将 ffmpeg 解压到项目目录下的 ffmpeg/bin）
ffmpeg_bin = os.path.join(project_root, "ffmpeg", "bin")
if os.path.exists(ffmpeg_bin):
    os.environ["PATH"] = ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")
    print(f"[INFO] Added ffmpeg to PATH: {ffmpeg_bin}")
else:
    print(f"[WARN] ffmpeg not found at {ffmpeg_bin}")

# 然后导入 MediaEngine
from MediaEngine import MediaAgent

def main():
    agent = MediaAgent()
    query = "张雪峰去世舆情"
    result = agent.run_analysis(query)

    print("=" * 80)
    print("最终报告摘要：")
    print("=" * 80)
    print(result['summary'])

if __name__ == "__main__":
    main()