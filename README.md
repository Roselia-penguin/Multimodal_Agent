# Media Agent（多模态媒体分析智能体）

这是一个面向视频/图片等多模态内容的“媒体分析智能体”项目。它会：
1. 通过 Bocha 的多模态搜索获取网页、图片以及可能的结构化信息（modal cards）
2. 对搜索到的“疑似视频”做视频理解（抽关键帧 + 图像描述 + 音频转写）
3. 对文本/视觉线索做情感分析（当前情感模型主要使用文本与视觉；音频部分在情感技能里暂未启用）
4. 将增强后的多模态信息交给 LLM 生成总结/报告段落

---

## 目录结构

- `MediaEngine/`: 核心实现
  - `agent.py`: 主智能体 `MediaAgent`（搜索 -> 增强 -> 总结）
  - `search.py`: `BochaMultimodalSearch`（与 Bocha 搜索 API 交互）
  - `base.py`: `LLMClient`（OpenAI 兼容接口调用，含重试装饰器）
  - `prompts.py`: 各阶段提示词与 JSON Schema（用于 LLM 结构化输出）
  - `skills/`: 具体技能
    - `video_skill.py`: `VideoUnderstandingSkill`（关键帧、字幕/转写）
    - `multimodal_sentiment_skill.py`: `MultimodalSentimentSkill`（文本/视觉情感）
    - `structured_data_skill.py`: `StructuredDataSkill`（modal cards / 结构化数据）
  - `nodes/`: 输出节点
    - `summary_node.py`: `SummaryNode`（把增强信息拼进总结提示词）
- `ffmpeg/`: ffmpeg 运行文件（`test.py` 会尝试把 `ffmpeg/bin` 加入 `PATH`）
- `test.py`: 当前项目的最小运行示例

---

## 快速开始

### 1) 准备环境

推荐使用 Python 3.10+，并在项目根目录创建虚拟环境后安装依赖。

依赖主要来自代码中的 import（不包含显式 requirements 文件），可参考下面的包列表安装：

```bash
pip install loguru openai requests opencv-python numpy pillow transformers faster-whisper torch
pip install beautifulsoup4 pandas
```

说明：
- `torch` 需要你本机 CUDA/CPU 环境匹配（GPU/CPU 安装方式不同）
- `faster-whisper` 在解析音频时通常依赖 `ffmpeg`

### 2) 配置 API Key（必需）

项目通过环境变量读取配置（见 `MediaEngine/utils/config.py`）。请至少设置：

```powershell
# LLM（OpenAI 兼容）相关
$env:MEDIA_ENGINE_API_KEY="你的_LLM_API_KEY"
$env:MEDIA_ENGINE_MODEL_NAME="deepseek-chat"  # 或你自己的模型名
$env:MEDIA_ENGINE_BASE_URL="https://api.deepseek.com/v1"  # 或你的 OpenAI 兼容 base_url

# Bocha 搜索相关
$env:BOCHA_API_KEY="你的_BOCHA_API_KEY"
$env:BOCHA_BASE_URL="https://api.bocha.cn/v1/ai-search"
```

重要安全提示：`MediaEngine/utils/config.py` 中目前也写了默认字符串 key（用于开发占位）。实际使用请务必用环境变量覆盖，并避免把真实密钥提交到仓库。

### 3) 确保 ffmpeg 可用

`test.py` 会检查 `ffmpeg/bin` 是否存在并加入 `PATH`：
- 若存在：`faster-whisper` 解码音频会更稳定
- 若不存在：可能导致转写失败或报错

---

## 运行

在项目根目录直接运行示例：

```powershell
python test.py
```

输出会打印最终总结文本：
- `result['summary']`

`test.py` 里当前示例查询为：`张雪峰去世舆情`。

---

## 作为库使用（示例）

你也可以直接在自己的脚本里调用：

```python
from MediaEngine import MediaAgent

agent = MediaAgent()
result = agent.run_analysis("你的查询/话题")
print(result["summary"])
```

`run_analysis(query)` 内部完成：`search(query)` -> 多技能增强 -> `summarize(...)`。

---

## 关键实现行为（帮助你理解输出）

### 视频理解（`VideoUnderstandingSkill`）
- 从视频中等间隔抽取固定数量关键帧（默认 `num_frames=5`）
- 对每帧使用 BLIP 生成图片描述（caption）
- 使用 `faster-whisper` 对视频音频进行中文转写（`language="zh"`）
- 处理完后会删除临时下载的本地视频文件

### 多模态情感（`MultimodalSentimentSkill`）
- 文本情感：`nlptown/bert-base-multilingual-uncased-sentiment`
- 视觉情感：`trpakov/vit-face-expression`（按情绪类别做正负/中性映射）
- 当前代码里音频部分暂未启用（类注释也写明了这一点）

### 结构化数据（`StructuredDataSkill`）
- 主要解析搜索结果中的 `modal_cards`
- 代码中提供了 HTML 表格解析与从文本提取 key-value 的能力，但在当前流程里（`MediaAgent`）只把 `modal_cards` 直接拼进 LLM 总结上下文

---

## 常见问题

1. **转写失败/找不到 ffmpeg**
   - 确保存在 `ffmpeg/bin/ffmpeg(.exe)`，并且 `python test.py` 时 `test.py` 的 PATH 注入没有被跳过

2. **Bocha 搜索报 401/403**
   - 检查 `BOCHA_API_KEY` 是否正确，以及 `BOCHA_BASE_URL` 是否与你的服务一致

3. **LLM 报模型不可用**
   - 检查 `MEDIA_ENGINE_MODEL_NAME` 与 `MEDIA_ENGINE_BASE_URL` 的组合是否匹配

---

## 下一步建议（可选）

如果你希望我继续完善 README（例如补全 `requirements.txt` 或增加更完整的“报告生成流程/多轮搜索结构”说明），告诉我你希望的侧重点即可。

