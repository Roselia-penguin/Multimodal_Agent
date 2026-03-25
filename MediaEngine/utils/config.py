import os

class settings:
    # LLM配置 (支持DeepSeek等OpenAI兼容API)
    MEDIA_ENGINE_API_KEY = os.getenv("MEDIA_ENGINE_API_KEY", "sk-c41fb7b470b7458b8fd0e82cf2efc32d")
    MEDIA_ENGINE_MODEL_NAME = os.getenv("MEDIA_ENGINE_MODEL_NAME", "deepseek-chat")
    MEDIA_ENGINE_BASE_URL = os.getenv("MEDIA_ENGINE_BASE_URL", "https://api.deepseek.com/v1")

    # 搜索API配置 (请至少配置一个)
    BOCHA_WEB_SEARCH_API_KEY = os.getenv("BOCHA_API_KEY", "sk-89bf0be4774e4b0da421e96f10ef4deb")
    BOCHA_BASE_URL = os.getenv("BOCHA_BASE_URL", "https://api.bocha.cn/v1/ai-search")