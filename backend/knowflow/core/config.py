"""
M0 · 配置管理

【为什么需要它】
新手写法是 config.py 里写死 `API_KEY = "sk-abc123"`，一旦提交到 Git 就是事故。
工程化做法：
  1. 敏感信息只放环境变量 / .env 文件（.env 进 .gitignore）
  2. 用 pydantic-settings 做类型校验 —— 配置写错（比如 base_url 少个 /v1）启动即报错，
     而不是运行到半夜调用 LLM 时才 401
  3. 配置集中一处，换模型/换环境只改 env，不改代码（12-factor）

【自查可讲】
「我们把配置从代码里剥离出来，用 pydantic-settings 做强类型校验和多环境隔离，
敏感信息走环境变量，镜像里不含任何密钥。」
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """应用配置。所有字段都可被环境变量覆盖。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # 忽略 .env 里与本模型无关的变量
    )

    # ----- LLM (MiMo) -----
    mimo_api_key: SecretStr = Field(..., description="MiMo API Key，必填")
    mimo_base_url: str = Field("https://api.xiaomimimo.com/v1", description="OpenAI 兼容网关地址")
    mimo_model: str = Field("mimo-v2.6-pro", description="默认对话模型")
    mimo_thinking: str = Field("enabled", description="思考模式 enabled/disabled")
    mimo_timeout: float = Field(120.0, description="单次请求超时（秒）")
    mimo_max_retries: int = Field(3, description="最大重试次数")

    # ----- 基础设施（生产连真组件，开发缺省降级）-----
    # 工厂函数（create_queue / create_cache / create_vector_store）读这些字段决定后端：
    #   有 URL → 生产组件；无 URL 或连不上 → 内存降级，开发零依赖可跑
    database_url: str = Field(
        "sqlite:///./knowflow.db",
        description="数据库连接串。生产 postgresql+psycopg://...，开发缺省 SQLite",
    )
    redis_url: str = Field("", description="Redis 地址，留空则降级到内存缓存")
    kafka_bootstrap_servers: str = Field("", description="Kafka broker 地址，留空则用 InMemoryQueue")
    qdrant_url: str = Field(":memory:", description="Qdrant 地址，:memory: 为进程内模式")

    # ----- Embedding（M2 引入）-----
    embedding_model: str = Field("BAAI/bge-m3", description="Embedding 模型名")
    embedding_device: str = Field("cpu", description="cpu / cuda / mps")

    # ----- 应用 -----
    app_env: str = Field("dev", description="dev / test / prod")
    log_level: str = Field("INFO")

    # ----- 成本护栏（工程化必备：防止跑飞账单）-----
    max_tokens_per_request: int = Field(4096, description="单次回复 token 上限")
    daily_cost_budget_usd: float = Field(10.0, description="日成本预算，超过则拒绝调用")

    @property
    def is_prod(self) -> bool:
        return self.app_env == "prod"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """全局单例。lru_cache 保证配置只解析一次 —— 测试时可 clear() 重载。"""
    return Settings()
