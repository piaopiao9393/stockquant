"""配置管理 - 使用Pydantic Settings"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DataConfig(BaseSettings):
    """数据配置"""

    model_config = SettingsConfigDict(env_prefix="DATA_")

    # 数据源: akshare (默认,免费) / tushare (需Token)
    default_source: str = Field(default="akshare", description="默认数据源")

    # Tushare配置
    tushare_token: str = Field(default="", description="Tushare API Token")

    # 数据库配置（SQLite默认，可切换PostgreSQL）
    database_url: str = Field(
        default="sqlite:///data_storage/stockquant.db",
        description="数据库连接URL"
    )

    # 数据更新设置
    update_time: str = Field(default="15:35", description="每日数据更新时间 (HH:MM)")
    auto_update: bool = Field(default=True, description="是否自动更新")

    # 数据质量
    min_history_days: int = Field(default=60, description="最小历史数据天数")


class TradingConfig(BaseSettings):
    """交易配置"""

    model_config = SettingsConfigDict(env_prefix="TRADING_")

    # 交易模式: paper(模拟)/live(实盘)
    mode: str = Field(default="paper", description="交易模式")

    # 执行模式: manual(手动确认)/auto(自动)
    execution_mode: str = Field(default="manual", description="执行模式")

    # 账户设置
    initial_cash: float = Field(default=50000.0, description="初始资金")
    commission_rate: float = Field(default=0.0003, description="佣金费率")
    min_commission: float = Field(default=5.0, description="最低佣金")
    tax_rate: float = Field(default=0.001, description="印花税（卖出）")

    # 风控参数
    max_position_pct: float = Field(default=0.25, ge=0, le=1, description="单票最大仓位")
    max_total_position: float = Field(default=0.8, ge=0, le=1, description="总仓位上限")
    max_daily_loss_pct: float = Field(default=0.05, ge=0, le=1, description="单日最大亏损")
    max_drawdown_pct: float = Field(default=0.15, ge=0, le=1, description="最大回撤限制")

    # 订单限制
    max_orders_per_day: int = Field(default=20, ge=1, description="每日最大订单数")


class NotificationConfig(BaseSettings):
    """通知配置"""

    model_config = SettingsConfigDict(env_prefix="")

    # SMTP配置
    smtp_host: str = Field(default="", description="SMTP服务器")
    smtp_port: int = Field(default=587, description="SMTP端口")
    smtp_user: str = Field(default="", description="邮箱账号")
    smtp_password: str = Field(default="", description="邮箱密码")
    email_to: List[str] = Field(default=[], description="收件人列表")

    @field_validator("email_to", mode="before")
    @classmethod
    def parse_email_list(cls, v):
        """解析逗号分隔的邮箱列表"""
        if isinstance(v, str):
            return [email.strip() for email in v.split(",") if email.strip()]
        return v


class Settings(BaseSettings):
    """全局配置"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 环境
    env: str = Field(default="development", description="运行环境")
    debug: bool = Field(default=False, description="调试模式")
    log_level: str = Field(default="INFO", description="日志级别")

    # 子配置
    data: DataConfig = Field(default_factory=DataConfig)
    trading: TradingConfig = Field(default_factory=TradingConfig)
    notification: NotificationConfig = Field(default_factory=NotificationConfig)

    # 路径配置
    project_root: Path = Field(default=Path(__file__).parent.parent.parent.parent)

    @property
    def data_dir(self) -> Path:
        """数据目录"""
        path = self.project_root / "data_storage"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def log_dir(self) -> Path:
        """日志目录"""
        path = self.project_root / "logs"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def plan_dir(self) -> Path:
        """交易计划目录"""
        path = self.project_root / "plans"
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache()
def get_settings() -> Settings:
    """获取配置单例"""
    return Settings()
