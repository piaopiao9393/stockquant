"""统一数据获取接口 - 支持Tushare和AKShare两种数据源"""

import os
from enum import Enum
from typing import List, Optional

import pandas as pd
from loguru import logger


class DataSourceType(Enum):
    """数据源类型"""
    TUSHARE = "tushare"
    AKSHARE = "akshare"


class UnifiedDataFetcher:
    """统一数据获取器 - 自动选择Tushare或AKShare"""

    def __init__(
        self,
        source: Optional[DataSourceType] = None,
        token: Optional[str] = None,
        disable_proxy: bool = True,
    ):
        """初始化

        Args:
            source: 数据源类型，None则自动选择（优先Tushare，无token则用AKShare）
            token: Tushare API Token
            disable_proxy: 是否禁用代理
        """
        # 禁用代理
        if disable_proxy:
            for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                os.environ.pop(key, None)

        self.source_type = source or self._auto_select_source(token)
        self.fetcher = self._create_fetcher(token)

        logger.info(f"使用数据源: {self.source_type.value}")

    def _auto_select_source(self, token: Optional[str]) -> DataSourceType:
        """自动选择数据源"""
        if token or os.getenv('TUSHARE_TOKEN'):
            return DataSourceType.TUSHARE
        else:
            logger.info("未检测到Tushare Token，使用AKShare作为数据源")
            return DataSourceType.AKSHARE

    def _create_fetcher(self, token: Optional[str]):
        """创建具体的数据获取器"""
        if self.source_type == DataSourceType.TUSHARE:
            from .fetcher import TushareFetcher
            return TushareFetcher(token)
        else:
            from .akshare_fetcher import AKShareFetcher
            return AKShareFetcher(disable_proxy=False)  # 已经在上面禁用了

    def get_stock_list(self) -> pd.DataFrame:
        """获取股票列表"""
        return self.fetcher.get_stock_list()

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取日线数据"""
        return self.fetcher.get_daily_prices(symbol, start_date, end_date)

    def get_daily_indicator(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取每日估值指标"""
        return self.fetcher.get_daily_indicator(symbol, start_date, end_date)

    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取交易日历"""
        return self.fetcher.get_trade_calendar(start_date, end_date)

    def get_latest_trade_date(self) -> str:
        """获取最新交易日"""
        return self.fetcher.get_latest_trade_date()

    def is_trade_date(self, date_str: Optional[str] = None) -> bool:
        """判断是否为交易日"""
        return self.fetcher.is_trade_date(date_str)

    def batch_fetch(
        self,
        symbols: List[str],
        fetch_func,
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """批量获取数据"""
        return self.fetcher.batch_fetch(symbols, fetch_func, batch_size, delay)


def create_data_fetcher(
    prefer_source: Optional[str] = None,
    token: Optional[str] = None,
) -> UnifiedDataFetcher:
    """创建数据获取器的工厂函数

    Args:
        prefer_source: 优先使用的数据源 ('tushare' 或 'akshare')
        token: Tushare API Token

    Returns:
        UnifiedDataFetcher 实例

    Examples:
        >>> # 自动选择（优先Tushare）
        >>> fetcher = create_data_fetcher()
        >>>
        >>> # 指定使用AKShare（免费，无需Token）
        >>> fetcher = create_data_fetcher(prefer_source='akshare')
        >>>
        >>> # 指定使用Tushare
        >>> fetcher = create_data_fetcher(prefer_source='tushare', token='your_token')
    """
    source_map = {
        'tushare': DataSourceType.TUSHARE,
        'akshare': DataSourceType.AKSHARE,
    }

    source = source_map.get(prefer_source) if prefer_source else None
    return UnifiedDataFetcher(source=source, token=token)
