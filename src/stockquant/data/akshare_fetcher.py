"""AKShare数据获取封装 - 免费开源数据源"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd
from loguru import logger


class AKShareFetcher:
    """AKShare数据获取器 - 完全免费，无需注册"""

    def __init__(self, disable_proxy: bool = True):
        """初始化

        Args:
            disable_proxy: 是否禁用代理（解决某些网络环境下的连接问题）
        """
        # 禁用代理（如果请求）- 解决WinError 10053等连接问题
        if disable_proxy:
            for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
                os.environ.pop(key, None)

        try:
            import akshare as ak
            self.ak = ak
            # 设置AKShare输出不显示进度条
            self.ak.set_streamlit_page_config = lambda x: None
            logger.info("AKShare API初始化完成")
        except ImportError:
            logger.error("AKShare未安装，请运行: pip install akshare")
            raise

    def get_stock_list(self, max_retries: int = 3) -> pd.DataFrame:
        """获取股票列表"""
        for attempt in range(max_retries):
            try:
                # 获取A股列表 - 使用简单的实时行情接口
                logger.info("正在获取股票列表...")
                df = self.ak.stock_zh_a_spot_em()

                # 标准化列名以兼容Tushare格式
                df = df.rename(columns={
                    '代码': 'symbol',
                    '名称': 'name',
                    '最新价': 'close',
                    '涨跌幅': 'pct_chg',
                    '成交量': 'vol',
                    '成交额': 'amount',
                    '总市值': 'total_mv',
                    '流通市值': 'circ_mv',
                })

                # 创建ts_code格式
                df['ts_code'] = df['symbol'].apply(self._symbol_to_ts_code)

                logger.info(f"获取股票列表: {len(df)}只")
                return df
            except Exception as e:
                logger.warning(f"获取股票列表失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # 指数退避
                else:
                    logger.error(f"获取股票列表失败: {e}")
                    # 返回空的DataFrame而不是抛出异常
                    return pd.DataFrame(columns=['symbol', 'name', 'ts_code'])

    def get_daily_prices(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """获取日线数据

        Args:
            symbol: 股票代码，如"000001"或"000001.SZ"
            start_date: 开始日期，格式YYYYMMDD或YYYY-MM-DD
            end_date: 结束日期，格式YYYYMMDD或YYYY-MM-DD
            max_retries: 最大重试次数
        """
        for attempt in range(max_retries):
            try:
                # 转换日期格式
                if start_date:
                    start_date = start_date.replace('-', '')
                if end_date:
                    end_date = end_date.replace('-', '')

                # 转换symbol格式
                pure_symbol = symbol.split('.')[0]

                # 获取日线数据
                logger.debug(f"获取 {symbol} 日线数据...")
                df = self.ak.stock_zh_a_hist(
                    symbol=pure_symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq"  # 前复权
                )

                if df is None or df.empty:
                    return pd.DataFrame()

                # 标准化列名
                df = df.rename(columns={
                    '日期': 'trade_date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'vol',
                    '成交额': 'amount',
                    '振幅': 'amplitude',
                    '涨跌幅': 'pct_chg',
                    '涨跌额': 'change',
                    '换手率': 'turnover',
                })

                # 添加ts_code
                df['ts_code'] = self._symbol_to_ts_code(pure_symbol)

                # 转换日期格式
                df['trade_date'] = pd.to_datetime(df['trade_date'])

                # 添加复权价格（前复权后的价格已经是复权价格）
                df['adj_factor'] = 1.0
                df['adj_open'] = df['open']
                df['adj_high'] = df['high']
                df['adj_low'] = df['low']
                df['adj_close'] = df['close']

                logger.debug(f"{symbol} 日线数据: {len(df)}条")
                return df

            except Exception as e:
                logger.warning(f"获取{symbol}日线数据失败 (尝试 {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    logger.error(f"获取{symbol}日线数据失败: {e}")
                    return pd.DataFrame()

    def get_daily_indicator(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取每日估值指标"""
        try:
            # AKShare实时行情已包含估值指标
            df = self.get_daily_prices(symbol, start_date, end_date)

            if df.empty:
                return pd.DataFrame()

            # 获取最新估值数据
            pure_symbol = symbol.split('.')[0]
            spot_df = self.ak.stock_zh_a_spot_em()
            spot_row = spot_df[spot_df['代码'] == pure_symbol]

            if not spot_row.empty:
                # 添加估值指标
                df['pe'] = spot_row['市盈率-动态'].values[0] if '市盈率-动态' in spot_row.columns else None
                df['pb'] = spot_row['市净率'].values[0] if '市净率' in spot_row.columns else None
                df['total_mv'] = spot_row['总市值'].values[0] if '总市值' in spot_row.columns else None
                df['circ_mv'] = spot_row['流通市值'].values[0] if '流通市值' in spot_row.columns else None
                df['turnover_rate'] = spot_row['换手率'].values[0] if '换手率' in spot_row.columns else None

            logger.debug(f"{symbol} 估值指标: {len(df)}条")
            return df

        except Exception as e:
            logger.error(f"获取{symbol}估值指标失败: {e}")
            return pd.DataFrame()

    def get_financial_indicator(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """获取财务指标"""
        try:
            pure_symbol = symbol.split('.')[0]

            # 获取主要财务指标
            df = self.ak.stock_financial_analysis_indicator(symbol=pure_symbol)

            if df is None or df.empty:
                return pd.DataFrame()

            # 标准化列名
            df = df.rename(columns={
                '报告日': 'end_date',
                '摊薄每股收益': 'eps',
                '加权每股收益': 'dt_eps',
                '每股净资产': 'bps',
                '每股经营现金流': 'mgjyxjje',
                '营业总收入': 'total_revenue',
                '营业总收入同比': 'total_revenue_yoy',
                '营业利润': 'op_profit',
                '营业利润同比': 'op_profit_yoy',
                '净利润': 'net_profit',
                '净利润同比': 'net_profit_yoy',
                '销售毛利率': 'gross_profit_margin',
                '加权净资产收益率': 'roe',
                '摊薄净资产收益率': 'roe_dt',
                '资产负债率': 'debt_to_assets',
            })

            df['ts_code'] = self._symbol_to_ts_code(pure_symbol)
            df['end_date'] = pd.to_datetime(df['end_date'])

            # 按日期筛选
            if start_date:
                start = pd.to_datetime(start_date)
                df = df[df['end_date'] >= start]
            if end_date:
                end = pd.to_datetime(end_date)
                df = df[df['end_date'] <= end]

            logger.debug(f"{symbol} 财务指标: {len(df)}条")
            return df

        except Exception as e:
            logger.error(f"获取{symbol}财务指标失败: {e}")
            return pd.DataFrame()

    def get_trade_calendar(self, start_date: str, end_date: str) -> pd.DataFrame:
        """获取交易日历"""
        try:
            # AKShare获取交易日历
            df = self.ak.tool_trade_date_hist_sina()

            df = df.rename(columns={
                'trade_date': 'cal_date',
            })

            df['cal_date'] = pd.to_datetime(df['cal_date'])
            df['is_open'] = 1

            # 按日期筛选
            start = pd.to_datetime(start_date)
            end = pd.to_datetime(end_date)
            df = df[(df['cal_date'] >= start) & (df['cal_date'] <= end)]

            return df
        except Exception as e:
            logger.error(f"获取交易日历失败: {e}")
            return pd.DataFrame()

    def is_trade_date(self, date_str: Optional[str] = None) -> bool:
        """判断是否为交易日"""
        if date_str is None:
            date_str = datetime.now().strftime("%Y%m%d")

        try:
            df = self.get_trade_calendar(date_str, date_str)
            return not df.empty
        except Exception as e:
            logger.error(f"判断交易日失败: {e}")
            return False

    def get_latest_trade_date(self) -> str:
        """获取最新交易日"""
        today = datetime.now()

        try:
            df = self.ak.tool_trade_date_hist_sina()
            df['trade_date'] = pd.to_datetime(df['trade_date'])

            # 过滤未来日期
            df = df[df['trade_date'] <= today]

            if not df.empty:
                latest = df['trade_date'].max()
                return latest.strftime("%Y%m%d")
        except Exception as e:
            logger.error(f"获取最新交易日失败: {e}")

        return (today - timedelta(days=1)).strftime("%Y%m%d")

    def _symbol_to_ts_code(self, symbol: str) -> str:
        """转换symbol为ts_code格式"""
        symbol = symbol.strip()
        if '.' in symbol:
            return symbol

        # 判断交易所
        if symbol.startswith('6'):
            return f"{symbol}.SH"
        else:
            return f"{symbol}.SZ"

    def batch_fetch(
        self,
        symbols: List[str],
        fetch_func,
        batch_size: int = 10,
        delay: float = 0.5,
    ) -> pd.DataFrame:
        """批量获取数据"""
        results = []

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            for symbol in batch:
                try:
                    df = fetch_func(symbol)
                    if not df.empty:
                        results.append(df)
                except Exception as e:
                    logger.warning(f"获取{symbol}数据失败: {e}")

                time.sleep(delay)

            logger.info(f"批次 {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} 完成")

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()
