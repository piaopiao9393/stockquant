"""LLM 策略 - 集成 LLM 挖掘的因子到策略框架"""

from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger

from stockquant.factors import FactorMeta, FactorRegistry
from stockquant.llm.core import FactorAST
from stockquant.llm.miner import LLMFactorMiner, MiningConfig
from stockquant.strategy.core.strategy import MultiFactorStrategy, Signal


class LLMFactorStrategy:
    """LLM 因子策略 - 自动挖掘和使用 LLM 生成的因子"""

    def __init__(
        self,
        miner: LLMFactorMiner,
        base_strategy: Optional[MultiFactorStrategy] = None,
        max_llm_factors: int = 3,
        filters: Optional[Dict] = None,
        max_holdings: int = 10,
        top_pct: float = 0.1,
    ):
        """
        Args:
            miner: LLM 因子挖掘器
            base_strategy: 基础多因子策略（可选，用于组合）
            max_llm_factors: 最大使用 LLM 因子数量
            filters: 股票池过滤条件
            max_holdings: 最大持仓数
            top_pct: 选股百分位
        """
        self.miner = miner
        self.base_strategy = base_strategy
        self.max_llm_factors = max_llm_factors
        self.filters = filters or {}
        self.max_holdings = max_holdings
        self.top_pct = top_pct

        self.discovered_factors: List[FactorAST] = []
        self.factor_weights: Dict[str, float] = {}
        self.is_mined = False

    async def mine_factors(
        self,
        market_context: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        returns: Dict[str, List[float]],
    ) -> bool:
        """挖掘因子并注册到系统中

        Args:
            market_context: 市场上下文
            market_data: 市场数据 {symbol: DataFrame}
            returns: 收益率数据 {symbol: [returns]}

        Returns:
            是否成功挖掘到因子
        """
        context = {
            **market_context,
            "market_data": market_data,
            "returns": returns,
        }

        logger.info(f"开始 LLM 因子挖掘，目标: {self.max_llm_factors} 个因子")

        result = await self.miner.mine(context, num_factors=self.max_llm_factors)

        if not result.success or not result.factors:
            logger.warning("LLM 因子挖掘未找到有效因子")
            return False

        self.discovered_factors = result.factors[:self.max_llm_factors]

        # 注册因子到系统中
        for factor in self.discovered_factors:
            self._register_factor(factor)

        # 计算因子权重（基于 IC 均值）
        self._calculate_weights()

        self.is_mined = True

        logger.info(
            f"成功挖掘并注册 {len(self.discovered_factors)} 个 LLM 因子: "
            f"{[f.name for f in self.discovered_factors]}"
        )

        return True

    def _register_factor(self, factor: FactorAST) -> None:
        """将 FactorAST 注册到因子注册中心"""
        meta = FactorMeta(
            name=factor.name,
            category="llm",
            direction=factor.direction,
            required_fields=["close", "open", "high", "low", "vol"],
            description=factor.description,
            min_history=20,
        )

        # 创建执行函数
        executor = factor.to_executable()

        # 包装成注册需要的形式
        @FactorRegistry.register(meta)
        def calc_llm_factor(df: pd.DataFrame) -> float:
            try:
                # 计算因子值并返回最新值
                values = executor(df)
                if values is not None and len(values) > 0:
                    return float(values[-1])
                return 0.0
            except Exception as e:
                logger.debug(f"计算 LLM 因子 {factor.name} 失败: {e}")
                return 0.0

    def _calculate_weights(self) -> None:
        """计算因子权重"""
        if not self.discovered_factors:
            return

        # 基于 IC 均值计算权重
        ics = []
        for factor in self.discovered_factors:
            ic = factor.metrics.get("ic_mean", 0)
            ics.append(max(ic, 0))  # 只使用正的 IC

        total_ic = sum(ics)
        if total_ic > 0:
            self.factor_weights = {
                f.name: ic / total_ic
                for f, ic in zip(self.discovered_factors, ics)
            }
        else:
            # 等权
            n = len(self.discovered_factors)
            self.factor_weights = {f.name: 1.0 / n for f in self.discovered_factors}

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """生成交易信号

        Args:
            data: {股票代码: DataFrame}

        Returns:
            信号列表
        """
        if not self.is_mined:
            logger.warning("尚未挖掘因子，请先调用 mine_factors()")
            return []

        # 合并基础策略和 LLM 因子
        all_factors = {}
        all_weights = {}

        # 添加基础策略的因子
        if self.base_strategy:
            all_factors.update(self.base_strategy.factors)

        # 添加 LLM 因子
        all_factors.update(self.factor_weights)

        # 创建临时策略生成信号
        strategy = MultiFactorStrategy(
            name="LLM_MultiFactor",
            factors=all_factors,
            filters=self.filters,
            max_holdings=self.max_holdings,
            top_pct=self.top_pct,
        )

        # 生成信号
        signals = strategy.generate_signals(data)

        # 添加 LLM 因子信息到信号元数据
        for signal in signals:
            signal.metadata["llm_factors"] = [
                {
                    "name": f.name,
                    "description": f.description,
                    "ic": f.metrics.get("ic_mean", 0),
                    "expression": f.to_expression(),
                }
                for f in self.discovered_factors
            ]

        return signals

    def get_factor_report(self) -> Dict[str, Any]:
        """获取因子挖掘报告"""
        if not self.discovered_factors:
            return {"status": "no_factors"}

        return {
            "status": "success",
            "num_factors": len(self.discovered_factors),
            "factors": [
                {
                    "name": f.name,
                    "description": f.description,
                    "expression": f.to_expression(),
                    "complexity": f.complexity,
                    "depth": f.depth,
                    "ic_mean": f.metrics.get("ic_mean", 0),
                    "ir": f.metrics.get("ir", 0),
                    "sharpe": f.metrics.get("sharpe_ratio", 0),
                    "weight": self.factor_weights.get(f.name, 0),
                }
                for f in self.discovered_factors
            ],
        }


class LLMAdaptiveStrategy:
    """LLM 自适应策略 - 定期重新挖掘因子"""

    def __init__(
        self,
        miner: LLMFactorMiner,
        retrain_frequency: int = 20,  # 每多少个交易日重新训练
        **kwargs,
    ):
        self.miner = miner
        self.retrain_frequency = retrain_frequency
        self.strategy_kwargs = kwargs
        self.strategy: Optional[LLMFactorStrategy] = None
        self.trade_count = 0
        self.last_mine_date = None

    async def update(
        self,
        current_date,
        market_context: Dict[str, Any],
        market_data: Dict[str, pd.DataFrame],
        returns: Dict[str, List[float]],
    ) -> bool:
        """更新策略（按需重新挖掘因子）"""
        need_mine = (
            self.strategy is None
            or self.trade_count % self.retrain_frequency == 0
            or (self.last_mine_date is not None and current_date != self.last_mine_date)
        )

        if need_mine:
            logger.info(f"触发 LLM 因子重新挖掘 (交易次数: {self.trade_count})")

            self.strategy = LLMFactorStrategy(
                miner=self.miner,
                **self.strategy_kwargs,
            )

            success = await self.strategy.mine_factors(
                market_context, market_data, returns
            )

            if success:
                self.last_mine_date = current_date
                return True
            else:
                logger.warning("LLM 因子挖掘失败，使用上一次的策略")
                return False

        self.trade_count += 1
        return True

    def generate_signals(self, data: Dict[str, pd.DataFrame]) -> List[Signal]:
        """生成交易信号"""
        if self.strategy is None:
            logger.warning("策略尚未初始化")
            return []
        return self.strategy.generate_signals(data)
