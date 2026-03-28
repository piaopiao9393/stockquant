# StockQuant - A股量化交易系统

基于多因子策略的低频量化交易系统，集成**LLM智能因子挖掘**功能，适合5万-50万资金规模。

> **默认使用免费数据源**：无需Tushare积分，开箱即用！

---

## 功能特点

### 核心功能
- **多因子策略框架**：价值/质量/动量/波动/规模/分红/技术 7大类因子
- **免费数据源**：默认使用AKShare（无需Token），可选Tushare Pro
- **LLM智能因子挖掘**：基于AlphaAgent (KDD 2025) 的三智能体自动因子发现
- **Web可视化界面**：Streamlit构建，支持策略配置、选股结果展示、交易计划管理
- **邮件推送**：交易建议自动发送到邮箱
- **数据自动更新**：每日收盘后自动拉取数据

### LLM因子挖掘特色
- **三智能体架构**：IdeaAgent → FactorAgent → EvalAgent
- **AST因子表示**：抽象语法树表达，支持原创性检测
- **自动IC/回测评估**：挖掘即评估，筛选高质量因子
- **因子仓库**：SQLite持久化，支持导入导出
- **策略集成**：挖掘的因子可直接用于多因子策略

---

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/piaopiao9393/stockquant.git
cd stockquant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制模板文件
cp .env.example .env
```

**.env 已默认配置AKShare（免费），可直接使用：**

```bash
# 默认数据源: akshare (免费) / tushare (需积分)
DATA_DEFAULT_SOURCE=akshare

# Tushare配置（如使用tushare则必填）
TUSHARE_TOKEN=your_token_here

# LLM API配置 (用于因子挖掘，可选)
DEEPSEEK_API_KEY=your_deepseek_key_here
OPENAI_API_KEY=your_openai_key_here

# 邮件推送配置 (可选)
SMTP_HOST=smtp.qq.com
SMTP_USER=your_email@qq.com
SMTP_PASSWORD=your_password
EMAIL_TO=receive@example.com
```

### 3. 测试数据连接

```bash
# 测试AKShare数据连接（免费）
python test_akshare_simple.py

# 测试默认数据源
python test_default_source.py
```

### 4. 运行示例

```bash
# 运行回测示例
python examples/backtest_example.py

# 运行LLM因子挖掘（需配置LLM API）
python examples/llm_factor_mining.py
```

### 5. 启动Web界面

```bash
streamlit run src/stockquant/web/app.py
# 或
stockquant web
```

浏览器访问：http://localhost:8501

---

## 项目结构

```
stockquant/
├── src/stockquant/              # 主代码包
│   ├── main.py                  # 系统主入口
│   ├── cli.py                   # 命令行工具
│   ├── config/                  # 配置管理（Pydantic Settings）
│   │   ├── __init__.py
│   │   └── settings.py          # 统一配置（支持.env）
│   ├── data/                    # 数据层
│   │   ├── models.py            # SQLAlchemy数据模型
│   │   ├── database.py          # 数据库连接
│   │   ├── fetcher.py           # Tushare接口
│   │   ├── akshare_fetcher.py   # AKShare接口（免费，默认）
│   │   ├── unified_fetcher.py   # 统一数据接口
│   │   ├── dataservice.py       # 数据服务层
│   │   ├── cache.py             # 数据缓存
│   │   └── updater.py           # 数据更新
│   ├── factors/                 # 因子库（7大类20+因子）
│   │   ├── registry.py          # 因子注册中心
│   │   ├── base.py              # 因子基类
│   │   └── definitions.py       # 预定义因子
│   ├── strategy/                # 策略框架
│   │   ├── base.py              # 策略基类
│   │   ├── core/                # 核心策略实现
│   │   │   └── strategy.py      # 多因子策略
│   │   └── prebuilt/            # 预置策略
│   │       ├── value_quality.py      # 价值+质量
│   │       ├── small_cap_value.py    # 小盘+价值
│   │       ├── lowvol_dividend.py    # 低波+红利
│   │       └── momentum.py           # 动量策略
│   ├── backtest/                # 回测引擎
│   │   ├── engine.py            # 事件驱动回测
│   │   ├── performance.py       # 绩效分析
│   │   ├── result.py            # 回测结果
│   │   ├── slippage.py          # 滑点模型
│   │   └── risk_manager.py      # 风险管理
│   ├── portfolio/               # 组合管理
│   │   ├── account.py           # 账户管理
│   │   └── plan_generator.py    # 交易计划生成
│   ├── llm/                     # LLM因子挖掘模块
│   │   ├── core/                # AST核心
│   │   │   ├── ast_node.py      # AST节点类型
│   │   │   ├── factor_ast.py    # FactorAST类
│   │   │   ├── similarity.py    # 相似度计算
│   │   │   └── regularization.py # 正则化约束
│   │   ├── agents/              # 三智能体系统
│   │   │   ├── base.py          # 智能体基类
│   │   │   ├── idea_agent.py    # 假设生成
│   │   │   ├── factor_agent.py  # 因子构建
│   │   │   └── eval_agent.py    # 因子评估
│   │   ├── miner/               # 因子挖掘
│   │   │   └── base.py          # LLMFactorMiner
│   │   ├── storage/             # 因子存储
│   │   │   └── repository.py    # SQLite仓库
│   │   ├── strategy/            # LLM策略集成
│   │   │   └── llm_strategy.py
│   │   ├── prompts/             # LLM提示词
│   │   │   ├── idea_prompts.py
│   │   │   ├── factor_prompts.py
│   │   │   └── eval_prompts.py
│   │   └── utils/               # 工具函数
│   │       └── llm_client.py    # LLM客户端
│   ├── web/                     # Streamlit Web界面
│   │   ├── app.py               # 主应用
│   │   ├── pages/               # 页面
│   │   └── components/          # 组件
│   └── notification/            # 邮件通知
│       └── mailer.py
├── tests/                       # 测试用例
├── examples/                    # 使用示例
│   ├── backtest_example.py
│   └── llm_factor_mining.py
├── data_storage/                # 数据存储（SQLite）
├── plans/                       # 交易计划输出
├── .env.example                 # 环境变量模板
├── requirements.txt             # 依赖
└── pyproject.toml               # 项目配置
```

---

## 核心模块使用指南

### 1. 数据层（默认免费AKShare）

```python
from stockquant.data import create_data_fetcher

# 方式1：自动选择（默认AKShare，免费）
fetcher = create_data_fetcher()

# 方式2：明确指定AKShare（免费，无需Token）
fetcher = create_data_fetcher(prefer_source='akshare')

# 方式3：使用Tushare（需积分）
fetcher = create_data_fetcher(prefer_source='tushare', token='your_token')

# 获取日线数据
df = fetcher.get_daily_prices('000001.SZ', start_date='20241201', end_date='20241231')
print(df[['trade_date', 'open', 'close', 'high', 'low', 'vol']])

# 获取股票列表
stocks = fetcher.get_stock_list()

# 获取交易日历
calendar = fetcher.get_trade_calendar('20241201', '20241231')
```

### 2. 多因子策略

```python
from stockquant.strategy import MultiFactorStrategy
from stockquant.factors import FactorRegistry

# 创建策略（价值+质量+低波动）
strategy = MultiFactorStrategy(
    factors=['EP_TTM', 'BP', 'ROE_TTM', 'Gross_Margin', 'Realized_Vol'],
    weights=[0.3, 0.2, 0.2, 0.15, 0.15]
)

# 生成交易信号
signals = strategy.generate_signals(
    date='20241201',
    symbols=['000001.SZ', '000002.SZ', '600000.SH']
)

# 查看信号
for signal in signals[:5]:
    print(f"{signal.symbol}: 评分={signal.score:.3f}, 置信度={signal.confidence:.2f}")
```

**内置因子分类：**

| 类别 | 代表因子 | 说明 |
|------|---------|------|
| **价值** | EP_TTM, BP, SP_TTM, Dividend_Yield, CFP | 估值比率 |
| **质量** | ROE_TTM, ROA_TTM, Gross_Margin, Debt_Equity | 财务质量 |
| **动量** | MOM_1M, MOM_3M, MOM_6M, MOM_12M | 价格趋势 |
| **波动率** | Realized_Vol, Max_Drawdown_20D, RSI_14 | 风险指标 |
| **市值** | Log_MarketCap, Turnover_20D | 市值&流动性 |
| **分红** | Dividend_Yield, Payout_Ratio | 分红能力 |
| **技术** | MACD, KDJ, BOLL_Position | 技术指标 |

### 3. 回测引擎

```python
from stockquant.backtest import BacktestEngine
from stockquant.strategy.prebuilt import ValueQualityStrategy

# 创建策略
strategy = ValueQualityStrategy(min_market_cap=50e8)

# 配置回测
engine = BacktestEngine(
    initial_cash=50000,
    commission_rate=0.0003,      # 佣金万3
    max_position_pct=0.25,       # 单票最大25%
    max_total_position=0.80      # 总仓位最大80%
)

# 运行回测
result = engine.run(
    strategy=strategy,
    symbols=['000001.SZ', '000002.SZ', ...],  # 股票池
    start_date='20230101',
    end_date='20241231',
    rebalance_freq='weekly'      # 调仓频率：daily/weekly/monthly
)

# 查看结果
print(result.summary)            # 年化收益、夏普、最大回撤等
result.plot()                    # Plotly可视化图表
```

**回测绩效指标：**
- 年化收益率、年化波动率
- 夏普比率、索提诺比率、卡玛比率
- 最大回撤、胜率、盈亏比
- 累计收益曲线、回撤曲线

### 4. LLM智能因子挖掘

```python
from stockquant.llm import LLMFactorMiner, create_llm_strategy
from stockquant.llm.utils.llm_client import create_llm_client

# 方式1：快速使用（推荐）
miner = LLMFactorMiner.with_default_agents(
    llm_config={"provider": "deepseek", "api_key": "sk-xxx"}
)

# 挖掘价值因子
result = miner.mine_factor(
    category="value",
    hypothesis="高ROE且低估值的股票长期表现更好",
    max_attempts=3
)

if result.success:
    print(f"挖掘成功: {result.factor.name}")
    print(f"表达式: {result.factor.to_expression()}")
    print(f"IC均值: {result.metrics.get('ic_mean', 0):.4f}")

# 方式2：使用挖掘的因子构建策略
strategy = create_llm_strategy(
    factor_names=["llm_factor_001", "llm_factor_002"],
    weights=[0.6, 0.4]
)
```

**LLM因子挖掘流程：**

1. **IdeaAgent** - 分析市场，生成投资假设
2. **FactorAgent** - 将假设转化为因子表达式（AST）
3. **EvalAgent** - 评估因子质量（IC、回测、经济学逻辑）
4. **Regularization** - 正则化约束（原创性、复杂度、假设一致性）
5. **FactorRepository** - 持久化到SQLite

---

## 预置策略

### 价值质量策略（ValueQualityStrategy）

```python
from stockquant.strategy.prebuilt import ValueQualityStrategy

strategy = ValueQualityStrategy(
    min_market_cap=50e8,         # 最小市值50亿
    min_roe=0.10,                # 最小ROE 10%
    max_pe=30,                   # 最大PE 30
    max_pb=3,                    # 最大PB 3
    position_pct=0.20            # 单票仓位20%
)
```

### 小盘价值策略（SmallCapValueStrategy）

```python
from stockquant.strategy.prebuilt import SmallCapValueStrategy

strategy = SmallCapValueStrategy(
    max_market_cap=100e8,        # 最大市值100亿（小盘）
    min_dividend_yield=0.02,     # 最小股息率2%
    position_pct=0.15            # 单票仓位15%（小盘风控）
)
```

### 低波红利策略（LowVolDividendStrategy）

```python
from stockquant.strategy.prebuilt import LowVolDividendStrategy

strategy = LowVolDividendStrategy(
    min_dividend_yield=0.03,     # 最小股息率3%
    max_volatility=0.25,         # 最大波动率25%
    position_pct=0.20
)
```

---

## 命令行工具

```bash
# 查看帮助
stockquant --help

# 初始化数据库
stockquant init

# 更新市场数据
stockquant update

# 运行策略生成交易建议
stockquant run

# 运行回测
stockquant backtest --strategy value_quality --start 2023-01-01 --end 2023-12-31

# 启动Web界面
stockquant web
```

---

## 测试

```bash
# 运行所有测试
python -m pytest tests/

# 测试AKShare数据连接
python test_akshare_simple.py

# 测试默认数据源
python test_default_source.py

# 运行LLM模块测试
python -m tests.llm.verify_core      # 核心模块测试
python -m tests.llm.test_agents      # 智能体测试
python -m tests.llm.verify_startup   # 启动验证
```

---

## 风险控制

系统内置四层风险控制：

1. **单票仓位限制** - 默认不超过25%
2. **总仓位限制** - 默认不超过80%
3. **日亏损限制** - 单日亏损超5%提醒
4. **最大回撤限制** - 回撤超15%提醒

```python
# 在回测引擎中配置风控
engine = BacktestEngine(
    max_position_pct=0.25,       # 单票最大25%
    max_total_position=0.80,     # 总仓位最大80%
    max_daily_loss_pct=0.05,     # 单日最大亏损5%
    max_drawdown_pct=0.15        # 最大回撤15%
)
```

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 语言 | Python 3.10+ |
| 数据处理 | pandas, numpy |
| 数据接口 | AKShare（默认）, Tushare Pro |
| 数据库 | SQLite (SQLAlchemy ORM) |
| Web界面 | Streamlit |
| 可视化 | Plotly |
| 配置管理 | Pydantic Settings |
| 任务调度 | APScheduler |
| 邮件通知 | smtplib |
| LLM集成 | OpenAI, DeepSeek APIs |
| 测试 | pytest |

---

## 数据源对比

| 特性 | AKShare（默认） | Tushare Pro |
|------|----------------|-------------|
| **费用** | 免费 | 需积分 |
| **注册** | 无需 | 需注册 |
| **稳定性** | 良好 | 优秀 |
| **数据延迟** | 实时 | 实时 |
| **历史数据** | 完整 | 完整 |
| **适合场景** | 个人研究、回测 | 生产环境 |

**切换数据源：**
```bash
# 修改 .env 文件
DATA_DEFAULT_SOURCE=akshare   # 免费
DATA_DEFAULT_SOURCE=tushare   # 需积分
```

---

## 常见问题

### Q1: AKShare连接失败怎么办？

AKShare默认禁用代理以避免连接问题。如果仍失败：
1. 检查网络连接
2. 尝试更换网络环境
3. 或切换到Tushare（需积分）

### Q2: 如何配置LLM API？

在 `.env` 文件中添加：
```bash
DEEPSEEK_API_KEY=sk-your-key-here
# 或
OPENAI_API_KEY=sk-your-key-here
```

### Q3: 如何保存挖掘的因子？

```python
from stockquant.llm.storage import FactorRepository

repo = FactorRepository("data_storage/llm_factors.db")
repo.save_factor(factor)
```

---

## 免责声明

本系统仅供学习研究使用，不构成投资建议。股市有风险，投资需谨慎。

---

## 参考

- **AlphaAgent**: [KDD 2025] Discovering Alpha Factors with Multi-Agent LLM
- **Qlib**: Microsoft's quantitative investment platform
- **AKShare**: 开源财经数据接口
- **Tushare Pro**: 中文财经数据接口平台
