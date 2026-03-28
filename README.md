# StockQuant - A股量化交易系统

基于多因子策略的低频量化交易系统，集成**LLM智能因子挖掘**功能，适合5万-50万资金规模。

## 功能特点

### 核心功能
- **多因子策略框架**：价值/质量/动量/波动/规模/分红/技术 7大类因子
- **LLM智能因子挖掘**：基于AlphaAgent (KDD 2025) 的三智能体自动因子发现
- **Web可视化界面**：Streamlit构建，支持策略配置、选股结果展示、交易计划管理
- **邮件推送**：交易建议自动发送到邮箱
- **数据自动更新**：每日收盘后自动拉取Tushare数据

### LLM因子挖掘特色
- 🤖 **三智能体架构**：IdeaAgent → FactorAgent → EvalAgent
- 🌲 **AST因子表示**：抽象语法树表达，支持原创性检测
- 📊 **自动IC/回测评估**：挖掘即评估，筛选高质量因子
- 💾 **因子仓库**：SQLite持久化，支持导入导出
- 🔗 **策略集成**：挖掘的因子可直接用于多因子策略

## 快速开始

### 1. 安装依赖

```bash
# 克隆项目
git clone https://github.com/piaopiao9393/stockquant.git
cd stockquant

# 安装依赖
pip install -e .
```

### 2. 配置环境变量

```bash
# 复制模板文件
cp .env.example .env

# 编辑 .env 文件，填入以下配置
```

**.env 配置示例：**
```bash
# Tushare数据接口 (必需)
TUSHARE_TOKEN=your_tushare_token_here

# LLM API配置 (用于因子挖掘，可选)
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here

# 邮件推送配置 (可选)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASSWORD=your_app_password
NOTIFICATION_EMAIL=receive_email@example.com
```

### 3. 初始化数据库

```bash
stockquant init
```

### 4. 启动Web界面

```bash
streamlit run src/stockquant/web/app.py
```

浏览器访问：http://localhost:8501

---

## LLM因子挖掘使用指南

### 基础用法

```python
import asyncio
from stockquant.llm.agents import IdeaAgent, FactorAgent, EvalAgent
from stockquant.llm.miner import LLMFactorMiner, MiningConfig
from stockquant.llm.utils.llm_client import create_llm_client

async def mine_factors():
    # 1. 创建LLM客户端
    client = create_llm_client("openai", model="gpt-4")

    # 2. 创建三智能体
    idea_agent = IdeaAgent(llm_client=client, temperature=0.8)
    factor_agent = FactorAgent(llm_client=client, temperature=0.6)
    eval_agent = EvalAgent(llm_client=client, temperature=0.3)

    # 3. 配置挖掘器
    config = MiningConfig(
        max_iterations=5,
        max_factors_per_hypothesis=3,
        originality_threshold=0.85,
        min_ic_threshold=0.02,
    )
    miner = LLMFactorMiner(idea_agent, factor_agent, eval_agent, config)

    # 4. 准备市场上下文
    market_context = {
        "market_regime": "high_volatility",
        "volatility": "high",
        "trend": "sideways",
        "market_data": {...},  # 市场数据
        "returns": {...},       # 收益率数据
    }

    # 5. 执行挖掘
    result = await miner.mine(market_context, num_factors=5)

    # 6. 查看结果
    for factor in result.factors:
        print(f"因子: {factor.name}")
        print(f"  表达式: {factor.to_expression()}")
        print(f"  IC均值: {factor.metrics.get('ic_mean', 0):.4f}")
        print(f"  Sharpe: {factor.metrics.get('sharpe_ratio', 0):.2f}")

# 运行
asyncio.run(mine_factors())
```

### 因子持久化

```python
from stockquant.llm.storage import FactorRepository
from stockquant.llm.core import FactorAST

# 创建仓库
repo = FactorRepository("data/llm_factors.db")

# 保存挖掘的因子
for factor in result.factors:
    repo.save_factor(factor)

# 列出所有因子
factors = repo.list_factors(min_ic=0.03)

# 导出到JSON
repo.export_factors("factors_backup.json")

# 从JSON导入
repo.import_factors("factors_backup.json")
```

### 在策略中使用LLM因子

```python
from stockquant.llm.strategy import LLMFactorStrategy

# 创建LLM策略
strategy = LLMFactorStrategy(
    miner=miner,
    max_llm_factors=3,
    filters={"min_market_cap": 5e8},
    max_holdings=10,
)

# 挖掘因子并生成信号
async def run_strategy():
    # 挖掘因子
    await strategy.mine_factors(market_context, market_data, returns)

    # 生成交易信号
    signals = strategy.generate_signals(data)

    # 查看因子报告
    report = strategy.get_factor_report()
    print(report)
```

---

## 项目结构

```
stockquant/
├── src/stockquant/
│   ├── config/              # 配置管理
│   ├── data/                # 数据模块 (Tushare集成)
│   ├── factors/             # 因子注册中心
│   ├── strategy/            # 策略框架 (含7大类基础因子)
│   ├── portfolio/           # 投资组合管理
│   ├── backtest/            # 回测引擎
│   ├── notification/        # 邮件通知系统
│   ├── web/                 # Streamlit Web界面
│   ├── cli.py               # 命令行工具
│   └── llm/                 # 🆕 LLM因子挖掘模块
│       ├── core/            # AST核心 (节点/相似度/正则化)
│       ├── agents/          # 三智能体 (Idea/Factor/Eval)
│       ├── miner/           # 因子挖掘器
│       ├── strategy/        # LLM策略集成
│       ├── storage/         # 因子仓库 (SQLite)
│       ├── prompts/         # LLM提示词模板
│       └── utils/           # LLM客户端 (OpenAI/Claude/DeepSeek)
├── tests/                   # 测试用例
│   └── llm/                 # LLM模块测试
├── examples/                # 使用示例
├── data_storage/            # 数据存储
└── plans/                   # 交易计划
```

---

## 命令行工具

```bash
# 初始化系统
stockquant init

# 更新数据
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

# 运行LLM模块测试
python -m tests.llm.verify_core      # 核心模块测试 (9个)
python -m tests.llm.test_agents      # 智能体测试 (5个)
python -m tests.llm.verify_startup   # 启动验证 (23个)
```

---

## 风险控制

- 单票仓位不超过25%
- 总仓位不超过80%
- 单只亏损8%强制止损提醒
- 小市值股票（<50亿）仓位限制<15%

---

## 技术栈

| 组件 | 技术 |
|------|------|
| 数据处理 | pandas, numpy |
| 数据接口 | Tushare Pro |
| Web界面 | Streamlit |
| LLM集成 | OpenAI, Anthropic, DeepSeek APIs |
| 数据存储 | SQLite |
| 通知 | smtplib |
| 测试 | pytest |

---

## 免责声明

本系统仅供学习研究使用，不构成投资建议。股市有风险，投资需谨慎。

---

## 参考

- AlphaAgent: [KDD 2025] Discovering Alpha Factors with Multi-Agent LLM
- Qlib: Microsoft's quantitative investment platform
- Tushare Pro: 中文财经数据接口平台
