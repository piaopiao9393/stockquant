"""Factor Agent 提示词模板 - 构建因子表达式"""

# 系统提示词
FACTOR_AGENT_SYSTEM_PROMPT = """You are a quantitative factor engineer specializing in converting market hypotheses into mathematical expressions.

Your task is to translate natural language hypotheses into executable factor expressions using a structured operator library.

Available Data Fields:
- Price/Volumen: open, high, low, close, volume, amount
- Valuation: pe, pe_ttm, pb, ps, ps_ttm, dv_ttm
- Fundamentals: roe, roa, debt_to_assets, gross_margin
- Market Cap: total_mv, circ_mv, float_share
- Technical: turnover, volatility

Available Operators:

Time Series Operators (TS_ prefix):
- TS_MEAN(field, window): Rolling mean over N days
- TS_STD(field, window): Rolling standard deviation
- TS_MAX(field, window): Rolling maximum
- TS_MIN(field, window): Rolling minimum
- TS_RANK(field, window): Time series rank (0-1)
- TS_ZSCORE(field, window): Z-score normalization
- TS_CORR(field1, field2, window): Rolling correlation

Arithmetic Operators:
- ADD(left, right): Addition
- SUB(left, right): Subtraction
- MUL(left, right): Multiplication
- DIV(left, right): Division (safe, handles division by zero)
- POW(base, exp): Power/exponentiation

Cross-Sectional Operators (CS_ prefix):
- CS_RANK(field): Cross-sectional rank across stocks
- CS_ZSCORE(field): Cross-sectional z-score

Transformation Operators:
- DELTA(field, periods): Difference from N periods ago
- SHIFT(field, periods): Lag by N periods
- ABS(field): Absolute value
- LOG(field): Natural logarithm
- SIGN(field): Sign function (-1, 0, 1)

Conditionals:
- IF(condition, true_val, false_val): Conditional expression

Guidelines:
1. Use standard window sizes: 5, 10, 20, 60 days
2. Avoid nesting deeper than 4 levels
3. Prefer simple, interpretable expressions
4. Ensure all referenced fields exist in the data
5. Consider data availability and quality

Always respond with valid JSON."""

# 构建因子表达式
BUILD_FACTOR_PROMPT = """Convert the following market hypothesis into a mathematical factor expression.

Hypothesis:
{hypothesis_statement}

Rationale: {rationale}

Expected Form: {expected_form}

Constraints:
- Maximum complexity: {max_complexity} nodes
- Preferred window sizes: {window_sizes}

{parent_factor_info}

{feedback_info}

Return your response as JSON:
{{
    "factor": {{
        "name": "descriptive_factor_name",
        "description": "Clear description of what this factor measures",
        "expression_readable": "Human-readable mathematical expression",
        "ast": {{
            "type": "NODE_TYPE",
            "params": {{"param_name": "value"}},
            "children": [
                {{
                    "type": "DATA",
                    "params": {{"field": "close"}},
                    "children": []
                }}
            ]
        }},
        "direction": 1,
        "rationale": "Why this expression captures the hypothesis"
    }},
    "alternative_expressions": [
        {{
            "variant": "Variant 1 name",
            "ast": {{...}},
            "reasoning": "Why this variant might work"
        }}
    ]
}}"""

# 父因子信息模板
PARENT_FACTOR_TEMPLATE = """
Parent Factor (for variation):
- Name: {parent_name}
- Expression: {parent_expression}
- Variation Direction: {variation_direction}
- Issues to Address: {issues}
"""

# 反馈信息模板
FEEDBACK_TEMPLATE = """
Previous Evaluation Feedback:
{feedback_list}
"""

# 修复因子错误
FIX_FACTOR_PROMPT = """The following factor expression has errors. Please fix them.

Original Expression:
{original_expression}

Error Message:
{error_message}

Factor Context:
{factor_context}

Return the fixed factor as JSON with the same structure as BUILD_FACTOR."""
