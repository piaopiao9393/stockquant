"""Idea Agent 提示词模板 - 生成市场假设"""

# 系统提示词
IDEA_AGENT_SYSTEM_PROMPT = """You are a quantitative finance researcher specializing in discovering market inefficiencies and generating testable hypotheses.

Your expertise includes:
- Behavioral finance and market psychology
- Technical analysis patterns
- Fundamental analysis factors
- Market microstructure
- Factor investing

When generating hypotheses:
1. Base them on observable market phenomena or established financial theories
2. Make them specific and testable with quantitative data
3. Consider A-share market characteristics (T+1, limit up/down, retail-heavy)
4. Avoid overly complex or data-mined ideas
5. Focus on economically intuitive relationships

Always structure your response as valid JSON."""

# 生成市场假设的提示词
GENERATE_HYPOTHESIS_PROMPT = """Based on the following market context, generate {num_hypotheses} novel market hypotheses that could lead to alpha-generating factors.

Market Context:
- Current Market Regime: {market_regime}
- Volatility Environment: {volatility}
- Trend Direction: {trend}
- Recent Factor Performance: {recent_performance}

Existing Hypotheses (avoid similar ideas):
{existing_hypotheses}

User Direction (optional): {user_direction}

For each hypothesis, provide:
1. **Observation**: What market pattern or anomaly have you observed?
2. **Knowledge**: What financial theory or empirical evidence supports this?
3. **Reasoning**: How do you connect the observation to economic mechanisms?
4. **Specification**: What are the concrete implementation constraints?
5. **Time Horizon**: Is this a short-term or long-term effect?

Return your response as a JSON object with this structure:
{{
    "hypotheses": [
        {{
            "id": "hypo_1",
            "statement": "Clear, testable hypothesis statement",
            "observation": "What you observed in the market",
            "knowledge": "Supporting theory or evidence",
            "reasoning": "Logical connection between observation and mechanism",
            "specification": "Implementation details like time windows, thresholds",
            "time_horizon": "short_term | medium_term | long_term",
            "expected_factor_form": "Describe what the factor might look like mathematically",
            "confidence": 0.8,
            "category": "momentum | value | quality | volatility | liquidity"
        }}
    ],
    "reasoning": "Overall analysis of the current market environment"
}}"""

# 改进假设的提示词
IMPROVE_HYPOTHESIS_PROMPT = """Given the following hypothesis and evaluation feedback, suggest improvements or variations.

Original Hypothesis:
{hypothesis}

Evaluation Feedback:
{feedback}

Suggest {num_variations} improved versions that address the issues while maintaining the core insight.

Return as JSON:
{{
    "improved_hypotheses": [
        {{
            "id": "hypo_improved_1",
            "statement": "Improved hypothesis",
            "changes": "What was changed and why",
            "confidence": 0.75
        }}
    ]
}}"""
