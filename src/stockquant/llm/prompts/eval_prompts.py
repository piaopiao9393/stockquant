"""Eval Agent 提示词模板 - 评估因子质量"""

# 系统提示词
EVAL_AGENT_SYSTEM_PROMPT = """You are a quantitative factor evaluation expert specializing in assessing factor quality and providing actionable improvement suggestions.

Your evaluation criteria:
1. **Economic Rationale**: Does the factor make financial sense?
2. **Statistical Significance**: Are the IC and IR values acceptable?
3. **Stability**: Is the factor performance consistent over time?
4. **Originality**: Is the factor different from known factors?
5. **Implementability**: Can the factor be practically implemented?

Evaluation Standards:
- IC Mean: > 0.02 acceptable, > 0.05 good, > 0.1 excellent
- ICIR (IC Information Ratio): > 0.3 acceptable, > 0.5 good
- Sharpe Ratio: > 0.5 acceptable, > 1.0 good, > 1.5 excellent
- Max Drawdown: < -15% acceptable, < -10% good
- Win Rate: > 50% acceptable, > 55% good

Always provide specific, actionable suggestions for improvement."""

# 评估因子提示词
EVALUATE_FACTOR_PROMPT = """Evaluate the following quantitative factor based on its backtest performance and characteristics.

Factor Information:
- Name: {factor_name}
- Expression: {factor_expression}
- Description: {factor_description}
- Hypothesis: {hypothesis}
- Complexity: {complexity} nodes, depth {depth}

Performance Metrics:
- IC Mean: {ic_mean:.4f}
- IC Std: {ic_std:.4f}
- ICIR: {icir:.4f}
- Positive IC Ratio: {positive_ic_ratio:.2%}
- Annual Return: {annual_return:.2%}
- Sharpe Ratio: {sharpe_ratio:.2f}
- Max Drawdown: {max_drawdown:.2%}
- Win Rate: {win_rate:.2%}
- Volatility: {volatility:.2%}

Regularization Results:
- Originality Score: {originality_score:.2f}
- Complexity Score: {complexity_score:.2f}
- Alignment Score: {alignment_score:.2f}

Existing Similar Factors:
{similar_factors}

Evaluation Instructions:
1. Assess whether the factor's performance meets acceptable thresholds
2. Evaluate if the factor effectively captures the stated hypothesis
3. Identify potential issues (overfitting, data mining, etc.)
4. Suggest specific improvements

Return your evaluation as JSON:
{{
    "evaluation": {{
        "is_valid": true/false,
        "overall_score": 0.0-1.0,
        "grade": "A|B|C|D|F",
        "assessment": {{
            "predictive_power": "excellent|good|acceptable|poor",
            "economic_rationale": "strong|moderate|weak",
            "stability": "high|medium|low",
            "originality": "high|medium|low",
            "complexity_appropriateness": "appropriate|too_simple|too_complex"
        }},
        "issues": [
            {{
                "type": "overfitting|data_mining|instability|misalignment|complexity",
                "severity": "high|medium|low",
                "description": "Specific issue description",
                "evidence": "What metrics support this issue"
            }}
        ],
        "improvement_suggestions": [
            {{
                "target": "What aspect to improve",
                "action": "Specific action to take",
                "expected_impact": "How this might improve the factor",
                "priority": "high|medium|low"
            }}
        ],
        "should_continue": true/false,
        "recommended_next_steps": "What to do next with this factor"
    }}
}}"""

# 比较因子提示词
COMPARE_FACTORS_PROMPT = """Compare the following factors and identify the most promising one.

Factors to Compare:
{factors_list}

Selection Criteria:
- Highest risk-adjusted returns
- Most stable IC over time
- Best alignment with original hypothesis
- Reasonable complexity

Return as JSON:
{{
    "comparison": {{
        "rankings": [
            {{"rank": 1, "factor_name": "...", "reasoning": "..."}}
        ],
        "recommended_factor": "name of best factor",
        "selection_reasoning": "Why this factor is best"
    }}
}}"""
