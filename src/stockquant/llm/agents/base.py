"""智能体基类"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class AgentResponse:
    """智能体响应"""

    success: bool
    data: Dict[str, Any]
    reasoning: str = ""
    tokens_used: int = 0


class BaseAgent(ABC):
    """智能体基类"""

    def __init__(
        self, name: str, llm_client, system_prompt: str = "", temperature: float = 0.7
    ):
        self.name = name
        self.llm_client = llm_client
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.conversation_history: List[Dict] = []

    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> AgentResponse:
        """执行智能体任务"""
        pass

    def _build_messages(
        self, user_prompt: str, include_history: bool = True
    ) -> List[Dict]:
        """构建消息列表"""
        messages = []

        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if include_history:
            messages.extend(self.conversation_history)

        messages.append({"role": "user", "content": user_prompt})

        return messages

    def _update_history(self, user_msg: str, assistant_msg: str):
        """更新对话历史"""
        self.conversation_history.append({"role": "user", "content": user_msg})
        self.conversation_history.append(
            {"role": "assistant", "content": assistant_msg}
        )

        # 限制历史长度
        max_history = 10
        if len(self.conversation_history) > max_history * 2:
            self.conversation_history = self.conversation_history[-max_history * 2 :]

    def clear_history(self):
        """清除对话历史"""
        self.conversation_history = []

    def log_action(self, action: str, details: Dict):
        """记录智能体行为"""
        logger.info(f"[{self.name}] {action}: {details}")
