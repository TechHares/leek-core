from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from leek_core.position import PositionTracker

@dataclass
class Context:
    position_tracker: "PositionTracker" = None


leek_context: Context = Context()


def initialize_context(position_tracker: "PositionTracker"):
    """
    初始化全局上下文
    
    参数:
        position_tracker: 仓位跟踪器实例
    """
    global leek_context
    leek_context.position_tracker = position_tracker