from datetime import datetime, timedelta
from typing import Dict, List

from leek_core.models import ExecutionContext, PositionInfo
from .strategy import StrategyPolicy
from leek_core.utils import get_logger
from leek_core.models import Field, FieldType

logger = get_logger(__name__)


class StrategySignalLimit(StrategyPolicy):
    """
    信号限频风控（StrategySignalLimit）

    本风控策略用于限制策略信号在指定时间窗口（小时）内的最大触发次数，防止信号过于频繁导致风控失效或下单过多。

    主要参数说明（由 leek_core.models.Field 定义）：
    - per_hours (int): 时间窗口，单位为小时。
    - max_signals (int): 时间窗口内允许的最大信号数量。
    
    使用说明：
    - 继承自 StrategyPolicy，需实现 evaluate 方法。
    - 在 evaluate 中根据 per_hours 和 max_signals 判断信号是否通过风控。
    - 可扩展为不同时间窗口、不同信号类型的限频风控。
    """
    display_name = "信号限频风控"
    init_params = [
        Field(
            name='per_hours',
            label='时间窗口(小时)',
            type=FieldType.INT,
            default=2,
            required=True,
            description='时间窗口，单位为小时'
        ),
        Field(
            name='max_signals',
            label='最大信号数',
            type=FieldType.INT,
            default=10,
            required=True,
            description='时间窗口内允许的最大信号触发次数'
        )
    ]

    def __init__(self, per_hours: int, max_signals: int):
        """
        初始化 StrategySignalLimit 类。

        :param per_hours: 时间窗口，单位为小时。
        :param max_signals: 时间窗口内允许的最大信号数量。
        """
        super().__init__()
        self.per_hours = per_hours
        self.max_signals = max_signals
        # 存储每个策略实例的信号时间列表
        self.strategy_signal_times: Dict[str, List[datetime]] = {}

    def evaluate(self, signal: ExecutionContext, context: PositionInfo) -> bool:
        """
        检查信号是否符合风控规则，即特定策略实例在 per_hours 小时内的信号数量是否超过 max_signals。

        :param signal: 信号数据
        :param context: 持仓信息
        :return: 如果信号有效返回 True，否则返回 False。
        """
        strategy_instance_id = signal.strategy_instance_id
        signal_time = signal.created_time
        # 获取当前时间窗口的起始时间
        time_window_start = signal_time - timedelta(hours=self.per_hours)
        # 获取该策略实例的信号时间列表，若不存在则初始化为空列表
        signal_times = self.strategy_signal_times.get(strategy_instance_id, [])
        # 过滤掉时间窗口外的旧信号
        valid_signal_times = [t for t in signal_times if t >= time_window_start]

        if len(valid_signal_times) >= self.max_signals:
            logger.info(f"策略实例 {strategy_instance_id} 在 {self.per_hours} 小时内信号数量达到上限 {self.max_signals}，信号被拒绝。")
            return False

        # 记录新的信号时间
        valid_signal_times.append(signal_time)
        self.strategy_signal_times[strategy_instance_id] = valid_signal_times
        logger.info(f"策略实例 {strategy_instance_id} 接收新信号，当前 {self.per_hours} 小时内信号数量: {len(valid_signal_times)}")
        return True
