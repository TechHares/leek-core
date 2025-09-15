from decimal import Decimal
from typing import Dict, List, Set

from leek_core.models import PositionSide, Position, Data, DataType, Field, FieldType
from .position import SubStrategy
from leek_core.utils import get_logger

logger = get_logger(__name__)


class PositionTargetTrailingExit(SubStrategy):
    """
    目标追踪离场：止损、目标、预留百分比。

    多头：突破目标后，出场价 = 开仓价 + (最高价-开仓价)*reserve_ratio，且仅上调(ST)。
    空头：突破目标后，出场价 = 开仓价 - (开仓价-最低价)*reserve_ratio，且仅下调(ST)。
    回落(或反弹)到出场价则退出。
    """
    display_name: str = "目标追踪离场"
    init_params: List[Field] = [
        Field(
            name="stop_loss_ratio",
            label="止损比例",
            type=FieldType.FLOAT,
            default=2,
            min=0,
            max=100,
            required=True,
            description="相对开仓价的止损比例，例如 5 表示 5%"
        ),
        Field(
            name="target_ratio",
            label="目标比例",
            type=FieldType.FLOAT,
            default=5,
            min=0,
            max=1000,
            required=True,
            description="相对开仓价的目标比例，达到后启动追踪止盈"
        ),
        Field(
            name="reserve_ratio",
            label="预留百分比",
            type=FieldType.FLOAT,
            default=60,
            min=0,
            max=100,
            required=True,
            description="在止损价与突破后极值之间保留的比例(0~100)"
        ),
    ]
    accepted_data_types: Set[DataType] = {DataType.KLINE}

    def __init__(self, stop_loss_ratio: Decimal = Decimal('2'), target_ratio: Decimal = Decimal('5'),
     reserve_ratio: Decimal = Decimal('60')):
        """
        参数以百分比输入：
        - stop_loss_ratio: 止损百分比(0~100)，如 2 表示 2%
        - target_ratio: 目标百分比(0~100)，如 5 表示 5%
        - reserve_ratio: 预留百分比(0~100)，如 60 表示 60%
        内部统一转换为 [0,1] 的小数。
        """
        self.stop_loss_ratio = Decimal(str(stop_loss_ratio)) / Decimal('100')
        self.target_ratio = Decimal(str(target_ratio)) / Decimal('100')
        self.reserve_ratio = Decimal(str(reserve_ratio)) / Decimal('100')
        self._state_by_position: Dict[str, Dict[str, Decimal | bool]] = {}

    def _get_state(self, position: Position) -> Dict[str, Decimal | bool]:
        """按 position_id 维护追踪所需的本地状态。"""
        pid = position.position_id
        state = self._state_by_position.get(pid)
        if state is None:
            state = {
                'target_broken': False,
                'highest_since_break': None,
                'lowest_since_break': None,
                'trailing_stop': None,
            }
            self._state_by_position[pid] = state
        return state

    def _compute_stop_and_target(self, position: Position) -> tuple[Decimal, Decimal]:
        """
        基于开仓价计算止损价与目标价：
        - 多头：SL = cost*(1-stop), TP = cost*(1+target)
        - 空头：SL = cost*(1+stop), TP = cost*(1-target)
        """
        cost = position.cost_price
        if position.side == PositionSide.LONG:
            stop_price = cost * (Decimal('1') - self.stop_loss_ratio)
            target_price = cost * (Decimal('1') + self.target_ratio)
        else:
            stop_price = cost * (Decimal('1') + self.stop_loss_ratio)
            target_price = cost * (Decimal('1') - self.target_ratio)
        return stop_price, target_price

    def evaluate(self, data: Data, position: Position) -> bool:
        """
        返回 True 继续持有，返回 False 触发离场。
        逻辑：
        1) 目标未破：先检查是否触发止损；若最高(或最低)价突破目标，进入追踪阶段。
        2) 目标已破：基于开仓价(cost_price)与突破后的极值计算追踪价，ST方式仅单向调整；
           多头回落至追踪价，空头反弹至追踪价则离场。
        """
        assert data.data_type == DataType.KLINE, "数据类型错误"
        state = self._get_state(position)
        stop_price, target_price = self._compute_stop_and_target(position)
        cost = position.cost_price

        if position.side == PositionSide.LONG:
            if not state['target_broken']:
                if data.close <= stop_price:
                    logger.info(f"目标追踪离场: {position.position_id} 触发离场, 当前价格: {data.close}, 止损价: {stop_price}")
                    return False
                if data.high is not None and data.high >= target_price:
                    state['target_broken'] = True
                    state['highest_since_break'] = data.high
                    # 使用开仓价为锚点初始化追踪出场价，ST方式仅上调
                    state['trailing_stop'] = cost + (state['highest_since_break'] - cost) * self.reserve_ratio
                return True

            if data.high is not None:
                state['highest_since_break'] = max(state['highest_since_break'], data.high)
                # 仅上调：基于开仓价与破目标后的最高价
                new_trailing = cost + (state['highest_since_break'] - cost) * self.reserve_ratio
                if state['trailing_stop'] is None or new_trailing > state['trailing_stop']:
                    state['trailing_stop'] = new_trailing
            if data.close <= state['trailing_stop']:
                logger.info(f"目标追踪离场: {position.position_id} 触发离场, 当前价格: {data.close}, 追踪出场价: {state['trailing_stop']}")
                return False
            return True

        elif position.side == PositionSide.SHORT:
            if not state['target_broken']:
                if data.close >= stop_price:
                    logger.info(f"目标追踪离场: {position.position_id} 触发离场, 当前价格: {data.close}, 止损价: {stop_price}")
                    return False
                if data.low is not None and data.low <= target_price:
                    state['target_broken'] = True
                    state['lowest_since_break'] = data.low
                    # 使用开仓价为锚点初始化追踪出场价，ST方式仅下调
                    state['trailing_stop'] = cost - (cost - state['lowest_since_break']) * self.reserve_ratio
                return True

            if data.low is not None:
                state['lowest_since_break'] = min(state['lowest_since_break'], data.low)
                # 仅下调：基于开仓价与破目标后的最低价
                new_trailing = cost - (cost - state['lowest_since_break']) * self.reserve_ratio
                if state['trailing_stop'] is None or new_trailing < state['trailing_stop']:
                    state['trailing_stop'] = new_trailing
            if data.close >= state['trailing_stop']:
                logger.info(f"目标追踪离场: {position.position_id} 触发离场, 当前价格: {data.close}, 追踪出场价: {state['trailing_stop']}")
                return False
            return True

        return False



