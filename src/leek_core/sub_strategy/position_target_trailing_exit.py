from decimal import Decimal
from typing import Dict, List, Set, Optional

from leek_core.models import PositionSide, Position, Data, DataType, Field, FieldType
from .position import SubStrategy
from leek_core.utils import get_logger

logger = get_logger(__name__)


class PositionTargetTrailingExit(SubStrategy):
    """
    目标追踪离场：止损、目标、预留百分比。

    多头：收盘价突破目标后，出场价 = 开仓价 + (极值收盘价-开仓价)*reserve_ratio，且仅上调(ST)。
    空头：收盘价突破目标后，出场价 = 开仓价 - (开仓价-极值收盘价)*reserve_ratio，且仅下调(ST)。
    回落(或反弹)到出场价则退出。
    
    注意：使用收盘价判断目标突破，避免影线假突破导致在同一根K线内被错误平仓。

    利润保留比例（可选）：
    默认启用动态利润保留，根据盈利幅度动态调整保留比例，避免大盈利时回吐过多利润：
    - 盈利 < small_profit_threshold: 使用原始 reserve_ratio
    - 盈利 small_profit_threshold-large_profit_threshold: 使用二次函数插值调整到 profit_retention_ratio（或 reserve_ratio），先慢后快
    - 盈利 > large_profit_threshold: 使用 profit_retention_ratio（或 reserve_ratio）
    
    profit_retention_ratio 表示保留利润的比例，例如 80 表示保留80%的利润。
    如果盈利10%，profit_retention_ratio=80%，则触发价格为保留8%的利润。
    如果 profit_retention_ratio 为空，则使用 reserve_ratio 的值。
    
    插值方式：使用二次函数（weight²）实现先慢后快的变化，小盈利时变化缓慢，大盈利时快速调整。
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
        Field(
            name="profit_retention_ratio",
            label="利润保留比例",
            type=FieldType.FLOAT,
            default=None,
            min=0,
            max=100,
            required=False,
            description="大盈利时保留利润的比例(0~100)，例如80表示保留80%的利润。为空时使用reserve_ratio的值"
        ),
        Field(
            name="small_profit_threshold",
            label="小盈利阈值",
            type=FieldType.FLOAT,
            default=5,
            min=0,
            max=100,
            required=False,
            description="小盈利阈值百分比(0~100)，盈利低于此值时使用reserve_ratio，默认5%"
        ),
        Field(
            name="large_profit_threshold",
            label="大盈利阈值",
            type=FieldType.FLOAT,
            default=15,
            min=0,
            max=100,
            required=False,
            description="大盈利阈值百分比(0~100)，盈利高于此值时使用profit_retention_ratio，默认15%"
        ),
    ]
    accepted_data_types: Set[DataType] = {DataType.KLINE}

    def __init__(self, stop_loss_ratio: Decimal = Decimal('2'), target_ratio: Decimal = Decimal('5'),
     reserve_ratio: Decimal = Decimal('60'), profit_retention_ratio: Optional[Decimal] = None,
     small_profit_threshold: Decimal = Decimal('5'), large_profit_threshold: Decimal = Decimal('15')):
        """
        参数以百分比输入：
        - stop_loss_ratio: 止损百分比(0~100)，如 2 表示 2%
        - target_ratio: 目标百分比(0~100)，如 5 表示 5%
        - reserve_ratio: 预留百分比(0~100)，如 60 表示 60%
        - profit_retention_ratio: 利润保留百分比(0~100)，如 80 表示保留80%的利润。
                                 为空时使用 reserve_ratio 的值
        - small_profit_threshold: 小盈利阈值百分比(0~100)，默认5%
        - large_profit_threshold: 大盈利阈值百分比(0~100)，默认15%
        内部统一转换为 [0,1] 的小数。
        """
        self.stop_loss_ratio = Decimal(str(stop_loss_ratio)) / Decimal('100')
        self.target_ratio = Decimal(str(target_ratio)) / Decimal('100')
        self.reserve_ratio = Decimal(str(reserve_ratio)) / Decimal('100')
        if profit_retention_ratio is not None:
            self.profit_retention_ratio = Decimal(str(profit_retention_ratio)) / Decimal('100')
        else:
            self.profit_retention_ratio = None
        self.small_profit_threshold = Decimal(str(small_profit_threshold))
        self.large_profit_threshold = Decimal(str(large_profit_threshold))
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

    def _get_profit_retention_ratio(self, position: Position, extreme_price: Decimal) -> Decimal:
        """
        根据盈利幅度动态调整利润保留比例。
        盈利越大，保留比例越小（默认），避免大盈利时回吐过多。
        
        策略：
        - 盈利 < small_profit_threshold: 使用原始 reserve_ratio
        - 盈利 small_profit_threshold-large_profit_threshold: 使用二次函数插值（weight²）调整到 profit_retention_ratio（或 reserve_ratio），先慢后快
        - 盈利 > large_profit_threshold: 使用 profit_retention_ratio（或 reserve_ratio）
        
        :param position: 仓位信息
        :param extreme_price: 突破后的极值（最高价或最低价）
        :return: 动态调整后的利润保留比例 [0,1]
        """
        # 确定目标保留比例
        target_ratio = self.profit_retention_ratio if self.profit_retention_ratio is not None else self.reserve_ratio
        
        cost = position.cost_price
        if position.side == PositionSide.LONG:
            # 计算盈利比例
            profit_ratio = (extreme_price - cost) / cost
        else:
            # 空头：盈利比例 = (cost - extreme_price) / cost
            profit_ratio = (cost - extreme_price) / cost
        
        # 转换为百分比
        profit_pct = profit_ratio * Decimal('100')
        
        # 盈利 < small_profit_threshold: 使用原始比例
        if profit_pct <= self.small_profit_threshold:
            return self.reserve_ratio
        
        # 盈利 > large_profit_threshold: 使用目标保留比例
        if profit_pct >= self.large_profit_threshold:
            return target_ratio
        
        # 盈利在阈值之间: 二次函数插值（先慢后快）
        threshold_range = self.large_profit_threshold - self.small_profit_threshold
        if threshold_range <= Decimal('0'):
            return target_ratio
        # 归一化权重 [0, 1]
        normalized_weight = (profit_pct - self.small_profit_threshold) / threshold_range
        # 使用二次函数：weight²，实现先慢后快的变化
        actual_weight = normalized_weight * normalized_weight
        dynamic_ratio = self.reserve_ratio + (target_ratio - self.reserve_ratio) * actual_weight
        # 确保在合理范围内
        return max(min(dynamic_ratio, max(self.reserve_ratio, target_ratio)), 
                   min(self.reserve_ratio, target_ratio))

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
                    profit_pct = ((data.close - cost) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 触发止损离场, "
                              f"开仓价: {cost}, 当前价格: {data.close}, 止损价: {stop_price}, "
                              f"盈利比例: {profit_pct}%")
                    return False
                if data.close >= target_price:
                    state['target_broken'] = True
                    state['highest_since_break'] = data.close
                    # 使用开仓价为锚点初始化追踪出场价，ST方式仅上调
                    retention = self._get_profit_retention_ratio(position, state['highest_since_break'])
                    state['trailing_stop'] = cost + (state['highest_since_break'] - cost) * retention
                    profit_pct = ((state['highest_since_break'] - cost) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    retention_pct = (retention * Decimal('100')).quantize(Decimal('0.01'))
                    retained_profit_pct = (profit_pct * retention).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 目标突破, 进入追踪阶段, "
                              f"开仓价: {cost}, 极值价: {state['highest_since_break']}, 盈利比例: {profit_pct}%, "
                              f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%, "
                              f"追踪出场价: {state['trailing_stop']}")
                return True

            old_highest = state['highest_since_break']
            state['highest_since_break'] = max(state['highest_since_break'], data.close)
            # 仅上调：基于开仓价与破目标后的极值收盘价
            retention = self._get_profit_retention_ratio(position, state['highest_since_break'])
            new_trailing = cost + (state['highest_since_break'] - cost) * retention
            if state['trailing_stop'] is None or new_trailing > state['trailing_stop']:
                if state['highest_since_break'] > old_highest:
                    profit_pct = ((state['highest_since_break'] - cost) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    retention_pct = (retention * Decimal('100')).quantize(Decimal('0.01'))
                    retained_profit_pct = (profit_pct * retention).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 更新追踪出场价, "
                              f"极值价: {state['highest_since_break']}(前: {old_highest}), 盈利比例: {profit_pct}%, "
                              f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%, "
                              f"新追踪出场价: {new_trailing}(前: {state['trailing_stop']})")
                state['trailing_stop'] = new_trailing
            if data.close <= state['trailing_stop']:
                profit_pct = ((state['highest_since_break'] - cost) / cost * Decimal('100')).quantize(Decimal('0.01'))
                retention_pct = ((state['trailing_stop'] - cost) / (state['highest_since_break'] - cost) * Decimal('100')).quantize(Decimal('0.01')) if state['highest_since_break'] > cost else Decimal('0')
                retained_profit_pct = ((state['trailing_stop'] - cost) / cost * Decimal('100')).quantize(Decimal('0.01'))
                logger.info(f"目标追踪离场: {position.position_id} 触发追踪离场, "
                          f"开仓价: {cost}, 当前价格: {data.close}, 极值价: {state['highest_since_break']}, "
                          f"追踪出场价: {state['trailing_stop']}, 最高盈利比例: {profit_pct}%, "
                          f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%")
                return False
            return True

        elif position.side == PositionSide.SHORT:
            if not state['target_broken']:
                if data.close >= stop_price:
                    profit_pct = ((cost - data.close) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 触发止损离场, "
                              f"开仓价: {cost}, 当前价格: {data.close}, 止损价: {stop_price}, "
                              f"盈利比例: {profit_pct}%")
                    return False
                if data.close <= target_price:
                    state['target_broken'] = True
                    state['lowest_since_break'] = data.close
                    # 使用开仓价为锚点初始化追踪出场价，ST方式仅下调
                    retention = self._get_profit_retention_ratio(position, state['lowest_since_break'])
                    state['trailing_stop'] = cost - (cost - state['lowest_since_break']) * retention
                    profit_pct = ((cost - state['lowest_since_break']) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    retention_pct = (retention * Decimal('100')).quantize(Decimal('0.01'))
                    retained_profit_pct = (profit_pct * retention).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 目标突破, 进入追踪阶段, "
                              f"开仓价: {cost}, 极值价: {state['lowest_since_break']}, 盈利比例: {profit_pct}%, "
                              f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%, "
                              f"追踪出场价: {state['trailing_stop']}")
                return True

            old_lowest = state['lowest_since_break']
            state['lowest_since_break'] = min(state['lowest_since_break'], data.close)
            # 仅下调：基于开仓价与破目标后的极值收盘价
            retention = self._get_profit_retention_ratio(position, state['lowest_since_break'])
            new_trailing = cost - (cost - state['lowest_since_break']) * retention
            if state['trailing_stop'] is None or new_trailing < state['trailing_stop']:
                if state['lowest_since_break'] < old_lowest:
                    profit_pct = ((cost - state['lowest_since_break']) / cost * Decimal('100')).quantize(Decimal('0.01'))
                    retention_pct = (retention * Decimal('100')).quantize(Decimal('0.01'))
                    retained_profit_pct = (profit_pct * retention).quantize(Decimal('0.01'))
                    logger.info(f"目标追踪离场: {position.position_id} 更新追踪出场价, "
                              f"极值价: {state['lowest_since_break']}(前: {old_lowest}), 盈利比例: {profit_pct}%, "
                              f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%, "
                              f"新追踪出场价: {new_trailing}(前: {state['trailing_stop']})")
                state['trailing_stop'] = new_trailing
            if data.close >= state['trailing_stop']:
                profit_pct = ((cost - state['lowest_since_break']) / cost * Decimal('100')).quantize(Decimal('0.01'))
                retention_pct = ((cost - state['trailing_stop']) / (cost - state['lowest_since_break']) * Decimal('100')).quantize(Decimal('0.01')) if state['lowest_since_break'] < cost else Decimal('0')
                retained_profit_pct = ((cost - state['trailing_stop']) / cost * Decimal('100')).quantize(Decimal('0.01'))
                logger.info(f"目标追踪离场: {position.position_id} 触发追踪离场, "
                          f"开仓价: {cost}, 当前价格: {data.close}, 极值价: {state['lowest_since_break']}, "
                          f"追踪出场价: {state['trailing_stop']}, 最高盈利比例: {profit_pct}%, "
                          f"保留比例: {retention_pct}%, 保留利润: {retained_profit_pct}%")
                return False
            return True

        return False



