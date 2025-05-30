#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ClickHouse 数据源实现。
从 ClickHouse 数据库中读取 K 线数据。
"""
from datetime import datetime
from decimal import Decimal
from typing import Iterator, List

from leek_core.models import TimeFrame, DataType, TradeInsType, KLine, Field, FieldType, ChoiceType, AssetType
from leek_core.utils import get_logger
from .base import DataSource

logger = get_logger(__name__)


class ClickHouseKlineDataSource(DataSource):
    supported_data_type: DataType = DataType.KLINE
    display_name = "ClickHouse K线"
    # 声明支持的资产类型
    supported_asset_type: DataType = {AssetType.STOCK, AssetType.CRYPTO}
    backtest_supported: bool = True
    init_params: List[Field] = [
        Field(name="host", label="主机", type=FieldType.STRING, required=True, default="127.0.0.1"),
        Field(name="port", label="端口", type=FieldType.INT, required=True, default=9000),
        Field(name="user", label="用户名", type=FieldType.STRING, required=True, default="default"),
        Field(name="password", label="密码", type=FieldType.STRING, default=""),
        Field(name="database", label="数据库名", type=FieldType.STRING, default="default"),
    ]
    # 声明显示名称
    verbose_name = "ClickHouse K线"
    """
    ClickHouse K线数据源实现类。
    
    支持从 ClickHouse 数据库中读取 K 线数据，
    表结构示例：
    CREATE TABLE klines (
        id UInt64,
        market String,
        timeframe String,
        timestamp UInt64,
        symbol String,
        quote_currency String,
        ins_type UInt8, -- 交易品种类型（TradeInsType的枚举值）
        open Decimal(18, 8),
        high Decimal(18, 8),
        low Decimal(18, 8),
        close Decimal(18, 8),
        volume Decimal(18, 4),
        amount Decimal(18, 2)
    ) ENGINE = ReplacingMergeTree()
    PRIMARY KEY (market, timeframe, timestamp, symbol, quote_currency, ins_type)
    ORDER BY (market, timeframe, timestamp, symbol, quote_currency, ins_type)
    PARTITION BY timeframe
    SETTINGS index_granularity=4096,
            index_granularity_bytes=1048576,
            enable_mixed_granularity_parts=1;
    """

    def __init__(self, host: str, port: int = 9000, user: str = 'default',
                 password: str = '', database: str = 'default'):
        """
        初始化 ClickHouse 数据源。
        
        参数:
            name: 数据源名称
            host: ClickHouse 服务器主机
            port: ClickHouse 服务器端口
            user: 用户名
            password: 密码
            database: 数据库名
            table: 表名
            market: 市场标识（数据来源），默认为'okx'
            quote_currency: 计价币种，默认为'USDT'
            ins_type: 交易品种类型，默认为现货(SPOT)
            instance_id: 数据源实例ID，用于跟踪数据流向
        """
        super().__init__()
        self.config = {
            'host': host,
            'port': port,
            'user': user,
            'password': password,
            'database': database,
            'table': "klines",
        }
        self._client = None
        self._markets = set()
        self._symbols = set()
        self._timeframes = set()
        self._quote_currencies = set()
        self._ins_types = set()

    def connect(self) -> bool:
        """
        连接到 ClickHouse 数据库。
        
        返回:
            bool: 连接成功返回 True，否则返回 False
        """
        if self.is_connected:
            return True

        from clickhouse_driver import Client
        from clickhouse_driver.errors import Error as ClickHouseError
        try:
            self._client = Client(
                host=self.config['host'],
                port=self.config['port'],
                user=self.config['user'],
                password=self.config['password'],
                database=self.config['database']
            )
            # 测试连接
            self._client.execute('SELECT 1')

            # 扫描可用的交易对和时间周期
            self._scan_data()
            return True
        except ClickHouseError as e:
            logger.error(f"连接 ClickHouse 数据库失败: {e}")
            self.is_connected = False
            return False

    def disconnect(self) -> bool:
        """
        断开与 ClickHouse 数据库的连接。
        
        返回:
            bool: 断开连接成功返回 True，否则返回 False
        """
        if self._client:
            try:
                self._client.disconnect()
            except Exception as e:
                logger.error(f"断开 ClickHouse 数据库连接时出错: {e}")

        self._client = None
        self.is_connected = False
        return True

    def _scan_data(self):
        """
        扫描数据库，获取可用的交易对、时间周期和计价币种。
        """
        if not self._client:
            return
        from clickhouse_driver.errors import Error as ClickHouseError
        try:
            # 获取唯一市场标识
            query = f"SELECT DISTINCT market FROM {self.config['table']}"
            result = self._client.execute(query)
            self._markets = set(item[0] for item in result)

            # 获取唯一的交易对
            query = f"SELECT DISTINCT symbol FROM {self.config['table']}"
            result = self._client.execute(query)
            self._symbols = set(item[0] for item in result)

            # 获取唯一的时间周期
            query = f"SELECT DISTINCT timeframe FROM {self.config['table']}"
            result = self._client.execute(query)
            self._timeframes = set(TimeFrame(item[0]) for item in result)

            # 获取唯一的计价币种 
            # 这里只根据 market 过滤，因为我们希望看到所有的计价币种
            query = f"SELECT DISTINCT quote_currency FROM {self.config['table']}"
            result = self._client.execute(query)
            self._quote_currencies = set(item[0] for item in result)

            # 获取唯一的交易品种类型
            query = f"SELECT DISTINCT ins_type FROM {self.config['table']}"
            result = self._client.execute(query)
            self._ins_types = set(int(item[0]) for item in result)
        except ClickHouseError as e:
            logger.error(f"扫描 ClickHouse 数据库时出错: {e}")

    def get_supported_parameters(self) -> List[Field]:
        self.connect()
        ins_types = [(i, str(i)) for i in TradeInsType if i in self._ins_types]
        return [
            Field(name='symbol', label='交易标的', type=FieldType.RADIO, required=True, choices=list(self._symbols),
                  choice_type=ChoiceType.STR),
            Field(name='timeframe', label='时间周期', type=FieldType.RADIO, required=True,
                  choices=list(self._timeframes), choice_type=ChoiceType.STR),
            Field(name='market', label='市场标识', type=FieldType.RADIO, required=True, choices=list(self._markets),
                  choice_type=ChoiceType.STR),
            Field(name='quote_currency', label='计价币种', type=FieldType.RADIO, required=True,
                  choices=list(self._quote_currencies), choice_type=ChoiceType.STR),
            Field(name='ins_type', label='交易品种类型', type=FieldType.RADIO, required=True,
                  choices=ins_types, choice_type=ChoiceType.INT),
        ]

    def get_history_data(
            self,
            start_time: datetime | int = None,
            end_time: datetime | int = None,
            limit: int = None,
            symbol: str = None,
            timeframe: TimeFrame | str = None,
            market: str = None,
            quote_currency: str = None,
            ins_type: TradeInsType | int = None,
            **kwargs,
    ) -> Iterator[KLine]:
        """
        从 ClickHouse 获取指定交易对和时间周期的 K 线数据。
        
        参数:
            symbol: 交易对符号
            timeframe: K线时间周期
            start_time: 开始时间（可以是 datetime 对象或毫秒时间戳），为空时默认取最近一个月
            end_time: 结束时间（可以是 datetime 对象或毫秒时间戳），为空时默认取当前时间
            limit: 返回的最大记录数
            market: 市场标识，如果为None则使用初始化时的market
            quote_currency: 计价币种，如果为None则使用初始化时的quote_currency
            ins_type: 交易品种类型，如果为None则使用初始化时的ins_type

        返回:
            Iterator[KLine]: K线对象的迭代器
        """
        if not self.is_connected or not self._client:
            logger.warning("未连接到 ClickHouse 数据库")
            # 生成器函数应直接返回，不需要创建迭代器
            return
        from clickhouse_driver.errors import Error as ClickHouseError
        try:
            if isinstance(timeframe, TimeFrame):
                timeframe_value = timeframe.value
            else:
                timeframe_value = str(timeframe)

            # 确定使用哪个market、计价币种和交易品种类型
            market_value = market if market is not None else self.config['market']
            quote_currency_value = quote_currency if quote_currency is not None else self.config['quote_currency']

            # 处理ins_type参数
            if ins_type is not None:
                if isinstance(ins_type, TradeInsType):
                    ins_type_value = ins_type.value
                else:
                    ins_type_value = ins_type
            else:
                ins_type_value = self.config['ins_type']

            # 如果start_time和end_time都为空，默认取最近一个月的数据
            current_time = datetime.now()
            if end_time is None:
                end_time = current_time

            if start_time is None:
                # 计算一个月前的时间
                one_month_ago = current_time.replace(month=current_time.month - 1) if current_time.month > 1 else \
                    current_time.replace(year=current_time.year - 1, month=12)
                start_time = one_month_ago

            # 构建查询语句
            query = f"""
                SELECT id, market, timestamp, symbol, quote_currency, ins_type, timeframe, open, high, low, close, volume, amount
                FROM {self.config['table']}
                WHERE symbol = %(symbol)s 
                  AND timeframe = %(timeframe)s
                  AND market = %(market)s
                  AND quote_currency = %(quote_currency)s
                  AND ins_type = %(ins_type)s
            """

            params = {
                'symbol': symbol,
                'timeframe': timeframe_value,
                'market': market_value,
                'quote_currency': quote_currency_value,
                'ins_type': ins_type_value
            }

            # 添加时间范围条件
            if isinstance(start_time, datetime):
                start_time = int(start_time.timestamp() * 1000)
            query += " AND timestamp >= %(start_time)s"
            params['start_time'] = start_time

            if isinstance(end_time, datetime):
                end_time = int(end_time.timestamp() * 1000)
            query += " AND timestamp <= %(end_time)s"
            params['end_time'] = end_time

            # 添加排序条件 - 按时间戳降序排序
            query += " ORDER BY timestamp ASC"

            # 添加限制条件
            if limit:
                query += f" LIMIT {limit}"

            # 执行查询
            result = self._client.execute(query, params)

            # 如果没有结果，返回空迭代器
            if not result:
                return

            # 使用生成器直接返回 KLine 对象
            for row in result:
                id_val, market, timestamp, symbol, quote_currency, ins_type, timeframe, open_val, high, low, close, volume, amount = row

                # 创建并生成 KLine 对象
                yield KLine(
                    symbol=symbol,
                    market=market,
                    open=Decimal(str(open_val)),
                    high=Decimal(str(high)),
                    low=Decimal(str(low)),
                    close=Decimal(str(close)),
                    volume=Decimal(str(volume)) if volume is not None else Decimal('0'),
                    amount=Decimal(str(amount)) if amount is not None else Decimal('0'),
                    start_time=timestamp,
                    end_time=timestamp + TimeFrame(timeframe).milliseconds,
                    current_time=timestamp + TimeFrame(timeframe).milliseconds,
                    timeframe=TimeFrame(timeframe),
                    quote_currency=quote_currency,
                    ins_type=TradeInsType(ins_type),
                    is_finished=True
                )

        except ClickHouseError as e:
            logger.error(f"获取 K 线数据时出错: {e}")
            # 当出错时返回空生成器
            return
