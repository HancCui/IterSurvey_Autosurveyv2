#%%
from datetime import datetime
from functools import wraps
import tiktoken

def format_duration(seconds):
    """将秒数格式化为 HH:MM:SS.ff"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"

class Timer:
    """上下文管理器：用于记录和打印执行时间"""
    def __init__(self, name):
        self.name = name
        self.start_datetime = None
        self.end_datetime = None

    def __enter__(self):
        self.start_datetime = datetime.now()
        print(f"\n{'='*60}")
        print(f"Starting:     {self.name}")
        print(f"Current Time: {self.start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_datetime = datetime.now()
        elapsed = (self.end_datetime - self.start_datetime).total_seconds()
        print(f"\n{'='*60}")
        print(f"Completed:    {self.name}")
        print(f"Used Time:    {format_duration(elapsed)}")
        print(f"Current Time: {self.end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}\n")

    @property
    def elapsed(self):
        if self.start_datetime is not None and self.end_datetime is not None:
            return (self.end_datetime - self.start_datetime).total_seconds()
        return None

class TimeTracker:
    """时间统计管理器（单例模式）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._stats = {}
            cls._instance._excluded_stats = {}
        return cls._instance

    def add_time(self, name, elapsed):
        """添加时间统计"""
        if name not in self._stats:
            self._stats[name] = 0.0
        self._stats[name] += elapsed

    def add_excluded_time(self, name, elapsed):
        """添加需要排除的时间统计"""
        if name not in self._excluded_stats:
            self._excluded_stats[name] = 0.0
        self._excluded_stats[name] += elapsed

    def get_time(self, name):
        """获取指定项的累计时间"""
        return self._stats.get(name, 0.0)

    def get_excluded_time(self, name):
        """获取指定项的排除累计时间"""
        return self._excluded_stats.get(name, 0.0)

    def get_all_stats(self):
        """获取所有时间统计"""
        return self._stats.copy()

    def get_all_excluded_stats(self):
        """获取所有排除时间统计"""
        return self._excluded_stats.copy()

    def get_total(self):
        """获取总计时间"""
        return sum(self._stats.values())

    def get_excluded_total(self):
        """获取排除时间总计"""
        return sum(self._excluded_stats.values())

    def reset(self):
        """重置所有统计"""
        self._stats = {}
        self._excluded_stats = {}

    def print_summary(self):
        """打印时间统计摘要"""
        print(f"\nTime Tracker Summary:")
        total = 0.0
        for name, elapsed in self._stats.items():
            print(f"  - {name}: {format_duration(elapsed)}")
            total += elapsed
        print(f"  Total: {format_duration(total)}")

        # 如果有排除时间，打印排除时间区域
        if self._excluded_stats:
            print(f"\nExcluded Time:")
            excluded_total = 0.0
            for name, elapsed in self._excluded_stats.items():
                print(f"  - {name}: {format_duration(elapsed)}")
                excluded_total += elapsed
            print(f"  Excluded Total: {format_duration(excluded_total)}")

        return total

def track_time(stat_name, excluded=False):
    """
    装饰器：自动记录函数执行时间

    Args:
        stat_name: 统计项名称（包含前缀，如 "[OutlineWriter] Update Paper Cards"）
        excluded: 是否记录为排除时间（默认 False）

    Usage:
        @track_time("[OutlineWriter] Update Paper Cards")
        def _update_paper_cards(self, ...):
            ...

        @track_time("[OutlineWriter] Excluded Task", excluded=True)
        def _excluded_task(self, ...):
            ...
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            tracker = TimeTracker()
            start = datetime.now()
            print(f"{stat_name} Start:      {start.strftime('%Y-%m-%d %H:%M:%S')}")

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                end = datetime.now()
                elapsed = (end - start).total_seconds()

                if excluded:
                    tracker.add_excluded_time(stat_name, elapsed)
                    total_time = tracker.get_excluded_time(stat_name)
                else:
                    tracker.add_time(stat_name, elapsed)
                    total_time = tracker.get_time(stat_name)

                print(f"{stat_name} End:        {end.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{stat_name} Use Time:   {format_duration(elapsed)}")
                print(f"{stat_name} Total Time: {format_duration(total_time)}\n")

        return wrapper
    return decorator

def track_token_usage(stage_name):
    """
    装饰器：自动记录函数的 API token 使用量

    Args:
        stage_name: 统计项名称（包含前缀，如 "[OutlineWriter] Update Paper Cards"）

    Usage:
        @track_token_usage("[OutlineWriter] Update Paper Cards")
        def _update_paper_cards(self, ...):
            ...

    注意：要求被装饰的函数所在的类必须有 get_token_usage() 方法
    默认都加入excluded
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # 记录开始时的 usage
            usage_before = self.get_token_usage()

            # 打印开始信息
            print(f"\n{stage_name} Start - Current Token Usage: "
                  f"Input={usage_before['input_tokens']:,}, "
                  f"Output={usage_before['output_tokens']:,}, "
                  f"Cost=${usage_before['cost']:.6f}")

            try:
                result = func(self, *args, **kwargs)
                return result
            finally:
                # 记录结束时的 usage
                usage_after = self.get_token_usage()

                # 计算差值
                usage_delta = {
                    'input_tokens': usage_after['input_tokens'] - usage_before['input_tokens'],
                    'output_tokens': usage_after['output_tokens'] - usage_before['output_tokens'],
                    'cost': usage_after['cost'] - usage_before['cost']
                }

                # 记录到 PriceTracker
                PriceTracker().record_excluded(stage_name, usage_delta)

                # 获取累计值
                accumulated = PriceTracker().get_excluded(stage_name)

                # 打印信息
                print(f"{stage_name}  End  - Current Token Usage: "
                      f"Input={usage_after['input_tokens']:,}, "
                      f"Output={usage_after['output_tokens']:,}, "
                      f"Cost=${usage_after['cost']:.6f}\n")
                print(f"{stage_name}  End  - Token Usage Delta:   "
                      f"Input={usage_delta['input_tokens']:,}, "
                      f"Output={usage_delta['output_tokens']:,}, "
                      f"Cost=${usage_delta['cost']:.6f}")
                print(f"{stage_name}  End  - Token Accumulate:    "
                      f"Input={accumulated['input_tokens']:,}, "
                      f"Output={accumulated['output_tokens']:,}, "
                      f"Cost=${accumulated['cost']:.6f}")
        return wrapper
    return decorator

class PriceTracker:
    """API 价格统计管理器（单例模式）"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.usage_data = {}
            cls._instance.excluded_usage_data = {}
            cls._instance.encoding = tiktoken.encoding_for_model("gpt-4o-mini")
            cls._instance.model_price = {
                "gpt-4o-mini": (0.15, 0.6)
            }
        return cls._instance

    def record_from_text(self, stage_name, input_text, output_text, model="gpt-4o-mini"):
        """从文本计算 tokens 并记录"""
        input_tokens = len(self.encoding.encode(input_text, disallowed_special=()))
        output_tokens = len(self.encoding.encode(output_text, disallowed_special=()))

        input_price, output_price = self.model_price.get(model, (0, 0))
        cost = (input_tokens/1000000) * input_price + (output_tokens/1000000) * output_price

        self.record(stage_name, {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'cost': cost
        })

    def record(self, stage_name, usage_dict):
        """
        记录某阶段的 API usage

        Args:
            stage_name: 阶段名称
            usage_dict: 包含 {'input_tokens': x, 'output_tokens': y, 'cost': z} 的字典
        """
        if stage_name not in self.usage_data:
            self.usage_data[stage_name] = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}

        self.usage_data[stage_name]['input_tokens'] += usage_dict.get('input_tokens', 0)
        self.usage_data[stage_name]['output_tokens'] += usage_dict.get('output_tokens', 0)
        self.usage_data[stage_name]['cost'] += usage_dict.get('cost', 0.0)

    def record_excluded(self, stage_name, usage_dict):
        """
        记录某阶段需要排除的 API usage

        Args:
            stage_name: 阶段名称
            usage_dict: 包含 {'input_tokens': x, 'output_tokens': y, 'cost': z} 的字典
        """
        if stage_name not in self.excluded_usage_data:
            self.excluded_usage_data[stage_name] = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}

        self.excluded_usage_data[stage_name]['input_tokens'] += usage_dict.get('input_tokens', 0)
        self.excluded_usage_data[stage_name]['output_tokens'] += usage_dict.get('output_tokens', 0)
        self.excluded_usage_data[stage_name]['cost'] += usage_dict.get('cost', 0.0)

    def get_all_usage(self):
        """获取所有阶段的 usage 数据"""
        return self.usage_data.copy()

    def get_all_excluded_usage(self):
        """获取所有阶段的排除 usage 数据"""
        return self.excluded_usage_data.copy()

    def get_excluded(self, stage_name):
        """
        获取指定阶段的累计 usage 数据

        Args:
            stage_name: 阶段名称

        Returns:
            包含 {'input_tokens': x, 'output_tokens': y, 'cost': z} 的字典
        """
        return self.excluded_usage_data.get(stage_name, {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0})

    def get_total(self):
        """获取总计的 tokens 和 cost"""
        total = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        for usage in self.usage_data.values():
            total['input_tokens'] += usage['input_tokens']
            total['output_tokens'] += usage['output_tokens']
            total['cost'] += usage['cost']
        return total

    def get_excluded_total(self):
        """获取排除用量的总计"""
        total = {'input_tokens': 0, 'output_tokens': 0, 'cost': 0.0}
        for usage in self.excluded_usage_data.values():
            total['input_tokens'] += usage['input_tokens']
            total['output_tokens'] += usage['output_tokens']
            total['cost'] += usage['cost']
        return total

    def reset(self):
        """重置所有统计"""
        self.usage_data = {}
        self.excluded_usage_data = {}

    def print_summary(self):
        """打印价格统计摘要"""
        print(f"\n{'='*60}")
        print(f"API Usage & Cost Breakdown:")
        print(f"{'='*60}")

        # 打印常规用量
        usage_stats = self.get_all_usage()
        for stage_name, usage in usage_stats.items():
            print(f"{stage_name:.<30}  Input: {usage['input_tokens']:>12,}  Output: {usage['output_tokens']:>12,}  Cost: ${usage['cost']:>12.6f}")

        total = self.get_total()
        print(f"{'='*60}")
        print(f"{'Total':.<30}  Input: {total['input_tokens']:>12,}  Output: {total['output_tokens']:>12,}  Cost: ${total['cost']:>12.6f}")

        # 如果有排除用量，打印排除用量区域
        excluded_stats = self.get_all_excluded_usage()
        if excluded_stats:
            print(f"\n{'='*60}")
            print(f"Excluded Usage & Cost:")
            print(f"{'='*60}")
            for stage_name, usage in excluded_stats.items():
                print(f"{stage_name:.<50}  Input: {usage['input_tokens']:>12,}  Output: {usage['output_tokens']:>12,}  Cost: ${usage['cost']:>12.6f}")

            excluded_total = self.get_excluded_total()
            print(f"{'='*60}")
            print(f"{'Excluded Total':.<50}  Input: {excluded_total['input_tokens']:>12,}  Output: {excluded_total['output_tokens']:>12,}  Cost: ${excluded_total['cost']:>12.6f}")

            # 计算净总计 (常规 - 排除)
            net_total = {
                'input_tokens': total['input_tokens'] - excluded_total['input_tokens'],
                'output_tokens': total['output_tokens'] - excluded_total['output_tokens'],
                'cost': total['cost'] - excluded_total['cost']
            }
            print(f"\n{'='*60}")
            print(f"{'Net Total':.<30}  Input: {net_total['input_tokens']:>12,}  Output: {net_total['output_tokens']:>12,}  Cost: ${net_total['cost']:>12.6f}")

        print(f"{'='*60}\n")

        return total
