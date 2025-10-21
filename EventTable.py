from sortedcontainers import SortedList
from bisect import bisect_left, bisect_right
import sys
from typing import Dict, List, Set, Optional

from ConstantEnums import Event

class EventTable:
    """
    高性能事件表系统（需安装sortedcontainers）
    特点：
    - 主表存储完整事件对象引用
    - 辅助表维护索引键快速访问
    - 更新操作零拷贝（直接引用替换）
    - 显式参数接口设计
    """
    
    def __init__(self):
        # 主存储：事件ID到事件对象的直接引用
        self.main_table: Dict[int, dict] = {}
        
        # 辅助表：索引键存储
        self.id_index = SortedList()  # 主键索引
        self.start_index = SortedList(key=lambda x: (x[0], x[1]))  # (start_ts, id)
        self.end_index = SortedList(key=lambda x: (x[0], x[1]))    # (end_ts, id)
        self.progress_index = SortedList(key=lambda x: (x[0], x[1])) # (progress, id)
        
        # 辅助键缓存：{id: (start_ts, end_ts, progress)}
        self.key_cache: Dict[int, tuple] = {}
        
        # ID生成器
        self._next_id = 1

    def create_event(self, start_ts: int, end_ts: Optional[int] = None, 
                    progress: Optional[int] = None) -> Optional[dict]:
        """
        创建新事件（显式参数接口）
        返回新事件ID
        """
        # 参数校验
        if start_ts < 0:
            raise ValueError("start_ts must >= 0")
        if end_ts is not None and start_ts > end_ts:
            raise ValueError("start_ts cannot > end_ts")
        if end_ts is None and (progress is None or progress < 0):
            raise ValueError("Ongoing event requires progress >=0")

        # 构建事件对象
        event_id = self._next_id
        event = {
            Event.ID.value: event_id,
            Event.START_TIMESTAMP.value: start_ts,
            Event.END_TIMESTAMP.value: end_ts if end_ts is not None else sys.maxsize,
            Event.PROGRESS.value: progress if (end_ts is None or end_ts == sys.maxsize) else None,
            Event.FINISHED.value: end_ts is not None and end_ts != sys.maxsize
        }
        
        # 更新存储
        self.main_table[event_id] = event
        self._update_secondary(event_id, event, is_new=True)
        self._next_id += 1
        return event

    def update_event(self, event_ref: dict):
        """精确更新变更字段对应的索引"""
        if Event.ID.value not in event_ref.keys() or event_ref[Event.ID.value] not in self.main_table:
            raise ValueError("Invalid event reference")
        if event_ref[Event.START_TIMESTAMP.value] < 0:
            raise ValueError("start_ts must >= 0")
        if event_ref is not None and event_ref[Event.START_TIMESTAMP.value] > event_ref[Event.END_TIMESTAMP.value]:
            raise ValueError("start_ts cannot > end_ts")
        if event_ref[Event.END_TIMESTAMP.value] not in {None,0} and (event_ref[Event.PROGRESS.value] is None or event_ref[Event.PROGRESS.value] < 0):
            raise ValueError("Ongoing event requires progress >=0")

        event_id = event_ref[Event.ID.value]
        original_ref = self.main_table[event_id]

        event_ref[Event.FINISHED.value] = (event_ref[Event.END_TIMESTAMP.value] != None and event_ref[Event.END_TIMESTAMP.value] != sys.maxsize)
        if event_ref[Event.FINISHED.value] and event_ref[Event.PROGRESS.value] is not None:
            event_ref[Event.PROGRESS.value] = None
        
        # # 引用未变化直接返回
        # if original_ref is event_ref:
        #     return

        # 获取新旧关键字段
        old_start, old_end, old_progress = self.key_cache[event_id]
        new_start = event_ref[Event.START_TIMESTAMP.value]
        new_end = event_ref[Event.END_TIMESTAMP.value]
        new_progress = event_ref[Event.PROGRESS.value]
        
        # 计算变更字段
        changed_fields = set()
        if old_start != new_start:
            changed_fields.add(Event.START_TIMESTAMP.value)
        if old_end != new_end:
            changed_fields.add(Event.END_TIMESTAMP.value)
        if old_progress != new_progress:
            changed_fields.add(Event.PROGRESS.value)
        
        # 没有关键字段变更只需替换引用
        if not changed_fields:
            self.main_table[event_id] = event_ref
            return
        
        # 状态变更检测（未结束 <-> 已结束）
        was_ended = old_end != sys.maxsize
        now_ended = new_end != sys.maxsize
        state_changed = was_ended != now_ended
        
        # 索引更新策略
        update_strategy = {
            Event.START_TIMESTAMP.value: self._update_start_index,
            Event.END_TIMESTAMP.value: self._update_end_index,
            Event.PROGRESS.value: self._update_progress_index
        }
        # 将旧值打包为字典
        old_values = {
            Event.START_TIMESTAMP.value: old_start,
            Event.END_TIMESTAMP.value: old_end,
            Event.PROGRESS.value: old_progress,
            Event.FINISHED.value: was_ended
        }

        # 传递给更新函数
        for field in changed_fields:
            update_strategy[field](event_id, old_values, event_ref, state_changed)
        
        
        # 更新缓存和引用
        self.key_cache[event_id] = (new_start, new_end, new_progress)
        self.main_table[event_id] = event_ref

    def _update_start_index(self, event_id: int, old: dict, new: dict, state_changed: bool):
        """精准更新开始时间索引"""
        # 删除旧索引
        self.start_index.discard((old[Event.START_TIMESTAMP.value], event_id))
        # 添加新索引
        self.start_index.add((new[Event.START_TIMESTAMP.value], event_id))

    def _update_end_index(self, event_id: int, old: dict, new: dict, state_changed: bool):
        """处理结束时间索引及关联变更"""
        # 更新结束时间索引
        self.end_index.discard((old[Event.END_TIMESTAMP.value], event_id))
        self.end_index.add((new[Event.END_TIMESTAMP.value], event_id))
        
        # 处理状态变化带来的进度索引变更
        if state_changed:
            if old[Event.FINISHED.value]:
                # 从已经结束变为未结束：添加进度索引
                self.progress_index.add((new[Event.PROGRESS.value], event_id))
            else:
                # 从未结束变为已经结束：移除进度索引
                self.progress_index.discard((old[Event.PROGRESS.value], event_id))

    def _update_progress_index(self, event_id: int, old: dict, new: dict, state_changed: bool):
        """条件更新进度索引"""
        if not state_changed and not new[Event.FINISHED.value] and old[Event.PROGRESS.value] != new[Event.PROGRESS.value]:
            # 仅当状态未变且仍为未结束时更新
            self.progress_index.discard((old[Event.PROGRESS.value], event_id))
            self.progress_index.add((new[Event.PROGRESS.value], event_id))
        if new[Event.FINISHED.value]:
            self.progress_index.discard((old[Event.PROGRESS.value], event_id))

    def search_events(self, 
            id_min: int = None, id_max: int = None,
            start_min: int = None, start_max: int = None,
            end_min: int = None, end_max: int = None,
            progress_min: int = None, progress_max: int = None,
            finished: bool = None
            ) -> List[dict]:
        """
        改进版多条件查询（支持全量查询包含已结束事件）
        参数规则：
        - None表示未设置该边界
        - 当至少一个边界非None时，未设置边界自动填充极值
        - 双None时跳过该条件
        - 如果设置了progress边界，则默认只查询未结束事件
        - 如果设置了finished，则默认只查询已结束或未结束事件，否则都查询
        """
        # 条件处理队列
        conditions = []
        
        # ID条件处理
        if id_min is not None or id_max is not None:
            id_min = id_min if id_min is not None else 0
            id_max = id_max if id_max is not None else sys.maxsize
            conditions.append((self._query_id_range, (id_min, id_max)))
        
        # 开始时间条件
        if start_min is not None or start_max is not None:
            start_min = start_min if start_min is not None else 0
            start_max = start_max if start_max is not None else sys.maxsize
            conditions.append((self._query_start_range, (start_min, start_max)))
        
        # 结束时间条件
        if end_min is not None or end_max is not None:
            end_min = end_min if end_min is not None else 0
            end_max = end_max if end_max is not None else sys.maxsize
            conditions.append((self._query_end_range, (end_min, end_max)))

        if progress_min is not None or progress_max is not None:
            # 动态生成进度查询集
            progress_min = progress_min if progress_min is not None else 0
            progress_max = progress_max if progress_max is not None else sys.maxsize
            conditions.append((self._query_progress_range, (progress_min, progress_max)))        


        if conditions == []:
            return list(self.main_table.values())
        else:
            result_ids = None
            for query_func, args in conditions:
                # 执行当前条件查询
                current_set = query_func(*args)
                
                # 短路1：当前条件无结果
                if not current_set:
                    return []
                
                # 首条件直接赋值
                if result_ids is None:
                    result_ids = current_set
                    continue
                    
                # 求交集
                result_ids &= current_set
                
                # 短路2：交集结果为空
                if not result_ids:
                    return []
            
            # 按ID排序返回引用
            events = [self.main_table[eid] for eid in sorted(result_ids)]
            if finished is not None:
                events = [event for event in events if event[Event.FINISHED.value] == finished]
            return events

    def get_event(self, event_id: int) -> Optional[dict]:
        return self.main_table.get(event_id, None)
    
    def get_recent_events(self, n:int)->list[dict]:
        """
        返回最近n个事件，最近发生的事件在前
        """
        recent_n = list(self.start_index.islice(stop=n, reverse=True))
        return [self.get_event(id) for (_,id) in recent_n]

    def delete_events(self, id_min: int = 0, id_max: int = sys.maxsize,
                    start_min: int = 0, start_max: int = sys.maxsize,
                    end_min: int = 0, end_max: int = sys.maxsize,
                    progress_min: int = 0, progress_max: int = sys.maxsize):
        """
        批量删除（参数与search_events一致）
        """
        targets = self.search_events(id_min, id_max, start_min, start_max,
                                     end_min, end_max, progress_min, progress_max)
        for event in targets:
            self._remove_from_secondary(event[Event.ID.value])
            del self.main_table[event[Event.ID.value]]
            del self.key_cache[event[Event.ID.value]]

    # 内部索引维护方法
    def _update_secondary(self, event_id: int, event: dict, is_new: bool):
        """更新辅助表和缓存"""
        # 移除旧索引（非新建时）
        if not is_new:
            self._remove_from_secondary(event_id)
        
        # 更新缓存
        new_keys = (event[Event.START_TIMESTAMP.value], event[Event.END_TIMESTAMP.value], 
                   event[Event.PROGRESS.value] if not event[Event.FINISHED.value] else None)
        self.key_cache[event_id] = new_keys
        
        # 添加新索引
        self.id_index.add(event_id)
        self.start_index.add((event[Event.START_TIMESTAMP.value], event_id))
        self.end_index.add((event[Event.END_TIMESTAMP.value], event_id))
        if not event[Event.FINISHED.value] and event[Event.PROGRESS.value] is not None:
            self.progress_index.add((event[Event.PROGRESS.value], event_id))

    def _remove_from_secondary(self, event_id: int):
        """从辅助表中移除"""
        if event_id not in self.key_cache:
            return
        
        # 获取旧键
        old_start, old_end, old_progress = self.key_cache[event_id]
        
        # 移除所有索引
        self.id_index.discard(event_id)
        self.start_index.discard((old_start, event_id))
        self.end_index.discard((old_end, event_id))
        if old_progress is not None:
            self.progress_index.discard((old_progress, event_id))

    # 查询方法
    def _query_id_range(self, min_val: int, max_val: int) -> Set[int]:
        """ID范围查询"""
        left = bisect_left(self.id_index, min_val)
        right = bisect_left(self.id_index, max_val)
        return set(self.id_index[left:right])

    def _query_start_range(self, min_val: int, max_val: int) -> Set[int]:
        """开始时间范围查询"""
        return self._range_query(self.start_index, min_val, max_val)

    def _query_end_range(self, min_val: int, max_val: int) -> Set[int]:
        """结束时间范围查询"""
        return self._range_query(self.end_index, min_val, max_val)

    def _query_progress_range(self, min_val: int, max_val: int) -> Set[int]:
        """进度范围查询"""
        return self._range_query(self.progress_index, min_val, max_val)

    def _range_query(self, index: SortedList, min_val: int, max_val: int) -> Set[int]:
        """左闭右开区间查询 [min_val, max_val)"""
        start = index.bisect_left((min_val, -sys.maxsize))
        
        # 处理最大边界值（sys.maxsize表示无穷大）
        if max_val == sys.maxsize:
            end = len(index)
        else:
            end = index.bisect_left((max_val, -sys.maxsize))  # 关键修改
        
        return {item[1] for item in index[start:end]}
    


if __name__ == "__main__":
    print("===== 事件表系统测试 =====")
    et = EventTable()

    print("\n--- 测试1：多事件创建（10个事件）---")
    # 创建不同类型事件（含已结束、进行中、特殊参数）
    events = [
        et.create_event(start_ts=1000, end_ts=2000),                     # 已结束事件
        et.create_event(start_ts=1500, progress=30),                     # 进行中
        et.create_event(start_ts=0, end_ts=sys.maxsize),                 # 极值测试
        et.create_event(start_ts=2000, progress=100),                    # 完成度100%
        et.create_event(start_ts=1800, progress = 0),                                  # 未设置结束时间
        et.create_event(start_ts=2000, end_ts=3000, progress=0),         # 进度0%
        et.create_event(start_ts=1600, end_ts=2400),                     # 常规事件
        et.create_event(start_ts=900, progress=50),                      # 早开始未结束
        et.create_event(start_ts=2000, end_ts=2500, progress=75),        # 高进度未完成
        et.create_event(start_ts=1500, progress=0)                       # 最小参数
    ]

    # 验证ID生成
    assert len({e[Event.ID.value] for e in events}) == 10, "ID重复"
    print(f"创建事件IDs：{[e[Event.ID.value] for e in events]}")

    print("\n--- 测试2：主表引用验证 ---")
    for e in events:
        assert et.get_event(e[Event.ID.value]) is e, f"事件{e[Event.ID.value]}引用不一致"

    print("\n--- 测试3：全量查询验证 ---")
    all_ids = {e[Event.ID.value] for e in et.search_events()}
    assert all_ids == {e[Event.ID.value] for e in events}, f"缺失事件：{all_ids ^ {e[Event.ID.value] for e in events}}"
    print(f"全量查询结果：{len(all_ids)}事件")

    print("\n--- 测试4：复合条件查询 ---")
    # 查询条件：start_ts >=1500 且 progress <100
    results = et.search_events(
        start_min=1500,
        progress_max=99
    )
    expected_ids = {events[i][Event.ID.value] for i in [1,4,9]}
    assert {e[Event.ID.value] for e in results} == expected_ids, f"结果不符：{results}"
    print(f"符合条件的事件IDs：{[e[Event.ID.value] for e in results]}")

    print("\n--- 测试5：状态转换批量测试 ---")
    # 结束3个未完成事件
    to_finish = [events[1], events[4], events[8]]
    for e in to_finish:
        e[Event.END_TIMESTAMP.value] = 4000
        et.update_event(e)
        assert e[Event.FINISHED.value], f"事件{e[Event.ID.value]}未标记完成"
        assert Event.PROGRESS.value not in e or e[Event.PROGRESS.value] == None, f"事件{e[Event.ID.value]}未清除进度"

    print("\n--- 测试6：删除操作扩展测试 ---")
    # 删除ID为偶数的前5个事件
    delete_ids = [e[Event.ID.value] for e in events[:5] if e[Event.ID.value]%2 ==0]
    et.delete_events(id_min=min(delete_ids), id_max=max(delete_ids)+1)
    remaining = et.search_events()
    assert all(e[Event.ID.value] not in delete_ids for e in remaining), "删除失败"
    print(f"剩余事件数：{len(remaining)} (应删除{len(delete_ids)}个)")

    print("\n--- 测试7：边界条件查询 ---")
    # 查询时间戳为系统最大值的唯一事件
    max_ts_event = et.search_events(start_min=sys.maxsize-1)
    assert len(max_ts_event) ==1 and max_ts_event[0][Event.ID.value]==events[2][Event.ID.value], "边界事件未找到"

    print("\n===== 全量测试通过 =====")