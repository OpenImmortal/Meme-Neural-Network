from typing import TYPE_CHECKING

from typing import Dict, Set, List, Tuple, DefaultDict
from collections import defaultdict
import heapq
import math
from ConstantEnums import *
if TYPE_CHECKING:
    from NeuralModels import Baby, Connection, Cell
    from ParamGroup import *


class BranchInfoActivationRegistry:
    def __init__(self):
        from ParamGroup import BranchInfo
        # 核心存储结构：时间戳 → {BranchInfoes}
        self.activation_map: Dict[int, Set[BranchInfo]] = defaultdict(set)
        
        # 辅助索引：conn_id → [按时间排序的BranchInfoInfoes]
        self.conn_index: Dict[int, List[BranchInfo]] = defaultdict(list)

    def add_BranchInfo(self, branchInfo: "BranchInfo") -> None:
        """注册新的激活分支， 在connection激活branch后调用"""
        self.activation_map[branchInfo.get_timestamp()].add(branchInfo)
        self.conn_index[branchInfo.get_conn_id()].append(branchInfo)
        # 保持时间有序性
        self.conn_index[branchInfo.get_conn_id()].sort(key=lambda x: x.get_timestamp())

    def find_optimal_matches(
        self, 
        target_branchInfoes: List["BranchInfo"],
        time_ranges: List[Tuple[int, int]],  # 每个target对应的(负范围, 正范围)
        k: int,  # 时间差值阈值
        reward_weights =  {'positive': 1.5, 'mixed': 1.2, 'pending': 1.0, 'negative': 0.8}
    ) -> Dict[float, Tuple[Tuple[int, int], Set["BranchInfo"]]]:
        """
        核心匹配算法实现
        返回结构：{分数: ((最小时间差, 最大时间差), 分支集合)}
        """

        # 阶段1：时间范围筛选
        valid_pairs = self._filter_by_time_ranges(target_branchInfoes, time_ranges)
        
        # 阶段2：优先级预排序
        sorted_candidates = self._sort_candidates(valid_pairs)
        
        # 阶段3：动态合并相同连接的分支
        merged_groups = self._merge_conn_groups(sorted_candidates, k)
        
        # 阶段4：计算最终得分
        return self._calculate_scores(merged_groups, valid_pairs, reward_weights)

    def _filter_by_time_ranges(
        self,
        targets: List["BranchInfo"],
        ranges: List[Tuple[int, int]],
        self_pairing: bool = False
    ) -> List[Tuple["BranchInfo", "BranchInfo"]]:
        """应用时间范围过滤规则
        - return: List[Tuple[target, branchInfo]]
        
        """
        # 默认禁止自连接.如果后续发现自连接必要，请调整相应代码


        valid = []
        for target, (neg, pos) in zip(targets, ranges):
            # 计算有效时间窗口（处理正负覆盖）
            start = target.get_timestamp() + neg
            end = target.get_timestamp() + pos
            if start > end:
                raise ValueError("Invalid range! start > end")
                
            # 查询时间范围内的所有分支
            for ts in range(start, end + 1):
                if ts in self.activation_map:
                    for branchInfo in self.activation_map[ts]:
                        if not self_pairing and target.get_conn_id() == branchInfo.get_conn_id():
                            continue
                        else:
                            valid.append((target, branchInfo))
        return valid

    def _sort_candidates(
        self, 
        pairs: List[Tuple["BranchInfo", "BranchInfo"]]
    ) -> List[Tuple["BranchInfo", "BranchInfo"]]:
        """按优先级排序候选对"""
        reward_weights = {'positive': 4, 'mixed': 3, 'pending': 2, 'negative': 1}
        return sorted(
            pairs,
            key=lambda x: (
                -reward_weights[x[1].reward_type],
                -(x[1].splitting_factor + x[1].connecting_factor),
                abs(x[1].get_timestamp() - x[0].get_timestamp())
            ),
            reverse=True
        )

    @staticmethod
    def _calculate_branch_time_diff(target:"BranchInfo", branch:"BranchInfo"):
        """
        计算branch相距target的时间，如果target发生在前（关注target的后向事件），为正，否则（关注target的前向事件）为负
        """
        if branch.get_timestamp()>=target.get_timestamp():
            current_diff = (branch.get_timestamp() - target.get_timestamp())
        else:
            current_diff = -(target.get_timestamp() - branch.get_timestamp() )
        return current_diff
    
    def _merge_conn_groups(
        self, 
        candidates: List[Tuple["BranchInfo", "BranchInfo"]], 
        k: int = 2
    ) -> List[List["BranchInfo"]]:
        """合并相同conn_id且与对应target时间差接近的分支"""
        conn_group_map = defaultdict(list[Tuple["BranchInfo", "BranchInfo"]])
        seen_pairs = set()
        for target, branch in candidates:
            if (target, branch) in seen_pairs:
                continue
            seen_pairs.add((target, branch))
            conn_group_map[branch.get_conn_id()].append( (target, branch) )

        merged_groups = []

            

        for conn_id, target_branch_pairs in conn_group_map.items():
            sorted_pairs = sorted(target_branch_pairs, 
                                key=lambda x: self._calculate_branch_time_diff(x[0],x[1]))
            
            current_group = []
            current_group_signal_type = None
            group_min_diff = None
            group_max_diff = None
            
            for target, branch in sorted_pairs:
                current_diff = self._calculate_branch_time_diff(target=target,branch=branch)
                branch_signal_type = branch.links[0][Link.EVENT.value][Event.SIGNAL.value]
                # # 基础过滤：绝对阈值
                # if current_diff > k:
                #     continue
                
                    
                if not current_group:
                    # 初始化第一个元素
                    current_group.append(branch)
                    current_group_signal_type = branch_signal_type
                    group_min_diff = current_diff
                    group_max_diff = current_diff
                else:
                    # 动态区间判断
                    lower_bound = group_min_diff - k
                    upper_bound = group_max_diff + k

                    if lower_bound <= current_diff <= upper_bound and current_group_signal_type == branch_signal_type:
                        # 加入当前组并更新极值
                        current_group.append(branch)
                        group_min_diff = min(group_min_diff, current_diff)
                        group_max_diff = max(group_max_diff, current_diff)
                    else:
                        # 结算当前组
                        merged_groups.append(current_group)
                        # 创建新组
                        current_group = [branch]
                        group_min_diff = current_diff
                        group_max_diff = current_diff
            
            # 处理末梢组
            if current_group:
                merged_groups.append(current_group)
        
        return merged_groups

    def _calculate_scores(
        self, 
        groups: List[List["BranchInfo"]],
        original_pairs: List[Tuple["BranchInfo", "BranchInfo"]],
        reward_weights,
        time_view_base: int  = 10,
        base_score: float = 1.0,
        diff_tolerance: float = 10
    ) -> List[Tuple[float, Dict]]:
        """最终版得分计算（修复键问题）"""
        # 创建基于唯一标识的映射
        pair_map = {(b.get_conn_id(), b.get_timestamp()): t for t, b in original_pairs}

        
        result = []
        
        for group in groups:
            if not group:
                continue
            
            conn_id = group[0].get_conn_id()
            if any(b.get_conn_id() != conn_id for b in group):
                continue

            time_diffs = []
            related_targets = set()
            
            for b in group:
                map_key = (b.get_conn_id(), b.get_timestamp())
                target = pair_map.get(map_key)
                if not target:
                    continue
                    
                diff = self._calculate_branch_time_diff(target=target,branch=b) ## 被触发时刻必须
                time_diffs.append(diff)
                related_targets.add(target)
            
            if not time_diffs:
                continue
                
            min_diff, max_diff = min(time_diffs), max(time_diffs)
            total_score = sum(
                (
                    # b.splitting_factor + b.connecting_factor + 重复考虑了，去掉
                    base_score
                    ) * reward_weights[b.reward_type]
                for b in group
            ) / (
                max(
                    math.log(
                            max(abs(max_diff),1),time_view_base
                        ),1
                ) * (
                    max((max_diff - min_diff)/diff_tolerance, 1)
                )
            )

            score = round(total_score, 3)
            result.append((
                score,
                {
                    "conn_id": conn_id,
                    "time_diff_range": (min_diff, max_diff),
                    "branches": set(group),
                    "score": score,
                    "related_targets": list(related_targets)
                }
            ))
        
        return sorted(result, key=lambda x: -x[0])





from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Set

import numpy as np
from collections import Counter


class CellBehaviorController:
    def __init__(self, baby:"Baby", registry:"BranchInfoActivationRegistry", cooldown = 1000, observation_window = 100):
        """初始化所有存储结构"""
        # 核心存储结构：四层嵌套字典
        # 结构：cell_id → conn_id → reward_type → {"branches": {ts: BranchInfo}, "connecting": x, "splitting": y}
        self.cell_storage: Dict[int, Dict[int, Dict[str, Dict]]] = defaultdict(
            lambda: defaultdict(
                lambda: {
                    "positive": {"branches": {}, "connecting": 0.0, "splitting": 0.0},
                    "negative": {"branches": {}, "connecting": 0.0, "splitting": 0.0},
                    "pending": {"branches": {}, "connecting": 0.0, "splitting": 0.0}
                }
            )
        )
        
        # 阈值存储：cell_id → 动作类型 → 方向 → 阈值
        self.thresholds: Dict[int, Dict[str, Dict[str, float]]] = defaultdict(
            lambda: {
                # TODO: 这只是一个初始值，请在 @fun _recalculate_thresholds 中修改
                "connecting": {"in": 0.1, "out": 0.1}, # in 代表增加cell的in degree，也就是作为新conn的downstream cell
                "splitting": {"in": 0.2, "out": 0.2}    # out 代表增加cell的out degree，也就是作为新conn的upstream cell
            }
        )
        
        # 冷却记录：(上游cell, 下游cell) → 过期时间
        self.cooldown = cooldown
        self.cooldown_map: Dict[Tuple[int, int], int] = {}
        
        # 注册表实例
        self.registry = registry

        self.baby:"Baby" = baby

        self.observation_window = observation_window

        self.vector_encoding =  {
            'positive': np.array([1, 0, 0]),
            'negative': np.array([-1, 0, 0]),
            'mixed':    np.array([0, 1, 0]),
            'pending':  np.array([0, 0, 1])
        }


    def get_upstream_cellID(self,conn_id:"int"):
        """
        获取连接的上游细胞。特别地，由于Receptor的输入信号没有上游细胞，仍采用Receptor（该输入信号conn的下游细胞）
        """
        connObj = self.baby.connections[conn_id]
        if connObj.isReceptor:
            return connObj.downstream_cell_id
        return connObj.upstream_cell_id


    #--------------------------------------------------#
    # 更新分支信息
    # 输入：BranchInfo对象
    # 处理：添加/更新或移除分支，更新统计值
    #--------------------------------------------------#
    def update_branch(self,timestamp:int, branch: "BranchInfo") -> None:
        """严格维护统计值正确性的更新方案"""
        # 步骤1：获取关联信息
        cell_id = self.get_upstream_cellID(branch.get_conn_id())
        conn_id = branch.get_conn_id()
        if branch.reward_type in ["positive", "mixed"]: 
            reward_type = "positive" 
        elif branch.reward_type in ["pending"]:
            reward_type = "pending"
        else:
            reward_type = "negative"
        ts = branch.get_timestamp()
        

        # 步骤2：检查reward_type是否变化
        old_reward_type = None
        for rt in ["positive", "pending", "negative"]:
            if ts in self.cell_storage[cell_id][conn_id][rt]["branches"]:
                old_reward_type = rt
                break
        if old_reward_type:
            # 从旧容器移除
            removed_branch = self.cell_storage[cell_id][conn_id][old_reward_type]["branches"].pop(ts)
            # 重新计算旧容器的统计值
            conn_data = self.cell_storage[cell_id][conn_id][old_reward_type]
            conn_data["connecting"] = sum(b.connecting_factor for b in conn_data["branches"].values())
            conn_data["splitting"] = sum(b.splitting_factor for b in conn_data["branches"].values())


        # 步骤3：处理过期分支
        if branch.outdated:
            if ts in self.cell_storage[cell_id][conn_id][reward_type]["branches"]:
                removed_branch = self.cell_storage[cell_id][conn_id][reward_type]["branches"].pop(ts)
                # 直接扣除旧值
                self.cell_storage[cell_id][conn_id][reward_type]["connecting"] -= removed_branch.connecting_factor
                self.cell_storage[cell_id][conn_id][reward_type]["splitting"] -= removed_branch.splitting_factor
            return
        
        # 步骤4：强制更新统计值
        is_existing = ts in self.cell_storage[cell_id][conn_id][reward_type]["branches"]
        self.cell_storage[cell_id][conn_id][reward_type]["branches"][ts] = branch
        
        if is_existing:
            # 重新计算该conn_id下全部统计值
            conn_data = self.cell_storage[cell_id][conn_id][reward_type]
            conn_data["connecting"] = sum(b.connecting_factor for b in conn_data["branches"].values())
            conn_data["splitting"] = sum(b.splitting_factor for b in conn_data["branches"].values())
        else:
            # 累加新分支值
            self.cell_storage[cell_id][conn_id][reward_type]["connecting"] += branch.connecting_factor
            self.cell_storage[cell_id][conn_id][reward_type]["splitting"] += branch.splitting_factor
        
        # 步骤5：触发监听
        self._trigger_action(timestamp=timestamp, cell_id=cell_id)


    #--------------------------------------------------#
    # 更新连接状态
    # 输入：Connection对象和动作类型 (断开，新建)
    # 处理：维护冷却表，删除相关数据，重新计算阈值
    #--------------------------------------------------#
    def update_connection(self, timestamp: int, conn: "Connection", action: str = "break") -> None:
        """步骤分解："""
        up_cell = conn.upstream_cell_id
        down_cell = conn.downstream_cell_id
        
        # 步骤1：处理断开连接
        if action == "break":
            # 添加冷却
            self.cooldown_map[(up_cell, down_cell)] = timestamp + self.cooldown
            
            # 删除相关conn_id的数据
            for cell in [up_cell, down_cell]:
                if conn.get_conn_id() in self.cell_storage[cell]:
                    del self.cell_storage[cell][conn.get_conn_id()]
        
        # 步骤2：重新计算阈值
        self._recalculate_thresholds(up_cell)
        self._recalculate_thresholds(down_cell)
        
        # 步骤3：触发监听
        self._trigger_action(timestamp=timestamp, cell_id=up_cell)
        self._trigger_action(timestamp=timestamp, cell_id=down_cell)


    #--------------------------------------------------#
    # 阈值重新计算（核心逻辑）
    # 输入：cell_id
    # 处理：根据度数动态调整四个阈值
    #--------------------------------------------------#
    def _recalculate_thresholds(self, cell_id: int) -> None:
        cell: "Cell" = self.baby.cells[cell_id]  # 假设存在获取Cell的方法
        total_deg = cell.get_degree()
        max_deg = cell.get_max_degree()
        
        # 计算connecting阈值
        base_conn = 0.01
        scale_conn = 1 + (total_deg / max_deg) ** 2
        deg_diff = cell.get_in_degree() - cell.get_out_degree()
        imbalance_factor = deg_diff * 0.2 if max(cell.get_in_degree(), cell.get_out_degree()) > max_deg // 2 else 0
        
        self.thresholds[cell_id]["connecting"]["in"] = base_conn * (scale_conn + max(imbalance_factor,0))
        self.thresholds[cell_id]["connecting"]["out"] = base_conn * (scale_conn + max(-imbalance_factor,0))

        # 计算splitting阈值
        base_split = 0.02
        scale_split = 1 - (total_deg / max_deg) ** 2
        self.thresholds[cell_id]["splitting"]["in"] = self.thresholds[cell_id]["connecting"]["in"] + base_split * max(scale_split, 0)
        self.thresholds[cell_id]["splitting"]["out"] = self.thresholds[cell_id]["connecting"]["out"] + base_split * max(scale_split, 0)


    def _calculate_new_conn_signal_type(self, up_conn_branch:"BranchInfo", down_conn_branch:"BranchInfo"):
        
        """
        确定反馈调节的新连接信号类型
        """
        new_conn_signal_type = None

        up_conn_event = up_conn_branch.links[0][Link.EVENT.value]
        up_conn_reward_type = up_conn_branch.reward_type

        down_conn_event = down_conn_branch.links[0][Link.EVENT.value]
        down_conn_id = down_conn_branch.get_conn_id()
        down_conn_obj = self.baby.connections[down_conn_id]
        down_conn_trigger_signal_type = Signal_E.E.value
        if down_conn_obj.isActuator:
            # 如果下游连接是执行器，那么其输出conn的状态总为Actuator Cell状态，进而其Signal type也由具体事件动态决定（而非connObj的signal type决定）
            down_conn_trigger_signal_type = down_conn_event[Event.SIGNAL.value]
        else:
            down_conn_trigger_signal_type = Signal_E.E.value
        down_conn_reward_type = down_conn_branch.reward_type
        

        if down_conn_reward_type in [RewardType.NEGATIVE.value]:
            if down_conn_trigger_signal_type == Signal_E.E.value:
                new_conn_signal_type = Signal_F.F.value
            elif down_conn_trigger_signal_type == Signal_F.F.value:
                new_conn_signal_type = Signal_E.E.value
            else:
                raise ValueError("Invalid down_conn_trigger_signal_type for Negative Reward")
        elif down_conn_reward_type in [RewardType.POSITIVE.value]:
            if down_conn_trigger_signal_type == Signal_E.E.value:
                new_conn_signal_type = Signal_E.E.value
            elif down_conn_trigger_signal_type == Signal_F.F.value:
                new_conn_signal_type = Signal_F.F.value
            else:
                raise ValueError("Invalid down_conn_trigger_signal_type for Positive Reward")
        elif down_conn_reward_type in [RewardType.PENDING.value, RewardType.MIXED.value]:
            new_conn_signal_type = Signal_E.E.value
        
        else:
            raise ValueError("Unknown down_conn_reward_type")

        return new_conn_signal_type
    #--------------------------------------------------#
    # 动作触发主函数
    # 输入：评分结果列表
    # 处理：修饰分数 → 筛选候选 → 验证有效性
    #--------------------------------------------------#
    def _trigger_action(self, timestamp:int, cell_id: int ) -> List[Dict]:
        """重构后的动作触发函数
        
        TODO 向baby传递消息，创建链接、细胞等等
        
        """
        final_candidates = []
        
        # 阶段1：处理主要目标
        for reward_cat in ["positive", "negative", "pending"]:
            # 仅处理有数据变化的类型
            if not self._has_updated_branches(cell_id, reward_cat):
                continue
                
            # 多阶段搜索（主要→次要）
            for phase in [0, 1]:
                # 步骤1：准备参数
                targets = self._collect_targets(cell_id, reward_cat)
                direction, time_ranges = self._get_time_ranges(targets = targets, phase=phase)
                reward_weights = self._select_reward_weights(reward_type = reward_cat, phase = phase)
                
                # 步骤2：注册表查询
                raw_results = self.registry.find_optimal_matches(
                    targets, 
                    time_ranges,
                    k=3,
                    reward_weights=reward_weights
                )
                
                # 步骤3：分数修饰

                for score, meta in raw_results:

                    # 步骤4：筛选候选
                    if not self._is_valid_connection(self.get_upstream_cellID(meta["conn_id"]) if direction else cell_id, cell_id if direction else self.get_upstream_cellID(meta["conn_id"]),curr_timestamp=timestamp):
                        continue
                    # 比较meta内branch的reward_type 的一致性是否高于阈值，若低于阈值则丢弃此组
                    meta_reward_type, snr = self._calculate_branches_reward_type_consistency(meta["branches"])
                    if snr < 10:
                        # 数据一致性不合格
                        continue
                    # 获取相关cell的阈值
                    up_cell = cell_id if direction else self.get_upstream_cellID(meta["conn_id"])
                    up_conn = list(meta["related_targets"])[0].get_conn_id() if direction else meta["conn_id"]
                    up_reward_type = reward_cat if direction else meta_reward_type
                    up_branch = list(meta["related_targets"])[0] if direction else list(meta["branches"])[0]


                    down_cell = self.get_upstream_cellID(meta["conn_id"]) if direction else cell_id
                    down_conn = meta["conn_id"] if direction else list(meta["related_targets"])[0].get_conn_id()
                    down_reward_type = meta_reward_type if direction else reward_cat
                    down_branch = list(meta["branches"])[0] if direction else list(meta["related_targets"])[0]


                    

                    
                    # 双向阈值校验
                    up_c,up_s,down_c, down_s = self._check_cell_pair_threshold(up_cell,up_conn, up_reward_type, down_cell,down_conn,down_reward_type, direction,score)
                    if not (up_c and down_c):
                        continue
                    
                    # 计算新建链接的信号类型
                    signal_type = self._calculate_new_conn_signal_type(up_conn_branch=up_branch, down_conn_branch=down_branch)
                    

                    # 分数调整公式
                    final_candidates.append((score, {'pairing':meta, "up_cell":(up_cell, up_s), "down_cell":(down_cell,down_s), 'direction':direction, "signal_type": signal_type}))


                    # # 找到即退出次要阶段
                    # if phase == 0: break  

        

        self.baby.add_triggered_actions(final_candidates)
        return final_candidates

    #--------------------------------------------------#
    # 辅助方法集
    #--------------------------------------------------#
    def _has_updated_branches(self, cell_id: int, reward_cat: str) -> bool:
        """检查指定类型分支是否有更新"""
        return any(
            len(conn_data[reward_cat]["branches"]) > 0
            for conn_data in self.cell_storage[cell_id].values()
        )

    def _collect_targets(self, cell_id: int, reward_cat: str) -> List["BranchInfo"]:
        """收集指定类型的所有分支"""
        return [
            b
            for conn_data in self.cell_storage[cell_id].values()
            for b in conn_data[reward_cat]["branches"].values()
        ]

    def _adjust_score(self, base_score: float, up_cell: int, down_cell: int) -> float:
        """分数调整算法"""
        # 获取连接因子总和
        total_conn = sum(
            self.cell_storage[up_cell][conn_id][rt]["connecting"]
            for conn_id in self.cell_storage[up_cell]
            for rt in ["positive", "negative"]
        )
        
        # 归一化处理
        norm_factor = total_conn / (self.thresholds[up_cell]["connecting"]["out"] * 100)
        return base_score * (1 + norm_factor)
    # 修正后逻辑
    def _get_time_ranges(self,targets: List["BranchInfo"], phase: int, min_communication_time = 3) -> tuple[bool,List[Tuple[int, int]]]:
        """
        @return
        - bool: time range 覆盖范围是否全都位于targets之后，即只关注后续事件（正向，True），只关注先前事件（反向，False） （如果出现不一致，则返回None）
        - List[Tuple[int, int]]: time range
        - min_communication_time: 一个conn完成通信的最短时间,不考虑互相之间无法建立通信的branch
        """
        
        ranges = []
        forward_dir = False
        backward_dir = False
        if phase == 0:
            for b in targets:
                if b.reward_type in ["positive"]:
                    # 正向时间窗口：仅关注后续事件 （后续止损）
                    ranges.append((b.get_timestamp() + min_communication_time, max(self.observation_window,b.get_timestamp() + min_communication_time)))
                    forward_dir = True
                elif b.reward_type in ["negative", "pending"]:
                    # 反向时间窗口：仅关注先前事件 （抑制自身/激活自身）
                    ranges.append((min(-self.observation_window,-min_communication_time) + b.get_timestamp(), -min_communication_time + b.get_timestamp()))
                    backward_dir = True
        elif phase == 1:
            for b in targets:
                if b.reward_type in ["positive"]:
                    # 反向时间窗口：仅关注先前事件 (自身巩固)
                    ranges.append((min(-self.observation_window,-min_communication_time) + b.get_timestamp(), -min_communication_time + b.get_timestamp()))
                    backward_dir = True
                elif b.reward_type in ["negative", "pending"]:

                    # 正向时间窗口：仅关注后续事件 (后续止损)
                    ranges.append((b.get_timestamp() + min_communication_time, max(b.get_timestamp() + min_communication_time, self.observation_window)))  
                    forward_dir = True

        if forward_dir and backward_dir:
            raise("Targets search direction inconsistency")
        direction = True if forward_dir else False
        return direction, ranges
    
    def _select_reward_weights(self, reward_type: str, phase: int) -> Dict:
        """phase 0: 主要目标阶段, phase 1: 次要目标阶段"""
        strategy_map = {
            "positive": {
                0: {"positive": 0.8, "mixed": 1.0, "pending": 1.2, "negative": 1.5},
                1: {"positive": 0.6, "mixed": 0.3, "pending": 0.3, "negative": 0.3}
            },
            "negative": {
                0: {"positive": 1.5, "mixed": 1.2, "pending": 1.0, "negative": 0.8},
                1: {"positive": 0.3, "mixed": 0.3, "pending": 0.3, "negative": 0.6}
            },
            "pending": {
                0: {"positive": 1.2, "mixed": 1.0, "pending": 0.6, "negative": 0.3},
                1: {"positive": 0.3, "mixed": 0.3, "pending": 0.6, "negative": 0.8}
            }
        }
        return strategy_map[reward_type][phase]
    
   
    

    def _check_cell_threshold(self,cell_id:int, conn_id, reward_type:str, action_type:str,io:str, score:float )->bool:
        """
        :param 
        - io - "in" or "out"

        - total_factor + score > threshold (for connecting)
        - total_factor > threshold (for splitting)
        """
        current_factor = self.cell_storage[cell_id][conn_id][reward_type][action_type]
        threshold = self.thresholds[cell_id][action_type][io]  

        # # 连接操作需要叠加分数，分裂操作直接比较
        # if action_type == "connecting":
        #     return (current_factor + score) > threshold
        return current_factor + score> threshold


    def _check_cell_pair_threshold(self, up_cell_id: int, up_conn_id:int, up_reward_type:str, down_cell_id: int, down_conn_id:int, down_reward_type:str, observation_direction:bool, score:float) -> Tuple[bool,bool,bool,bool]:
        """校验上下游cell的阈值组合，受分值影响
        @param observation_direction: True表示观测下游，False表示观测上游

        @return 
        - (upstream cell connecting, upstream cell splitting, downstream cell connecting, downstream cell splitting)

        """

        # 如果splitting为true，强制connecting为true

        # 上游分裂状态（始终需要独立检测）
        up_splitting = self._check_cell_threshold(up_cell_id,up_conn_id, up_reward_type, "splitting","out", 0)
        
        # 上游连接状态（观测下游时默认通过）
        up_connecting = up_splitting or (True if observation_direction else self._check_cell_threshold(up_cell_id,up_conn_id, up_reward_type,"connecting","out", score))  
        

        # 下游分裂状态（始终需要独立检测）
        down_splitting = self._check_cell_threshold(down_cell_id,down_conn_id, down_reward_type, "splitting","in", 0)

        # 下游连接状态（观测上游时默认通过）
        down_connecting = down_splitting or (True if not observation_direction else self._check_cell_threshold(down_cell_id, down_conn_id, down_reward_type,"connecting","in", score))
        

        return (up_connecting, up_splitting, down_connecting, down_splitting)
        
    def _is_valid_connection(self, up_cell_id, down_cell_id, curr_timestamp ) -> bool:

        if not all([up_cell_id,down_cell_id]):
            return False
        # 检查冷却
        if (up_cell_id, down_cell_id) in self.cooldown_map:
            if curr_timestamp < self.cooldown_map[(up_cell_id, down_cell_id)]:
                return False
            
            # 检查现有连接
            if self.baby.cells[up_cell_id].has_out_neighbor(down_cell_id):
                return False
            
            if self.baby.cells[up_cell_id].isActuator == True:
                return False
            if self.baby.cells[down_cell_id].isReceptor == True:
                return False
        return True


    def _calculate_branches_reward_type_consistency(self, branches: list["BranchInfo"]) -> tuple[str,float]:
        """
        基于向量空间正交性与信号功率的一致性检测函数
        编码方案：
        positive: [1, 0, 0]    negative: [-1, 0, 0]
        mixed:    [0, 1, 0]    unknown: [0, 0, 1]
        量化标准：
        SNR >= 10dB 且纯度>90% 为有效信号
        """
        # 正交编码字典
        vector_encoding = self.vector_encoding
        
        # 信号检测准备
        type_counter = Counter()
        signal_vectors = []
        
        # 第一阶段：基础信号收集
        for b in branches:
            vec = vector_encoding[b.reward_type]
            signal_vectors.append(vec)
            type_counter[b.reward_type] += 1
        
        # # 互斥性强制检测
        # if type_counter['positive'] > 0 and type_counter['negative'] > 0:
        #     return ('mixed', -np.inf)  # 互斥类型共存直接返回最低SNR
        
        # 计算信号空间均值
        mean_vector = np.mean(signal_vectors, axis=0)
        
        # 空间最近邻分类
        type_distances = {
            t: np.linalg.norm(mean_vector - vector_encoding[t])
            for t in vector_encoding
        }
        dominant_type = min(type_distances, key=type_distances.get)
        
        # 信号功率计算
        signal_power = np.sum([np.dot(v, vector_encoding[dominant_type])**2 
                            for v in signal_vectors])
        noise_power = np.sum([np.linalg.norm(v - vector_encoding[dominant_type])**2
                            for v in signal_vectors])
        
        # SNR计算（分贝单位）
        snr_db = 10 * np.log10(signal_power / (noise_power + 1e-9))
        
        # 纯度验证（>90%）
        purity = type_counter[dominant_type] / len(branches)
        if snr_db >= 10 and purity >= 0.9:
            return (dominant_type, snr_db)
        else:
            return ('pending' if purity < 0.5 else 'mixed', snr_db)

        """
        使用说明：
        1. 互斥规则：当positive和negative共存时直接返回mixed类型与-∞ SNR
        2. 信号质量：SNR>=10dB 且纯度>=90%时返回有效类型，否则根据纯度降级
        3. 向量空间：
        - 正负类型在x轴互为镜像
        - mixed占据y轴正交方向
        - unknown占据z轴正交方向
        4. 物理意义：
        - SNR>10dB 相当于信号功率是噪声功率的10倍
        - 纯度阈值确保至少90%的同类型数据
        """