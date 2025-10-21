from typing import TYPE_CHECKING

import numpy as np
from collections import defaultdict, deque
from StepperIterator import *
import networkx as nx
import copy
import bisect
import math



from ConstantEnums import *
from dataclasses import dataclass

def get_signal_strength(event:dict, timestamp, signal_type = Signal_E.E.value )->float:
    """
    获得事件在特定时间戳的信号强度
    - event: 事件字典，记录了一个信号的起止以及变化规律
    - timestamp: 时间戳
    - signal_type: 信号类型，如果事件信号类型与signal_type一致，直接返回信号强度，若互相抑制，则取反后返回
    返回值：信号强度

    - 获取步骤
    1. 确定timestamp位于事件的阶段2的范围内
    2. 若信号在对应阶段为线性变化，则根据两点式确定信号强度，若信号强度为负值，则说明已经停止，返回0
    3. 若信号为正值则根据signal_type确定最终返回值

    """

    # Step 1: Check if timestamp is within the range of the event (X0, Y0), (X1, Y1)
    if event.get(Event.Down_End_Attenuation.value.item.value,{}).get(Event.Down_End_Attenuation.value.X0.value, 0) > timestamp:
        return 0
    # Step 2: Calculate the signal strength based on the linear equation
    X0 = event.get(Event.Down_End_Attenuation.value.item.value,{}).get(Event.Down_End_Attenuation.value.X0.value, None)
    X1 = event.get(Event.Down_End_Attenuation.value.item.value,{}).get(Event.Down_End_Attenuation.value.X1.value, None)
    Y0 = event.get(Event.Down_End_Attenuation.value.item.value,{}).get(Event.Down_End_Attenuation.value.Y0.value, None)
    Y1 = event.get(Event.Down_End_Attenuation.value.item.value,{}).get(Event.Down_End_Attenuation.value.Y1.value, None)
    

    if timestamp == X0:
        return Y0
    elif timestamp == X1:
        return Y1
    
    slope = (Y1 - Y0) / (X1 - X0)
    if X1<X0:
        raise ValueError("Invalid event: X1 < X0")
    intercept = Y0 - slope * X0
    signal_strength = slope * timestamp + intercept
    # Step 3: Check if the signal is positive and if it matches the signal type
    if signal_strength >= 0 and signal_type == Signal_E.E.value:
        return signal_strength
    elif signal_strength < 0 and signal_type == Signal_F.F.value:
        return -signal_strength
    else:
        return 0
    

    




class BranchInfo:
    def __init__(self,key:tuple[int,int] = None, links = [], match_score = 0.0, reward = 0.0, penalty = 0.0, fingerprints = [], param_group = None, **kwargs):
        self.match_score = match_score
        self.reward = reward
        self.penalty = penalty
        self.links = links
        self.fingerprints:list[set] = fingerprints # 拓扑指纹，当前窗口对应的link只统计输入端口，过去窗口对应的link统计输入与输出端口,且各个窗口的fingerprint为两跳内的完整结构
        self.reward_type = RewardType.PENDING.value # 默认情况下结果未知

        self.parent: "ParameterGroup" = param_group # 此分支所属的父参数组的引用, 由参数组添加
        self.loss = 0 # 损失，能量。 奖励为负，惩罚为正

        self.fingerprint_similarities:dict[int:float] = {} # group_id - similarity pairs
        self.loss_snrs:dict[int:float] = {} # group_id - loss snr
        
        # self.loss_allocation:dict[int:dict] = {
        #     # id:{
        #     #   source_supply:float
        #     #   source_entropy: float
        #     #   sink_cap:float
        #     #   sink_entropy: float
        #     #   max_sink_cap:float
        #     #   kp:float
        #     #   ki:float 
        #     #   integral_error:float  
        #     #    
        #     #
        #     #   
        #     # ... other args
        #     #}


        # } 



        self.loss_caps = {} # group_id - sink caps
        self.loss_flow_controller = {}
        self.source_supply:float = 0
        self.source_entropy: float = 0  # 单位loss的熵
        self.sink_cap:float = 0
        self.max_sink_cap = 0
        self.sink_entropy: float = 0
        self.loss_flow_buffer:dict[tuple[int,int]:dict] = {} # 储存着来自不同link_id的delta_loss 以及对应的entropy（平均混乱度）
        self.outflow:float = 0 # 上一时刻流出量 
        self.loss_potential_diff_deadzone = 0.1 # TODO: 选定合适值
        self.loss_flow_controllers = {}
        self.kp = 1
        self.ki = 0.2
        self.integral_max = 10
        self.steady_state_steps = 0
        self.quiescence_threshold = 20
        self.frozen_loss_potential = 0   # 当使用loss更新paramsgroup的参数时，loss会被冻结，但pential会参与后续的计算影响loss flow

        self.connecting_factor = 0
        self.connecting_entropy_threshold = 5
        self.connecting_factor_conversion_rate = 0.05
        self.splitting_factor = 0
        self.splitting_entropy_threshold = 8 # loss开始转化为splitting factor的阈值
        self.splitting_factor_conversion_rate = 0.05 # 超出阈值部分的loss对factor的转化率






        self.consumer:set[int] = set() # 已经使用此分支的loss更新参数的group_id
        self.consumption_progress = 0.0 # 此branch的inner_loss已经被消耗的比例，[0~1]

        self.gradient:dict = {}

        self.trigger_activation_strengths:dict[int,float] = {} # 触发此信号的信号的 conn_id - w*signal_strength 如果此dict的item的length小于link的trigger event，则说明存在相邻的未知节点，不可进行 loss 转移
        self.sequential_activation_strengths:dict[int,float] = {} # 此信号触发的后续信号的 conn_id - w*signal_strength 如果此dict的item的length小于link的sequential event，则说明存在未知节点，不可进行 loss 转移

        self.outdated = False  # this branch shouldn't work anymore
        
        
        self.lr = 0.1




        self._key = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)
        if key:
            self._key = key
        else:
            self._key = (self.get_conn_id(),self.get_timestamp())

        if self._key[0] == None or self._key[0] < 0 or self._key[1] == None or self._key[1] < 0:
            raise ValueError("Invalid BranchInfo Hash!!")
        else:
            self._hashValue = hash(self._key)
        
    def get_branch_id(self)->tuple[int,int]:
        """
        同时，branch的id也是links[0]的唯一标识，两者必须一致。如果后续link无法触发，则此branch也会被遗弃
        """
        return copy.copy(self._key)
    def __hash__(self) -> int:
        return self._hashValue
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BranchInfo):
            return False
        return self._hashValue == other._hashValue

    def get_timestamp(self)->float:
        """
        获取branch被触发的时刻
        """
        if self._key:
            return self._key[1]
        else:
            return self.links[0][Link.EVENT.value][Event.START_TIMESTAMP.value]
    

    def get_conn_id(self):
        if self._key:
            return self._key[0]
        else:
            return self.links[0][Link.EVENT.value][Event.CONN_ID.value]
    
    def get_effective_timestamp(self):
        """
        获取生效时刻，在被触发时刻之后
        """
        return self.links[0][Link.EVENT.value][Event.Down_End_Attenuation.value.item.value][Event.Down_End_Attenuation.value.X0.value]
    
    def _update_param_group(self):
        """
        冻结loss_potential,并更新param group参数
        """
        self.frozen_loss_potential += self.source_supply + (self.max_sink_cap - self.sink_cap)
        update_method = self.parent.__getattribute__("_gradient_update")
        update_method(self.get_branch_id())

    def is_active(self) -> bool:
        """根据是否存在流入流出判断是否处于活跃的loss transfer状态"""
        return self.loss_flow_buffer or self.outflow


    # step 1 
    def apply_loss_flow(self, **kwargs) -> bool:
        """更新动态，并应用计算好的流量到目标sink（或添加到source）"""

        if kwargs.get("group_id",0):
            if kwargs["group_id"] !=self.parent.group_id:
                # 当前branch被更改了归属
                return False
            
        self.outflow = 0
        if not self.loss_flow_buffer:
            if self.steady_state_steps < self.quiescence_threshold:
                self.steady_state_steps += 1
                return True
            else:
                self._compute_action_factors()
                self._update_param_group()
                return False                
        self.steady_state_steps = 0
        
        total_supply = self.source_supply
        total_entropy = self.source_entropy * self.source_supply
        for sink_link_id, flow_block in self.loss_flow_buffer.items():

            total_supply += flow_block["loss"]
            total_entropy += abs(flow_block["loss"]* flow_block["entropy"])
                
        self.source_supply = total_supply
        self.source_entropy = total_entropy/ total_supply if total_supply else 0

        absorb = min(self.max_sink_cap-self.sink_cap, total_supply)
        if absorb:
            total_sink_cap = self.sink_cap + absorb
            total_sink_entropy = self.sink_entropy* self.sink_cap + absorb * self.source_entropy

            self.sink_entropy = total_sink_entropy/total_sink_cap
            self.sink_cap += absorb
            self.source_supply -=absorb

        self.loss_flow_buffer.clear()

        return True

    def _compute_action_factors(self):
        """
        根据sink_entropy和source_entropy分别计算connecting_factor和splitting_factor的增量，
        使用双曲正切函数确保转化率最高不超过0.5
        """
        # 连接因子计算（源熵部分）
        if self.source_supply > 0:
            # 有效源供给量上限为max_sink_cap的两倍
            effective_supply = min(self.source_supply, 2 * self.max_sink_cap)
            
            # 计算超出阈值程度（1.0为阈值）
            entropy_excess = max(0, (self.source_entropy - self.connecting_entropy_threshold) / 10.0)
            
            # 使用tanh函数限制转换率在0-0.5之间
            conn_ratio = 0.5 * np.tanh(entropy_excess * self.connecting_factor_conversion_rate)
            
            # 连接因子增量
            connecting_delta = effective_supply * conn_ratio
            self.connecting_factor += connecting_delta
            
        # 分裂因子计算（宿熵部分）
        if self.sink_cap > 0:
            # 计算超出阈值程度（1.0为阈值）
            entropy_excess = max(0, (self.sink_entropy - self.splitting_entropy_threshold) / 10.0)
            
            # 使用tanh函数限制转换率在0-0.5之间
            split_ratio = 0.5 * np.tanh(entropy_excess * self.splitting_factor_conversion_rate)
            
            # 分裂因子增量
            splitting_delta = self.sink_cap * split_ratio
            self.splitting_factor += splitting_delta

    # step 2
    def compute_loss_flow(self,**kwargs) -> bool:
        """计算当前branch到所有相邻sink的流量分配"""

        # 验证一致性：
        if kwargs.get("group_id",0):
            if kwargs["group_id"] !=self.parent.group_id:
                # 当前branch被更改了归属
                return False

        if self.source_supply == 0 or self.source_supply + self.frozen_loss_potential <=0:
            return True

        link:dict = self.links[0]
        trigger_events = link[Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value]
        triggers:list[BranchInfo] = [event[Event.LINK.value][Link.Condition.value.BRANCH.value] for event in trigger_events ]
        sequential_events = link[Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value]
        sequences:list[BranchInfo] = [event[Event.LINK.value][Link.Condition.value.BRANCH.value] for event in sequential_events ]

        neighbors = triggers + sequences
        if not neighbors:
            return True
        
        if any(not branch.gradient for branch in neighbors):
            return True

        if self.trigger_activation_strengths.__len__() <  triggers.__len__():
            self.trigger_activation_strengths = link[Link.Condition.value.ACTIVATION_STRENGTHS.value]            
        if self.sequential_activation_strengths.__len__() <  sequences.__len__():
            sequential_links = [branch.links[0] for branch in sequences]
            self.sequential_activation_strengths = [link[Link.Condition.value.ACTIVATION_STRENGTHS.value][self.get_branch_id()] for link in sequential_links]          
        
        if self.sequential_activation_strengths.__len__() + self.sequential_activation_strengths.__len__() < triggers.__len__() + sequences.__len__():
            raise("Missing activation info!")

        total_potential = 0
        sink_num = 0
        for sink_branch in neighbors:
            potential_diff = self.frozen_loss_potential + self.source_supply + (self.max_sink_cap - self.sink_cap) - (sink_branch.source_supply + sink_branch.frozen_loss_potential)
            if potential_diff > self.loss_potential_diff_deadzone:
                total_potential += (sink_branch.source_supply + sink_branch.frozen_loss_potential)
                sink_num +=1
        if not sink_num:
            return True
        target_potential = (total_potential + self.source_supply + self.frozen_loss_potential)/(sink_num + 1)
        
        if target_potential < self.frozen_loss_potential:
            return True 
        

        for sink_branch in neighbors:

            sink_branch_id = sink_branch.get_branch_id()
            sink_potential = -(sink_branch.max_sink_cap - sink_branch.sink_cap) + sink_branch.source_supply + sink_branch.frozen_loss_potential
            # 计算势能差 = 当前source供给 + 宿的剩余容量 - 宿的自身供给
            potential_diff = target_potential - sink_potential
            
            # 忽略死区内的势差，以及
            if potential_diff < self.loss_potential_diff_deadzone:
                continue
                
            # 从控制器获取积分状态（如果不存在则初始化）
            controller = self.loss_flow_controllers.get(
                sink_branch_id,
                {"error_integral": 0.0, "last_flow": 0.0, "expectation":-((sink_branch.max_sink_cap - sink_branch.sink_cap) - sink_branch.source_supply)}
            )

            
            # 比例项：势差正相关，与宿branch的activation强度正相关
            total_activation_strength = (sum(self.trigger_activation_strengths.values()) + sum(self.sequential_activation_strengths.values()))
            p_term = self.kp * potential_diff * (self.trigger_activation_strengths[sink_branch_id] if sink_branch_id in self.trigger_activation_strengths else self.sequential_activation_strengths[sink_branch_id] )/total_activation_strength
            
            # 积分项：使用累积误差（限制在合理范围）
            controller["error_integral"] += (np.sign(controller["error_integral"])*min(abs(controller["error_integral"]), abs(controller["expectation"] - sink_potential)) + (controller["expectation"] - sink_potential))
            controller["error_integral"] = np.clip(
                controller["error_integral"],
                -self.integral_max,
                self.integral_max
            )
            i_term = self.ki * controller["error_integral"]
            
            # 总流量 = 比例项 + 积分项
            proposed_flow = p_term + i_term
            
            # 流量边界约束（不可超过源供给/直接的势差）
            max_flow = min(
                self.source_supply, 
                proposed_flow,
                self.source_supply + self.frozen_loss_potential - sink_potential
            )
            clamped_flow = max(0.0, min(proposed_flow, max_flow))
            
            # 更新期望与流量
            controller["expectation"] = sink_potential + clamped_flow
            controller["last_flow"] = clamped_flow
            
            # 更新自身的loss
            self.source_supply -= clamped_flow

            # 更新自身活跃
            self.outflow += clamped_flow

            # 推送当前loss流量到sink branch的buffer
            sink_branch.loss_flow_buffer[self.get_branch_id()]["loss"] = clamped_flow
            sink_branch.loss_flow_buffer[self.get_branch_id()]["entropy"] = self.source_entropy + 1 + 1 -  (self.trigger_activation_strengths[sink_branch_id] if sink_branch_id in self.trigger_activation_strengths else self.sequential_activation_strengths[sink_branch_id] )/total_activation_strength  # 单位熵 在原来的基础上，增加距离因素（转移次数越多的loss越混乱），以及来源占比因素（越次要的激活所转移的loss越混乱）
        return True

    def _update_flow_config(self):
        """
        更新流量配置：基于指纹相似度和损失信噪比计算源供给和最大宿容量
        """
        group_id = self.parent.group_id
        
        # 验证所需数据存在
        if group_id not in self.fingerprint_similarities or group_id not in self.loss_snrs:
            return
        
        # 获取相似度和信噪比
        sim = self.fingerprint_similarities[group_id]
        snr = self.loss_snrs[group_id]
        
        # 计算可用loss比例
        avail_ratio = max(0, sim - self.consumption_progress)
        if avail_ratio <= 1e-5:
            return
        
        # 计算供给量
        supply = self.loss * avail_ratio
        
        # 根据信噪比和损失方向计算缩放因子
        scale = self._calc_scale_factor(snr, self.loss)
        
        # 更新配置 （这里采用+=是为了应对突然间的归属转移）
        self.source_supply += supply
        self.max_sink_cap += supply * scale
        
        # 记录消耗进度
        self.consumption_progress = sim

    def _calc_scale_factor(self, snr, loss):
        """
        细化的信噪比分段函数计算缩放因子
        """
        # 惩罚情况（正损失）
        if loss >= 0:
            if snr > 20:      # 极高稳定性 + 惩罚
                return 0.05   # 极低容量（问题很可能不在本组）
            elif snr > 15:    # 高稳定性 + 惩罚
                return 0.1
            elif snr > 10:    # 中高稳定性 + 惩罚
                return 0.2
            elif snr > 5:     # 中等稳定性 + 惩罚
                return 0.5
            elif snr > 2:     # 低稳定性 + 惩罚
                return 2.0
            else:             # 极低稳定性 + 惩罚
                return 5.0   # 极高容量（问题很可能在本组）
        
        # 奖励情况（负损失）
        else:
            if snr > 20:      # 极高稳定性 + 奖励
                return 0.1    # 低容量（已充分学习）
            elif snr > 15:    # 高稳定性 + 奖励
                return 0.2
            elif snr > 10:    # 中高稳定性 + 奖励
                return 0.4
            elif snr > 5:     # 中等稳定性 + 奖励
                return 0.8
            elif snr > 2:     # 低稳定性 + 奖励
                return 1.0
            else:             # 极低稳定性 + 奖励
                return 2.0    # 中等容量（学习不确定）


    def switch_reward_type(self):
        """
        更改reward,会导致重新计算梯度，重置自身的consumption_progress、factor以及其他必要信息。同时需要进行回滚操作，方便起见，采用替代方案，将loss以及原先的factor无条件累加到导致误判的sequences branch中（不再撤回已有的梯度更新）。
        """



class ParameterGroup:
    def __init__(self, group_id, parent: "PropagationManager" = None, topology:set = {}):
        """
        __init__ constructor for ParameterGroup class

        Parameters
        ----------
        group_id : int
            the unique id of the parameter group

        Attributes
        ----------
        group_id : int
            the unique id of the parameter group
        windows : dict[int, dict]
            a dictionary mapping window_id to window-specific parameters
        branches : dict[int, BranchInfo]
            a dictionary mapping branch_id to branch-specific information
        _branch_id_counter : int
            a counter for generating new branch_id
        """
        
        self.group_id = group_id
        self.windows = {
            # 0: {
            #     ParamWindow.WEIGHTS.value: {},
            #     ParamWindow.BIAS.value: {},
            #     Topology.item.value: {
            #         Topology.WEIGHT.value: 1.0, # 当前窗口的此拓扑节点在整体拓扑匹配中的权重
            #         Topology.INPUTS.value: {},  # 当前窗口激活的输入端口结构
            #         Topology.OUTPUTS.value: {}  # 当前窗口未激活的输出端口，空字典
            #     },
            #     ParamWindow.FINGERPRINT.value: {} # 当前窗口的拓扑指纹
                
                
            # },
            # 1: {
            #     ParamWindow.WEIGHTS.value: {},
            #     ParamWindow.BIAS.value: {},
            #     Topology.item.value: {
            #         Topology.WEIGHT.value: 1.0, 
            #         Topology.INPUTS.value: {},  
            #         Topology.OUTPUTS.value: {}  # 历史窗口激活的输出端口
            #     },
            #     ParamWindow.FINGERPRINT.value: {}
            # },
            # # ... 其他历史窗口
        }
        self.branches: dict[tuple[int,int], BranchInfo] = {
            # 用于参数组分裂。分裂指的是，当原本的Link的拓扑结构只能近似匹配此参数组的拓扑结构时（也就是说，拓扑匹配程度不为1），如果发生了奖励或者惩罚，则会在此参数组基础上，分裂出一个新的参数组。其Topology将采用Link的拓扑结构。
            # 例如，当一个参数组的输入端口为（A，B）->C，而Link的输入端口为A->B，且此时发生了多次的奖励或者惩罚，则分裂的参数组则以B->C为新的拓扑。
            # 内部储存着多个dict:
            # {
                # 所以branch储存的内容包括：
                # 1. Links的原始拓扑[数组存储]
                # 2. Links与参数组的匹配程度: 归一化float
                # 3. 损失与奖励    
            # }

        }  # 存储分支信息，在特定情况下将参数组进行分裂，生成新的参数组

        self.positive_reward_branches:set[tuple[int,int]] = set() # 回报类型为POSITIVE的branch的id的集合
        self.negative_reward_branches:set[tuple[int,int]] = set() # 回报类型为NEGATIVE的branch的id的集合
        self.pending_reward_branches:set[tuple[int,int]] = set() # 回报类型为PENDING的branch的id的集合
        self.mixed_reward_branches:set[tuple[int,int]] = set() # 回报类型为MIXED的branch的id的集合

        self.consumed_branches:set[tuple[int,int]] = set() # 已经被此参数组消耗以进行参数更新的branch的id的集合，任何branch的inner_loss只能更新一个参数组最多一次

        self.parent = parent
        self.topology = topology # 所在节点（Cell）的一跳输入输出拓扑结构。

    

        


    def _get_link_fingerprint(self, link:dict, current_window: bool = False, max_hop:int = 2):
        """
        获取单个link的拓扑指纹
        * Link 中有Trigger和Sequence两个字段，你可以从Trigger和Sequence中获取代表链接的标识符conn_id。conn_id的集合就是拓扑指纹。
        * Trigger字段是的触发端口（代表输入），Sequence字段是当前Link的后继端口（代表输出），你需要递归获取Link进行处理
        * 在递归处理时，如果存在上游/下游端口全部无法在target_fingerprint中找到，则停止递归后续节点。
        - 如果current_window为True，则只统计输入端口（input），否则统计输入端口和输出端口
        - 最多只统计两条跳范围的拓扑结构（输入的输入，输出的输出）
        """
        fingerprint = set()  # 使用集合避免重复
        visited: set[int] = set()

        # 初始化队列：(conn_id, hop_count, direction)
        input_queue = deque()
        output_queue = deque()

        # 第0跳初始化
        if current_window:
            # 仅处理输入端口（current_window模式）
            input_triggers = link.get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])
            input_queue.extend([(event, 0) for event in input_triggers])
        else:
            # 处理完整拓扑（历史模式）
            input_triggers = link.get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])
            output_sequences = link.get(Link.Sequence.value.item.value, {}).get(Link.Sequence.value.EVENTS.value, [])
            
            input_queue.extend([(event, 0) for event in input_triggers])
            output_queue.extend([(event, 0) for event in output_sequences])

        # 递归处理输入端口队列
        while input_queue:
            event, hop = input_queue.popleft()
            conn_id = event[Event.CONN_ID.value]
            if conn_id in visited or hop > max_hop:
                continue
            visited.add(conn_id)
            fingerprint.add(conn_id)

            # 递归获取上游端口（仅当允许递归时）
            if hop < max_hop:
                # 获取当前端口的输入源（模拟获取上游连接）
                upstream_events = event[Event.LINK.value].get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])
                
                # 过滤无效端口（必须存在于当前fingerprint中）
                valid_upstreams = [
                    e for e in upstream_events 
                    if e[Event.CONN_ID.value] in fingerprint or e[Event.CONN_ID.value] in visited
                ]
                
                # 递归加入队列（保持方向一致性）
                input_queue.extendleft([
                    (e, hop + 1) 
                    for e in valid_upstreams 
                    if e[Event.CONN_ID.value] not in visited
                ][::-1])  # 保持处理顺序

        # 递归处理输出端口队列（仅在非current_window模式）
        if not current_window:
            while output_queue:
                event, hop = output_queue.popleft()
                conn_id = event[Event.CONN_ID.value]
                if conn_id in visited or hop > max_hop:
                    continue
                visited.add(conn_id)
                fingerprint.add(conn_id)

                # 递归获取下游端口
                if hop < max_hop:
                    downstream_ports = event[Event.LINK.value].get(Link.Sequence.value.item.value, {}).get(Link.Sequence.value.EVENTS.value, [])
                    
                    valid_downstreams = [
                        e for e in downstream_ports 
                        if e[Event.CONN_ID.value] in fingerprint or e[Event.CONN_ID.value] in visited
                    ]
                    
                    output_queue.extend([
                        (e, hop + 1) 
                        for e in valid_downstreams 
                        if e[Event.CONN_ID.value] not in visited
                    ])

        return fingerprint
    

        
    def _get_links_fingerprint(self, links:list, max_hop = 2):
        """
        获取多个link的拓扑指纹,只有第一个link被视作与当前窗口对应，其他link被视作与历史窗口对应

        """
            
        fingerprints:set = {}

        # 遍历链接，为每个链接生成指纹
        for i, link in enumerate(links):
            if i >= 5 or i > len(self.windows):
                # 限制最多只记录windows+1个links的指纹（最多5个）
                break
            fingerprints[i] = self._get_link_fingerprint(link, current_window = (i == 0), max_hop = max_hop)

        return fingerprints
        



    def add_branch(self, links, match_score = 0, reward = 0, penalty = 0):
        """
        添加新的分支,并补充其指纹(两跳)
        - 一个BranchInfo实例只能被一个ParameterGroup实例所拥有
        - TODO: 移除陈年旧事，如果在添加时存在过于老旧的未处理branch，则将其直接移除。此逻辑后续需要完善
        """
        branchInfo = BranchInfo(links = links, match_score = match_score, reward = reward, penalty = penalty, fingerprints = self._get_links_fingerprint(links), param_group = self)

    
        self.branches[branchInfo.get_branch_id()] = branchInfo

        register_branchInfo = self.parent.baby.__getattribute__("register_branchInfo")
        register_branchInfo(branchInfo)

        # TODO: 移除相较最新branch_id小20的branchs (请继续完善)
        for branch_id in list(self.branches.keys()):
            if branch_id < self._branch_id_counter - 20:
                # 从对应集合中移除id
                if branch_id in self.positive_reward_branches:
                    self.positive_reward_branches.remove(branch_id)
                if branch_id in self.negative_reward_branches:
                    self.negative_reward_branches.remove(branch_id)
                if branch_id in self.pending_reward_branches:
                    self.pending_reward_branches.remove(branch_id)
                if branch_id in self.mixed_reward_branches:
                    self.mixed_reward_branches.remove(branch_id)
                
                # 移除branch
                del self.branches[branch_id]


    def compute_diff_set(link_topology, group_fingerprint):
        """
        计算实际链接与参数组指纹的差异集合
        - 多激活端口（Extra）：链接存在但指纹不存在的端口
        - 少激活端口（Missing）：指纹存在但链接未激活的端口
        """
        diff_set = {
            'extra': set(link_topology.keys()) - set(group_fingerprint.keys()),
            'missing': set(group_fingerprint.keys()) - set(link_topology.keys())
        }
        return diff_set
    
    def get_branch_fingerprints_intersection(self, reward_type)->list[set]:
        """
        获取当前参数组指定奖励类型的分支的指纹交集
        - 1 根据reward_type选择指定分支id的集合
        - 2 根据id对应的指纹，计算交集
        - 3 返回交集列表
        """
        finger_prints_intersection:list[set] = []
        branch_id_set = set()
        if reward_type == RewardType.POSITIVE.value:
            branch_id_set = self.positive_reward_branches
        elif reward_type == RewardType.NEGATIVE.value:
            branch_id_set = self.negative_reward_branches
        elif reward_type == RewardType.PENDING.value:
            branch_id_set = self.pending_reward_branches
        elif reward_type == RewardType.MIXED.value:
            branch_id_set = self.mixed_reward_branches

        for i,branch_id in enumerate(branch_id_set):
            fp = self.branches[branch_id]
            if i == 0:
                finger_prints_intersection = copy.deepcopy(fp)
            else:
                for j in range(0, min(len(finger_prints_intersection), len(fp))):
                    finger_prints_intersection[j] = finger_prints_intersection[j].intersection(fp[j])
        
        return finger_prints_intersection


    def greedy_select(sets_dict: dict[int, set], k=5) -> tuple[set[int], set, int]:
        """
        从集合字典中贪心选择k个集合，返回选中的ID集合、交集元素和大小
        :param sets_dict: {集合ID: 元素集合} 的字典
        :param k: 需要选择的集合数量
        :return: (选中ID集合, 交集元素集合, 交集大小)
        """
        if len(sets_dict) < k:
            return set(), set(), 0

        remaining = list(sets_dict.items())  # 保存(ID, set)的列表
        selected_ids = set()
        current_intersection = None

        for _ in range(k):
            best_id = None
            best_set = None
            best_size = -1

            # 遍历所有剩余集合
            for set_id, elements in remaining:
                if current_intersection is None:
                    temp_inter = elements
                else:
                    temp_inter = current_intersection & elements

                # 选择能带来最大交集的集合
                if len(temp_inter) > best_size:
                    best_id = set_id
                    best_set = elements
                    best_size = len(temp_inter)

            if best_id is None:
                break  # 没有可选集合

            # 更新状态
            selected_ids.add(best_id)
            remaining = [(sid, s) for sid, s in remaining if sid != best_id]
            current_intersection = best_set if current_intersection is None else current_intersection & best_set

        final_inter = current_intersection if current_intersection is not None else set()
        return selected_ids, final_inter, len(final_inter)

    def find_max_intersection(sets_dict: dict[int, set], top_elements:int =1000, k:int = 5) -> tuple[set, set[int]]:
        """
        查找最大交集的优化算法
        - param sets_dict: {集合ID: 元素集合} 的字典
        - param top_elements: 考虑的高频元素数量
        - param k: 需要选择的集合数量
        - return: (最大交集元素集合, 构成该交集的集合ID集合)
        """
        # 构建倒排索引：元素 -> 包含它的集合ID列表
        element_to_ids = defaultdict(list)
        for set_id, elements in sets_dict.items():
            for elem in elements:
                element_to_ids[elem].append(set_id)

        # 按元素出现频率排序
        sorted_elements = sorted(element_to_ids.keys(),
                            key=lambda x: len(element_to_ids[x]),
                            reverse=True)

        max_inter = set()
        best_ids = set()

        # 只检查高频元素对应的集合组合
        for elem in sorted_elements[:top_elements]:
            candidate_ids = element_to_ids[elem]
            if len(candidate_ids) < k:
                continue

            # 提取候选集合的子字典
            candidate_dict = {cid: sets_dict[cid] for cid in candidate_ids}

            # 执行贪心选择
            selected_ids, inter, size = ParameterGroup.greedy_select(candidate_dict, k)

            # 更新最优解
            if size > len(max_inter):
                max_inter = inter
                best_ids = selected_ids

        # 兜底策略：如果所有高频元素都无解，使用全局贪心
        if len(max_inter) == 0:
            best_ids, max_inter, _ = ParameterGroup.greedy_select(sets_dict, k)

        for i, elem in enumerate(max_inter):
            if i == 0:
                best_ids = set(element_to_ids[elem])
            else:
                best_ids = best_ids & set(element_to_ids[elem])

        return max_inter, best_ids

    def _calculate_set_similarity_difference(self, set1:set, set2:set, set3:set[int], difference_penalty = 2)->float:
        """
        计算set3分别相对set1和set2的相似度，返回是否set3与set1的相似度和与set2的相似度的差值：
        """
        # 1 计算set3和set1的交集
        # 2 计算set3和set1的异或集
        # 3 计算set3和set1的相似度得分 （交集 - 异或集 * difference_penalty）
        # 4 同样，计算set3和set2的相似度得分 
        # 5 比较相似度得分，返回是否set3在set1和set2中更接近set1
        if len(set3) == 0:
            return False
        intersection_set1 = set3.intersection(set1)
        intersection_set2 = set3.intersection(set2)
        set1_xor_set3 = set3.symmetric_difference(set1)
        set2_xor_set3 = set3.symmetric_difference(set2)
        set1_score = len(intersection_set1) - len(set1_xor_set3) * difference_penalty
        set2_score = len(intersection_set2) - len(set2_xor_set3) * difference_penalty
        return set1_score - set2_score

    def _calculate_fingerprint_similarity_difference(self, fingerprint1:list[set], fingerprint2:list[set],fingerprint3:list[set])->bool:
        """
        计算fingerprint3相对1,2两个指纹的相似度，返回相似度fingerprint3是否与1更相似
        """

        min_len = min(len(fingerprint1),len(fingerprint2),len(fingerprint3))

        for i in range(0, min_len):
            similarity_score_difference = ParameterGroup._calculate_set_similarity_difference(fingerprint1[i], fingerprint2[i], fingerprint3[i])
            if (similarity_score_difference<0):
                return False
            elif similarity_score_difference>0:
                return True
            else:
                continue

        if len(fingerprint3)>len(fingerprint2) and len(fingerprint3)>len(fingerprint1):
            if (len(fingerprint1)<len(fingerprint2)):
                for i in range(min_len, len(fingerprint1)):
                    if len(fingerprint1[i]) < len(fingerprint2[i]):
                        return True
                    elif len(fingerprint1[i]>len(fingerprint2[i])):
                        return False
                    else:
                        continue
                
                for i in range(len(fingerprint1), fingerprint2):
                    if len(fingerprint2[i])>0:
                        return True
                return False
            elif (len(fingerprint1)>len(fingerprint2)):
                for i in range(min_len, len(fingerprint2)):
                    if len(fingerprint1[i]) < len(fingerprint2[i]):
                        return False
                    elif len(fingerprint1[i]>len(fingerprint2[i])):
                        return True
                    else:
                        continue
                return False
            else:
                for i in range(min_len, len(fingerprint2)):
                    if len(fingerprint1[i]) < len(fingerprint2[i]):
                            return True
                    elif len(fingerprint1[i]>len(fingerprint2[i])):
                            return False
                return  False

        return False

    def _calculate_topology_similarity(self, topology1:set, topology2:set, hop1_weight = 0.8, hop2_weight = 0.2, difference_penalty = 2) -> float:
        """
        计算topology2相对topology1的相似度
        """
        if not topology1 and topology2:
            return -(hop1_weight+hop2_weight)
        topology1_hop1_set:set = topology1.intersection(self.topology)
        topology1_hop2_set:set = topology1.difference(self.topology)

        topology2_hop1_set:set = topology2.intersection(self.topology)
        topology2_hop2_set:set = topology2.difference(self.topology)

        hop1_intersection_set:set = topology1_hop1_set.intersection(topology2_hop1_set)
        hop1_symmetric_difference_set:set = topology1_hop1_set.symmetric_difference(topology2_hop1_set)
        
        hop2_intersection_set:set = topology1_hop2_set.intersection(topology2_hop2_set)
        hop2_symmetric_difference_set:set = topology1_hop2_set.symmetric_difference(topology2_hop2_set)

        topology_similarity = ((len(hop1_intersection_set) - difference_penalty*len(hop1_symmetric_difference_set)) * hop1_weight + (len(hop2_intersection_set) - difference_penalty*len(hop2_symmetric_difference_set)) * hop2_weight)/(len(topology1_hop1_set)*hop1_weight + len(topology1_hop2_set)* hop2_weight)
        return topology_similarity

    def _calculate_branch_fingerprint_similarity(self, branch_id, decay = 0.2)->float:
        """
        计算特定branch与自身的fingerprint的拓扑相似度
        -decay: 窗口相较前一窗口的权重比值
        """
        branch_fingerprint = self.branches[branch_id].fingerprints
        similarity = 0
        normalization = 0
        for i in range(0, max(len(self.windows),branch_fingerprint)):
            similarity += decay**i * self._calculate_topology_similarity(self.windows[i].get(ParamWindow.FINGERPRINT.value, set()) if i < len(self.windows) else set(), branch_fingerprint[i] if i < len(branch_fingerprint) else set())
            normalization += decay**i
        return max(similarity,0)/max(normalization,1)







  
        
    
    def _filter_branches_by_fingerprint(self, branches:dict[int,BranchInfo],  new_group_fingerprint:list[set], branch_ids:set[int])->set[int]:
        """
        根据new_group_fingerprint从branches中分离拓扑与之接近的branches
        - 1 若new_group的指纹规模大于当前指纹规模，则直接分离
        - 2 若new_group的指纹与当前指纹存在差集，直接分离
        - 3 若new_group的指纹为当前指纹的子集，则需要分别比较各个branch与当前指纹的相似度，其必须小于与新指纹的相似度
        - 4 若new_group的指纹与当前指纹相同，则需要比较后续窗口的指纹。若各窗口的指纹均相同，则不分裂参数组。
        """
            
        new_group_branch_ids = set()
        min_size = min(len(self.windows), len(new_group_fingerprint))
        for i in range(0, min_size):
            if new_group_fingerprint[i] == self.windows[i].get(ParamWindow.FINGERPRINT.value, set()):
                # 4 若new_group的指纹与当前指纹相同，则需要比较后续窗口的指纹。若各窗口的指纹均相同，则不分裂参数组。
                continue
            elif len(new_group_fingerprint[i]) > len(self.windows[i].get(ParamWindow.FINGERPRINT.value, set())):
                # 1 若new_group的指纹规模大于当前指纹规模，则直接分离
                return copy.deepcopy(branch_ids)
            elif len(new_group_fingerprint[i]) <= len(self.windows[i].get(ParamWindow.FINGERPRINT.value, set())):
                # 2 若new_group的指纹与当前指纹存在差集，直接分离
                if len(new_group_fingerprint[i].difference(self.windows[i].get(ParamWindow.FINGERPRINT.value, set()))) != set():
                    return copy.deepcopy(branch_ids)
                else:
                    # 3 若new_group的指纹为当前指纹的真子集，则需要分别比较各个branch与当前指纹的相似度，其必须小于与新指纹的相似度
                    for branch_id in branch_ids:
                        branch:BranchInfo = branches[branch_id]
                        if ParameterGroup._calculate_fingerprint_similarity_difference(fingerprint1=new_group_fingerprint, fingerprint2=[fp for idx,window in self.windows for fp in window.get(ParamWindow.FINGERPRINT.value, set())], fingerprint3=branch.fingerprints):
                            # 若branch的指纹与new_group的指纹相同，则需要比较其相似度
                            new_group_branch_ids.add(branch_id)

                            
                        
            else:
                # 报错：未知情况
                raise ValueError("Unknown case")
        

        return new_group_branch_ids
    

    def _get_new_group_fingerprint_by_reward(self, reward_types:set, max_window_len = 5)->list[set[int]]:
        """
        尝试分裂参数组
        - 1 根据reward_type获取对应的branch_set
        - 2 若branch的分支数目大于等于5，则分析其指纹
        - 2.1 获取指纹的最大公共交集，
        - 2.2 将指纹的公共交集视作新参数组的指纹，分割branches
        - 3 返回新参数组的指纹
        """
        branch_set = set()
        if RewardType.POSITIVE.value in reward_types:
            branch_set = branch_set.union(self.positive_reward_branches)
        if RewardType.NEGATIVE.value in reward_types:
            branch_set = branch_set.union(self.negative_reward_branches)
        if RewardType.PENDING.value in reward_types:
            raise ValueError("Invalid reward type")
        if RewardType.MIXED.value in reward_types:
            branch_set = branch_set.union(self.mixed_reward_branches)
        
        new_group_fingerprint = []
        new_group_branch_ids = branch_set


        for i in range(0, max_window_len):
            new_group_fingerprint[i],new_group_branch_ids =  ParameterGroup.find_max_intersection(sets_dict = [self.branches[branch_id].fingerprints for branch_id in new_group_branch_ids], top_elements=1000, k = 5)
        return new_group_fingerprint








    def _set_reward_loss_and_split(self, branch_id, reward_type, min_split_len = 5)->tuple[list[set[int]], set[int]]:

        """
        设置分支的奖励类型
        - 1 从当前分支所属的参数组中移除此分支
        - 2 向参数组中添加此分支
        - 3 更新分支的奖励类型
        ### 返回
        - 1 若需要分裂参数组，则返回 1 新的参数组的fingerprint和 2 划归新参数组的branch_ids
        - 2 若不需要分裂参数组，则返回 1 空的fingerprint 和 2 空的branch_ids
        """
        branch = self.branches[branch_id]
        branch.reward_type = reward_type
        if reward_type == RewardType.POSITIVE.value:
            if branch_id in self.positive_reward_branches:
                return [],set()
            self.positive_reward_branches.add(branch_id)
            self.negative_reward_branches.discard(branch_id)
            self.pending_reward_branches.discard(branch_id)
            self.mixed_reward_branches.discard(branch_id)

            if len(self.positive_reward_branches) >= min_split_len:
                new_group_fingerprint = self._get_new_group_fingerprint_by_reward(reward_types={RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value})
                new_group_branches = self._filter_branches_by_fingerprint(branches = self.branches, new_group_fingerprint=new_group_fingerprint, branch_ids = set(self.branches.keys()))
                if len(new_group_branches) >= min_split_len:
                    return new_group_fingerprint, new_group_branches

        
        elif reward_type == RewardType.NEGATIVE.value:
            if branch_id in self.negative_reward_branches:
                return [],set()

            self.positive_reward_branches.discard(branch_id)
            self.negative_reward_branches.add(branch_id)
            self.pending_reward_branches.discard(branch_id)
            self.mixed_reward_branches.discard(branch_id)

            if len(self.positive_reward_branches) >= min_split_len:
                new_group_fingerprint = self._get_new_group_fingerprint_by_reward(reward_types={RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value})
                new_group_branches = self._filter_branches_by_fingerprint(branches = self.branches, new_group_fingerprint=new_group_fingerprint, branch_ids = set(self.branches.keys()))
                if len(new_group_branches) >= min_split_len:
                    return new_group_fingerprint, new_group_branches
                
        elif reward_type == RewardType.PENDING.value:
            if branch_id in self.pending_reward_branches:
                return [],set()

            self.positive_reward_branches.discard(branch_id)
            self.negative_reward_branches.discard(branch_id)
            self.pending_reward_branches.add(branch_id)
            self.mixed_reward_branches.discard(branch_id)

            if len(self.positive_reward_branches) >= min_split_len:
                new_group_fingerprint = self._get_new_group_fingerprint_by_reward(reward_types={RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value})
                new_group_branches = self._filter_branches_by_fingerprint(branches = self.branches, new_group_fingerprint=new_group_fingerprint, branch_ids = set(self.branches.keys()))
                if len(new_group_branches) >= min_split_len:
                    return new_group_fingerprint, new_group_branches
                
        elif reward_type == RewardType.MIXED.value:
            if branch_id in self.mixed_reward_branches:
                return [],set()

            self.positive_reward_branches.discard(branch_id)
            self.negative_reward_branches.discard(branch_id)
            self.pending_reward_branches.discard(branch_id)
            self.mixed_reward_branches.add(branch_id)
            
            if len(self.positive_reward_branches) >= min_split_len:
                new_group_fingerprint = self._get_new_group_fingerprint_by_reward(reward_types={RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value})
                new_group_branches = self._filter_branches_by_fingerprint(branches = self.branches, new_group_fingerprint=new_group_fingerprint, branch_ids = set(self.branches.keys()))
                if len(new_group_branches) >= min_split_len:
                    return new_group_fingerprint, new_group_branches

        
        return [],set()

    @staticmethod
    def get_relative_modulation(base_signal_type = Signal_E.E.value, modulated_signal_type = None) -> float:
        """
        返回被调制信号相对基信号的调制系数
        """
        relative_modulation_coef_map = {
            (Signal_E.E.value,Signal_E.E.value): 1,
            (Signal_E.E.value,Signal_F.F.value): 1,
            (Signal_F.F.value,Signal_E.E.value): -1,
            (Signal_F.F.value,Signal_F.F.value): -1,
        } # (base_signal, modulated_signal): relative_modulation_coef

        return relative_modulation_coef_map.get((base_signal_type, modulated_signal_type), 0.0)


    def split(self, new_group_fingerprint:list[set], new_group_branch_ids:set[int], min_branch_num = 5)->"ParameterGroup":
        """
        分裂参数组
        - 1 从当前参数组的branches中移除new_group_branch_ids中的branch，并将其移到新group中
        - 2 从当前参数组的各个branches_id_set移除所有branch_id，并移到新group中对应的sets
        """
        if len(new_group_branch_ids) < min_branch_num:
            return None
        new_group = ParameterGroup(parent = self.parent)
        new_group.windows = {}
        new_group.branches = {branch_id: self.branches[branch_id] for branch_id in new_group_branch_ids}
        new_group.topology = self.topology
        for id,branch in new_group.branches.items():
            branch.parent = new_group
            
            

            if branch.reward_type == RewardType.POSITIVE.value:
                new_group.positive_reward_branches.add(id)
                del self.positive_reward_branches[id]
            elif branch.reward_type == RewardType.NEGATIVE.value:
                new_group.negative_reward_branches.add(id)
                del self.negative_reward_branches[id]
            elif branch.reward_type == RewardType.PENDING.value:
                new_group.pending_reward_branches.add(id)
                del self.pending_reward_branches[id]
            elif branch.reward_type == RewardType.MIXED.value:
                new_group.mixed_reward_branches.add(id)
                del self.mixed_reward_branches[id]
            else:
                raise ValueError("Unknown reward type")
            del self.consumed_branches[id]



        for i in range(0, len(new_group_fingerprint)):
            new_group.windows[i][ParamWindow.FINGERPRINT.value] = new_group_fingerprint[i]
            
            if i ==0:
                # TODO: 未来再解锁覆盖面更广的参数，目前只处理一跳
                weights_dict = new_group.windows[i][ParamWindow.WEIGHTS.value] = copy.deepcopy(self.windows[i][ParamWindow.WEIGHTS.value])
                bias_dict = new_group.windows[i][ParamWindow.BIAS.value] = copy.deepcopy(self.windows[i][ParamWindow.BIAS.value])
                


                missing_keys:set = new_group_fingerprint[i].intersection(new_group.topology) - set(weights_dict.keys())
                extra_keys:set = set(weights_dict.keys()) - new_group_fingerprint[i]

                if missing_keys != set():
                    for key in missing_keys:
                        modulated_signal_type = self.parent.baby.connections[key].constraints["signal_type"]
                        modulation_coef = self.get_relative_modulation(modulated_signal_type=modulated_signal_type)
                        weights_dict[key] = 1.0 * modulation_coef
                        bias_dict[key] = -0.5

                    for key in extra_keys:
                        del weights_dict[key]
                        del bias_dict[key]

        return new_group
    

    def _control_behavior(self, branch:BranchInfo):
        """
        通过loss计算 splitting_factor 和 connecting_factor
        
        """






        
    # def _update_params(self, branch:BranchInfo):
    #     """
    #     从branch中获取所需信息，直接更新此branch以及关联branch的参数

    #   （此函数被 _gradient_update替代）

    #     对抗性计算
    #     对抗性参数更新
    #     能量约束
    #     对抗性损失传递
    #     """
    #     if branch.parent.group_id != self.group_id:
    #         return False
        
    #     pass







    def _calculate_signal_gradient(self, reward_type, k, ts, b, t0,  loss_ft)->dict:
        """
        计算各输入信号梯度
        输入信号函数模型：
        f(t) = 0 if t < ts
             = -k * (t - ts) + b if t >= ts // currently, the input is stable during the test phase, when the input amplitude changes significantly, please consider the activation strength
        其中，f(t)>=0, 代表信号强度, ts 是信号的触发时间，k 和 b 是常数，用于控制信号的峰值和衰减速度。
        需要根据误差来进行梯度更新，同时需要优化信号的积分，减少能量消耗

        当loss为penalty时(>0)，直接计算梯度
        当loss为reward时(<0)，优化 k, ts， b使得在保证(t0， f(t0))位于信号曲线内部的情况下，减弱信号的积分（信号的能量消耗）


        - 返回： k， b， ts 的梯度 (相对loss)
        """


        grads = {'dk': 0, 'dts': 0, 'db': 0}
    
        # 物理约束检查
        assert t0 >= ts, "t0必须大于等于ts"
        assert k > 0 and b > 0, "k和b必须为正数"

        # 信号强度计算
        f_t0 = -k*(t0-ts) + b
        
        # 能量积分对参数的梯度（三角形面积公式：b²/(2k)）
        d_int_dk = -(b**2) / (2 * k**2)  # 积分对k的导数
        d_int_db = b / k                 # 积分对b的导数
        delta_t = t0 - ts
        
        if reward_type == RewardType.NEGATIVE.value:
            loss_ft = abs(loss_ft)
            # 惩罚情况：直接传递积分梯度
            grads['dk'] = loss_ft * d_int_dk
            grads['db'] = loss_ft * d_int_db
            grads['dts'] = 0  # 积分与ts无关
            
            # 如果信号持续时间过短，则钝化信号，延长信号持续时间
            epsilon = 2.0
            if b/k < epsilon:
                # 通过抑制k的增大来延长信号持续时间
                grads['dk'] *= 0.3
                grads['db'] += loss_ft * 0.5 * (2.0*k - b)/k

            # 钝化信号
        elif reward_type == RewardType.POSITIVE.value:
            # 奖励情况：需要在保持触发的前提下减少能量
            epsilon = 0.1  # 阈值判定参数

            loss_ft = -abs(loss_ft)
            
            # 当t0远离ts时，尝试通过锐化信号削弱能量消耗

            # 自适应平衡系数 (delta_t越大，触发约束权重越高)
            lambda_balance = np.clip(delta_t / 5.0, 0.2, 0.8)  # 5.0为时间基准
            
            # 触发约束梯度项
            df_dk = -(t0 - ts)
            df_db = 1.0
            df_dts = k
            
            # 复合梯度 (能量优化+触发保障)
            grads['dk'] = loss_ft * (d_int_dk * (1-lambda_balance) + df_dk * lambda_balance)
            grads['db'] = loss_ft * (d_int_db * (1-lambda_balance) + df_db * lambda_balance)
            grads['dts'] = loss_ft * lambda_balance * df_dts
            
            # 自然锐化约束 (当ts接近t0时冻结ts更新)
            if delta_t < 1.0:
                grads['dts'] *= np.clip(delta_t, 0.1, 1.0)
        
        return grads









    def _calculate_activation_gradient(self, reward_type, yt0, t0, input_info_dict:dict[int,dict], p ,fs, q, fo, **kargs )->dict:
        """
        计算输入信号激活输出信号的梯度：

        y(t0) = \Sigmal {w_i * f_i(t0) + b_i } - p * fs + q * fo 
        // 后续可能的改进( - 对自身输出的怀疑 + 对其他输出的怀疑 )
        其中，t0 为y(t)首次大于0的时刻， fs为自身疲劳度，fo为平均疲劳度，p为自身疲劳导致的抑制系数，q为其他输出端口平均疲劳导致的激活系数（疲劳就是消耗的能量，是一个积分模型）

        损失计算：
            计算各个输入信号激活的能量消耗，仅考虑f(t0)至之后的积分(对每个输入信号，有t0>=ts)，需要说明，loss不是y的值，而是触发y消耗的能量！！
            根据reward_type确定loss是reward还是penalty：
                如果是reward：
                    
                    如果y(t0)远大于0（大于给定数），则无需更新梯度
                    如果y(t0)接近0（小于给定数,假如是epsilon），则需要更新梯度，使得y(t0)向着epsilon给定值移动
                    energy = \Sigma {f_i(t) dt, t>t0， f(t) >=0 }
                    loss = - (epsilon^2 + 2 * epsilon * energy^ 0.5) //将综合的输入信号衰减以等腰直角三角形建模，延长腰长epsilon
                    loss可以进一步简化为 -2 * epsilon * energy^ 0.5

                    计算loss_ft时根据各个信号的energy消耗（信号各自的积分）对于总energy的贡献来分配。因为_calculate_signal_gradient用到的loss_ft也为能量
                    

                如果是loss：
                    期望y(t0)向着 -epsilon 移动，等腰三角形边长减少y(t0)+epsilon，能量也应该对应减少:
                    energy = 1 * \Sigma {f_i(t) dt, t>t0， f(t) >=0 }
                    loss 约等于 2 * (epsilon + y(t0)) * energy^0.5 

                    计算所有梯度                    
        - 输入： input_info_dict:
            {   
                id: {
                    "w": float,
                    "b": float,
                    "k": float,
                    "ft0": float  # 直接计算三角形面积获取t >= t0 时的积分               
                }
            }

        - 返回： 各组 w, b, t0, p， q 的梯度, f(t0)的loss
            {
                conn_grads:{
                    id: {
                        int: float, # id对应的信号所消耗的能量（积分计算）
                        dw: float,
                        db: float,
                        loss_ft: float
                    }
                },

                dp: float,
                dq: float,
                dt0: float,
                t0: float,
                total_integral: float

            }
        """




        grads = {'conn_grads':{},'dp': 0, 'dq': 0, 'dt0': 0, 't0': t0}
        total_integral = 0
        epsilon_reward = 0.5
        epsilon_penalty = 0.3
        
        # 阶段1: 计算各输入信号的积分（t>=t0部分的能量消耗）
        integral_dict = {}
        for uid, info in input_info_dict.items():
            f_t0 = info['ft0']
            k = info['k']
            ts = info.get('ts', 0)
            integral = (f_t0 ** 2) / (2 * k) if k != 0 and t0 >= ts else 0.0
            integral_dict[uid] = integral
            total_integral += integral
        grads['total_integral'] = total_integral
        
        # 阶段2: 分成奖励和惩罚两种情况处理
        if reward_type == RewardType.POSITIVE.value:
            if yt0 > epsilon_reward:
                # 安全区域，不更新任何梯度
                for uid in integral_dict:
                    grads['conn_grads'][uid] = {
                        'int': integral_dict[uid],
                        'dw': 0.0,
                        'db': 0.0,
                        'loss_ft': 0.0  # 信号参数无需更新
                    }
            else:
                # 敏感区域，需要同时优化能量积分和调整激活值
                energy_loss = -2 * epsilon_reward * (total_integral ** 0.5)
                activation_loss = (epsilon_reward - yt0) ** 2  # 驱动y(t0)向上
                
                # 计算梯度分配因子
                integral_scale = energy_loss / total_integral if total_integral != 0 else 0
                activation_scale = 2 * (epsilon_reward - yt0)  # activation_loss对y的导数
                
                for uid, integral in integral_dict.items():
                    info = input_info_dict[uid]
                    f_t0 = info['ft0']
                    w = info['w']
                    b = info['b']
                    k = info['k']
                    ts = info.get('ts', 0)
                    
                    # 积分相关梯度（影响k,b,ts）
                    grad_k_integral = integral_scale * (-f_t0**2 / (2 * k**2))
                    grad_b_integral = integral_scale * (f_t0 / k)
                    # 激活相关梯度（影响w,b）
                    grad_w_activation = activation_scale * f_t0  # ∂(activation_loss)/∂w
                    grad_b_activation = activation_scale * 1.0   # ∂(activation_loss)/∂b
                    
                    grads['conn_grads'][uid] = {
                        'int': integral,
                        'dw': grad_w_activation,         # w仅受激活损失影响
                        'db': grad_b_activation + grad_b_integral,  # b受两部分影响
                        'loss_ft': grad_k_integral       # 传递到信号参数计算
                    }

        elif reward_type == RewardType.NEGATIVE.value:
            # 惩罚模式需要同时增加能量消耗和调整激活值
            energy_loss = 2 * (epsilon_penalty + yt0) * (total_integral ** 0.5)
            activation_loss = (yt0 + epsilon_penalty) ** 2  # 驱动y(t0)向下
            
            integral_scale = energy_loss / total_integral if total_integral != 0 else 0
            activation_scale = 2 * (yt0 + epsilon_penalty)  # activation_loss对y的导数
            
            for uid, integral in integral_dict.items():
                info = input_info_dict[uid]
                f_t0 = info['ft0']
                w = info['w']
                b = info['b']
                k = info['k']
                ts = info.get('ts', 0)
                
                # 积分相关梯度（增大能量）
                grad_k_integral = integral_scale * (f_t0**2 / (2 * k**2))  # 注意符号反转!
                grad_b_integral = integral_scale * (-f_t0 / k)             # 驱动b增大积分
                # 激活相关梯度（减小y(t0)）
                grad_w_activation = activation_scale * f_t0  # ∂(activation_loss)/∂w
                grad_b_activation = activation_scale * 1.0   # ∂(activation_loss)/∂b
                
                grads['conn_grads'][uid] = {
                    'int': integral,
                    'dw': grad_w_activation,         # 惩罚模式下w梯度与f(t0)成正比
                    'db': grad_b_activation + grad_b_integral,
                    'loss_ft': grad_k_integral       # 传递给信号梯度计算
                }

        # 阶段3: 全局参数梯度计算（完整实现）
        if reward_type == 'reward' and yt0 <= epsilon_reward:
            # 奖励模式敏感区域 --------------------------------------------------
            # 能量损失项导数
            dL_dintegral = (-epsilon_reward) / (total_integral**0.5) if total_integral > 0 else 0
            # 激活项导数
            dL_dy = 2 * (epsilon_reward - yt0)
            
            # 参数p的梯度（来自激活项）
            grads['dp'] = dL_dy * (-fs)  # ∂y/∂p = -fs
            
            # 参数q的梯度（来自激活项）
            grads['dq'] = dL_dy * fo     # ∂y/∂q = fo
            
            # 时间t0的梯度（积分项 + 激活项）
            dt0_grad = 0.0
            for uid, info in input_info_dict.items():
                if t0 >= info.get('ts', 0):
                    f_t0, k, w = info['f(t0)'], info['k'], info['w']
                    # 积分项梯度
                    dt0_grad += dL_dintegral * (f_t0**2 / k)
                    # 激活项梯度
                    dt0_grad += dL_dy * (-w * k)  # ∂y/∂t0 = -w*k
            grads['dt0'] = dt0_grad

        elif reward_type == 'penalty':
            # 惩罚模式 ----------------------------------------------------------
            # 能量损失项导数
            dL_dintegral = (epsilon_penalty + yt0) / (total_integral**0.5) if total_integral > 0 else 0
            # 激活项导数
            dL_dy = 2 * (yt0 + epsilon_penalty)
            
            # 参数p的梯度（符号与reward模式相反）
            grads['dp'] = dL_dy * (-fs)  # 仍保持负号，因为y定义式中是 -p*fs
            
            # 参数q的梯度 
            grads['dq'] = dL_dy * fo
            
            # 时间t0的梯度
            dt0_grad = 0.0
            for uid, info in input_info_dict.items():
                if t0 >= info.get('ts', 0):
                    f_t0, k, w = info['ft0'], info['k'], info['w']
                    # 积分项梯度（增强能量）
                    dt0_grad += dL_dintegral * (f_t0**2 / k)
                    # 激活项梯度（抑制激活）
                    dt0_grad += dL_dy * (-w * k)
            grads['dt0'] = dt0_grad

        return grads

            
            
    def _calculate_branch_grandient(self, branch_id):
        """
        计算特定分支对参数组、conn参数的影响。branch需要有明确的reward_type. 

        梯度将会储存branch中，再逐步应用梯度到父 ParamGroup和Conn中。

        """
        fatigue_enabled = False
        branch: BranchInfo = self.branches[branch_id]

        if branch.gradient:
            return None

        reward_type = branch.reward_type

        if reward_type == RewardType.PENDING.value:
            return None
        
        

        link:dict =  branch.links[0]
        event:dict =link[Link.EVENT.value]
        param_group:ParameterGroup = branch.parent

        t0 = event[Event.START_TIMESTAMP.value]
        yt0 = link[Link.Condition.value.item.value][Link.Condition.value.Y0.value] # TODO: 在创建Link时记录此值

        fs = link[Link.Condition.value.item.value][Link.Condition.value.FATIGUE_SELF.value] if fatigue_enabled else 0
        fo = link[Link.Condition.value.item.value][Link.Condition.value.FATIGUE_OTHERS.value] if fatigue_enabled else 0
        p = param_group.windows[0][ParamWindow.P.value] if fatigue_enabled else 0
        q = param_group.windows[0][ParamWindow.Q.value] if fatigue_enabled else 0

        input_info_dict:dict = {}

        for trigger_event in link[Link.Trigger.value.item.value][Link.Trigger.value.EVENTS.value]:
            
            conn_id = trigger_event[Event.CONN_ID.value]

            input_info:dict = {}
            input_info['w'] = param_group.windows[0][ParamWindow.WEIGHTS.value][conn_id]
            input_info['b'] = param_group.windows[0][ParamWindow.BIAS.value][conn_id]
            attenuation = trigger_event[Event.Down_End_Attenuation.value.item.value]

            input_info['k'] = -(attenuation[Event.Down_End_Attenuation.value.Y1.value] - attenuation[Event.Down_End_Attenuation.value.Y0.value])/(attenuation[Event.Down_End_Attenuation.value.X1.value] - attenuation[Event.Down_End_Attenuation.value.X0.value]) # 令k>0 （取反）

            input_info['ft0'] = attenuation[Event.Down_End_Attenuation.value.Y0.value] - input_info['k'] * (t0 - attenuation[Event.Down_End_Attenuation.value.X0.value])

            input_info_dict[conn_id] = input_info


        activation_gradient:dict = self._calculate_activation_gradient(reward_type=reward_type, yt0=yt0, fs=fs, fo=fo, t0=t0, input_info_dict=input_info_dict, p=p, q=q)
        
        for trigger_event in link[Link.Trigger.value][Link.Trigger.value.EVENTS.value]:
            
            conn_id = trigger_event[Event.CONN_ID.value]
            conn_grad:dict = activation_gradient['conn_grads'][conn_id]
            ts = trigger_event[Event.Down_End_Attenuation.value][Event.Down_End_Attenuation.value.X0.value]
            b = trigger_event[Event.Down_End_Attenuation.value][Event.Down_End_Attenuation.value.Y0.value]
            conn_grad['ts'] = ts
            

            conn_grad_update = self._calculate_signal_gradient(reward_type=reward_type, k = conn_grad['k'], ts=ts, b=b, t0 = conn_grad['t0'], loss_ft=conn_grad['loss_ft'])
            conn_grad.update(conn_grad_update)
            

        branch.gradient = activation_gradient

        return activation_gradient




    def _select_branches(set1, set2, margin_window_len, max_window_len):
        """
        从set1中选取一个包含set2的子集，选取的子集的元素的应尽可能围绕set2元素的范围
        """

        # 将set1转换为排序后的列表
        list1 = sorted(set1)
        
        # 获取set2的最小值和最大值作为区间边界
        min_val = min(set2)
        max_val = max(set2)
        
        # 使用二分查找确定区间的起始和结束索引
        start_idx = bisect.bisect_left(list1, min_val)
        end_idx = bisect.bisect_right(list1, max_val) - 1
        
        # 计算区间内的元素数量
        interval_count = end_idx - start_idx + 1
        
        # 如果区间元素数量已超过最大窗口长度，直接截取前max_window_len个元素
        if interval_count >= max_window_len:
            selected = list1[start_idx : end_idx]
        else:
            # 计算剩余可扩展的数量
            remaining = max_window_len - interval_count
            
            # 计算左侧可扩展的数量
            left_max = min(margin_window_len, start_idx)
            left_add = min(left_max, remaining)
            remaining -= left_add
            
            # 计算右侧可扩展的数量
            right_max = min(margin_window_len, len(list1) - end_idx - 1)
            right_add = min(right_max, remaining)
            
            # 合并所有选中的元素
            left_part = list1[start_idx - left_add : start_idx]
            interval_part = list1[start_idx : end_idx + 1]
            right_part = list1[end_idx + 1 : end_idx + 1 + right_add]
            selected = left_part + interval_part + right_part
        
        # 转换为集合并返回
        return set(selected)

    def _calculate_snr(data: list[float]) -> float:
        """
        计算数列的信噪比（SNR，单位：dB）
        
        参数：
            data (list[float]): 输入浮点数列表
            
        返回：
            float: SNR分贝值，特殊情况处理：
                - 全零数据返回NaN
                - 数据全同且非零返回正无穷
                - 均值为零但存在噪声返回负无穷
                
        异常：
            ValueError: 输入空列表时抛出
        """
        if len(data) == 0:
            raise ValueError("输入列表不能为空")
        
        n = len(data)
        mean = sum(data) / n
        variance = sum((x - mean) ** 2 for x in data) / n
        std_dev = math.sqrt(variance)
        
        # 处理数据全同的情况
        if std_dev == 0:
            return float('inf') if mean != 0 else float('nan')
        
        # 处理均值为零的情况
        if mean == 0:
            return -math.inf
        
        # 常规SNR计算
        return 10 * math.log10(abs(mean) / std_dev)

    def _calculate_param_group_SNR(self)->tuple[float,set[int:"branch_id"]]:
        """
        统计branch的loss的SNR。优先考虑未将梯度修改应用于参数的branches，获取其时间跨度并计算跨度内branches的loss的SNR 10* log(平均值/标准差)
        """
        reward_confirmed_branches:set = self.negative_reward_branches.union(self.positive_reward_branches).union(self.mixed_reward_branches)
        unconsumed_branches:set = reward_confirmed_branches.difference(self.consumed_branches)

        if unconsumed_branches.__len__() ==0:
            return 0

        for id in unconsumed_branches:
            similarity = self._calculate_branch_fingerprint_similarity(branch_id = id)
            self.branches[id].fingerprint_similarities[self.group_id] = similarity


        branches:set = ParameterGroup._select_branches(reward_confirmed_branches, unconsumed_branches)
        
        losses = [self.branches[id].loss * self.branches[id].fingerprint_similarities[self.group_id]  for id in branches]
        snr = ParameterGroup._calculate_snr(losses)

        for id in unconsumed_branches:
            self.branches[id].loss_snrs[self.group_id] = snr
            self.branches[id]._update_flow_config()

        return snr


            
    def _gradient_update(self, branch_id):
        """
        根据branch_id对应的sink_cap和source_supply，利用gradient来更新梯度
        同时更新当前参数组的参数和对应Connection的信号参数
        """
        # 检查分支是否存在
        branch_info = self.branches.get(branch_id)
        if not branch_info:
            return False
        
        # 检查分支是否已被当前参数组消费过
        if self.group_id in branch_info.consumer:
            return False
        
        # 确保分支有梯度信息
        if not hasattr(branch_info, 'gradient') or not branch_info.gradient:
            self._calculate_branch_grandient(branch_id)
            # 如果计算后仍没有梯度，返回失败
            if not branch_info.gradient or not isinstance(branch_info.gradient, dict):
                return False
        
        # 计算可用的损失值
        total_available_loss = branch_info.sink_cap
        if branch_info.sink_cap == branch_info.max_sink_cap:
            # 当sink_cap达到最大值时，使用不超过2倍max_sink_cap的source_supply
            available_from_source = min(branch_info.source_supply, 2 * branch_info.max_sink_cap)
            total_available_loss += available_from_source
        
        # 设置学习率 - 从分支信息或全局获取
        lr = getattr(branch_info, 'learning_rate', 
                     getattr(self, 'learning_rate', 0.01))  # 默认学习率0.01
        
        # 获取梯度信息
        grads = branch_info.gradient
        
        """
        gradient结构说明:
        {
            'conn_grads': {
                conn_id: {
                    'dw': float,         # 权重w的梯度
                    'db': float,         # 偏置b的梯度
                    'dk': float,         # 信号参数k的梯度
                    'dts': float,        # 信号参数ts的梯度
                    'db_signal': float,  # 信号参数b的梯度 (原始值)
                    'int': float,        # 信号消耗的能量积分
                    'loss_ft': float     # 信号相关的损失梯度
                }
            },
            'dp': float,    # 参数p的梯度
            'dq': float,    # 参数q的梯度
            'dt0': float,   # t0的梯度
            't0': float,    # 原始t0值
            'total_integral': float  # 总积分值
        }
        """
        
        # 1. 更新当前参数组的参数（w, b, p, q）
        # 1.1 更新连接参数（权重w和偏置b）
        conn_grads = grads.get('conn_grads', {})
        for conn_id, conn_grad in conn_grads.items():
            # 确保连接在当前参数组中存在
            if (conn_id in self.windows[0][ParamWindow.WEIGHTS.value] and 
                conn_id in self.windows[0][ParamWindow.BIAS.value]):
                
                # 更新权重
                dw = conn_grad.get('dw', 0.0)
                if dw != 0:
                    self.windows[0][ParamWindow.WEIGHTS.value][conn_id] -= (
                        lr * total_available_loss * dw
                    )
                    # 权重必须不小于0 (还是必须大于0，或是无限制)？ TODO 测试下这里。
                    self.windows[0][ParamWindow.WEIGHTS.value][conn_id] = max(self.windows[0][ParamWindow.WEIGHTS.value][conn_id],0)


                
                # 更新偏置
                db = conn_grad.get('db', 0.0)
                if db != 0:
                    self.windows[0][ParamWindow.BIAS.value][conn_id] -= (
                        lr * total_available_loss * db
                    )
        
        # 1.2 更新全局参数p和q
        dp = grads.get('dp', 0.0)
        if dp != 0 and ParamWindow.P.value in self.windows[0]:
            self.windows[0][ParamWindow.P.value] -= (
                lr * total_available_loss * dp
            )
        
        dq = grads.get('dq', 0.0)
        if dq != 0 and ParamWindow.Q.value in self.windows[0]:
            self.windows[0][ParamWindow.Q.value] -= (
                lr * total_available_loss * dq
            )
        
        # 2. 更新对应Connection对象的信号参数（k, ts, b_signal）
        # 通过parent (PropagationManager) 获取Connection对象
        if self.parent:
            # 遍历所有有梯度的连接
            for conn_id, conn_grad in conn_grads.items():
                # 从PropagationManager获取Connection对象
                connection = self.parent.get_connection_by_id(conn_id)
                if not connection:
                    continue
                
                # 获取Connection的状态字典
                constraints:dict = connection.constraints
                if not constraints:
                    continue
                
                # 更新信号参数k（衰减率）
                dk = conn_grad.get('dk', 0.0)
                if dk != 0:
                    # 确保k参数存在
                    if "k" in constraints:
                        constraints["k"] = max(constraints["k"] - lr * total_available_loss * dk, 0.001)  # 防止变为0
                
                # 更新信号参数ts（起始时间）
                dts = conn_grad.get('dts', 0.0)
                if dts != 0:
                    # 确保ts参数存在
                    if "ts" in constraints:
                        # ts是时间戳，应用有意义的约束（如不低于最近事件的时间）
                        constraints["ts"] = max(constraints["ts"] - lr * total_available_loss * dts, 0)
                
                # 更新信号参数b（初始强度）
                db_signal = conn_grad.get('db_signal', 0.0)
                if db_signal != 0:
                    # 确保b参数存在
                    if "b" in constraints:
                        constraints["b"] = max(constraints["b"] - lr * total_available_loss * db_signal, 0.001)  # 防止变为0
        
        # 标记当前参数组已消费该分支
        branch_info.consumer.add(self.group_id)
        
        # # 可选：记录梯度应用情况
        # branch_info.last_updated = {
        #     'timestamp': time.time(),
        #     'available_loss': total_available_loss,
        #     'lr': lr,
        #     'param_group': self.group_id
        # }
        
        return True



    @staticmethod
    def static_differentiation(link:dict, conn:object, reward_type, loss_allocation_stepper:StepperIterator):
        # 特殊情况
        # 如果是F类型（抑制信号），且为执行机构的输出部分，不进行反向传播，不参与loss分配，各个因子直接采用设定值, （直接添加到loss_allocation_stepper，令其一回合返回 ）
        # if link[Link.EVENT.value][Event.SIGNAL.value] == Signal_F.F.value:
        # if conn.__getattribute__("isActuator") == True:
        if reward_type not in {RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value, RewardType.PENDING.value}:
            return
        branch:BranchInfo = link.get(Link.Condition.value.item.value, {}).get(Link.Condition.value.BRANCH.value, None)

        branch.sink_cap = 1
        branch.reward_type = reward_type
        branch.splitting_factor = 0
        branch.connecting_factor = 1
        branch.__setattr__("return_false",lambda **kwargs:False)
        key = (conn.__getattribute__("_connection_id"),link[Link.EVENT.value][Event.START_TIMESTAMP.value])
        schedule = []
        schedule.append({
            'obj':branch, # 这里应该是group里面所有unconsumed_branches
            'tick_method':"return_false",
            'kwargs':None
        })  
        loss_allocation_stepper.add(key=key, schedule=schedule)
        return

    def differentiation(self, link:dict, reward_type, loss_allocation_stepper:StepperIterator, min_branch_num = 5):
        """
        （Connection 分化）
        此函数根据传入的link来影响参数组的分裂，并判断loss当如何作用于参数组：
        1 认错：承认就是在此链接处导致了错误的发生，削弱激活，增强抑制 
        2 推诿：参数组完全无法区分错误与正确情况中上游的区别，坚信自己只是执行者，相信错误来自上游，只进行部分削弱，并将剩余削弱传递给上游，基于贡献度和度分配（成为他们的outer_loss）
        3 改进：承认此链接处导致了错误的发生，但同时此链接处也可能导致了正确的发生。通过分裂参数组分情况讨论即可。
        4 辅助改进：承认此链接处导致了错误的发生，但此错误无法通过1,2,3的任何一种方式解决，需要添加新的链接或节点以在此错误情况下对该链接进行抑制
        5 不确定：无法确定以上任何结论，直接将结果拖延，直至确定结论
        6 激荡：对于外部推诿而来的loss，如果自身处于推诿（情况2），则进一步传递此loss

        - link：需要分析的link
        - rewardtype:该link最终的回报类型
        - outer_loss:外部loss，由下游其他link推诿而来
        - min_branch_num: 最小分支数目，当分支数目小于此值时，无法分析

        # 分析过程：
        1 获取当前link所属的参数组branch（link中存储有特定的参数组引用），并确定reward_type
        2 确定需要更新的参数组：
            2.1 判断branch的分支数目是否大于特定值，如果小于特定值，则会因为数据过少，无法分析（对应情况5）
            2.2 如果branch的分支数目大于特定值，则尝试根据branch的拓扑结构将branches分裂归属多个参数组
                2.2.1 如果分裂成功，则step 3 对分裂出的新参数组进行参数更新 （对应情况3）
                2.2.2 如果分裂失败，则step 3 对原参数组进行参数更新
        3 对参数组进行参数更新:
            3.1 针对2中选定的参数组和对应的branches，计算其对参数组梯度的贡献，并分析这些贡献的方差/标准差 和 平均值 (可以构建变异系数或信噪比)组建置信函数。各个branch的贡献根据与指纹的拓扑相似度进行权重加成。
                3.1.1 如果置信度大于特定值，则认错，根据贡献、置信度，削弱激活，增强抑制（对应情况1）. （担责率： 0~1）
                3.1.2 如果置信度小于特定值，拓扑相似度高于特定阈值， 则在3.1.1 的基础上，推诿（对应情况 2）：（脱罪率： 0~1）
                    3.1.2.1 根据拓扑相似度和度（越高推诿的越多）将一定比例的[损失]分散给周围（无论来源）。
                        3.1.2.1.1 按照[周围节点在损失中的贡献度]分配[损失]的比例，以及担责率，分配给周围节点。
                    3.1.2.2 激荡状态下，将一部分来自外界的但存留在自身的[损失]替换为[探索动机]，激荡次数（loss包传递的次数，越多替换的越多，熵即平均传递次数，代表混乱度，混乱度越高，loss越不能用于更新参数，越触发进一步地探索）
                        3.1.2.2.1 根据自身的度（度越低转换的越多），将[探索动机]转化为[创建新链接动机]
                        3.1.2.2.2 根据自身的度，（度高低转换的越多），将[探索动机]转化为[创建新节点动机]
        """

        if reward_type not in {RewardType.POSITIVE.value, RewardType.NEGATIVE.value, RewardType.MIXED.value, RewardType.PENDING.value}:
            return

        # 1 获取branch引用并补充reward_type
        branch:BranchInfo = link.get(Link.Condition.value.item.value, {}).get(Link.Condition.value.BRANCH.value, None)
        if branch is None:
            raise ValueError("Branch cannot be None. Unable to backtrack activation.")
        new_group_fingerprint, new_group_branch_ids  = self._set_reward_loss_and_split(branch.get_branch_id(), reward_type = reward_type)
        self._calculate_branch_grandient(branch_id=branch.get_branch_id())

        # 2 确定需要更新的参数组
        param_group = branch.parent
        param_manager = param_group.parent
        # 2.1 判断branch的分支数目是否大于特定值
        if len(param_group.positive_reward_branches) + len(param_group.negative_reward_branches) + len(param_group.mixed_reward_branches) - len(param_group.consumed_branches) < min_branch_num:
            # 分支数目小于特定值，无法分析，返回情况5
            return 5

        # 2.2 如果branch的分支数目大于特定值，则尝试根据branch的拓扑结构将branches分裂归属多个参数组
        if len(new_group_branch_ids) > min_branch_num and len(new_group_fingerprint) > 0:
            # 2.2.1 如果分裂成功，则step 3 对分裂出的新参数组进行参数更新 （对应情况3）
            param_group = param_group.split(new_group_fingerprint, new_group_branch_ids)
            param_manager.add_group(param_group)
        else:
            # 2.2.2 如果分裂失败，则step 3 对原参数组进行参数更新
            pass

        # 3 loss分配和梯度更新 （同时完成connecting factor和splitting factor的计算 ）
        param_group._calculate_param_group_SNR()
        
        reward_confirmed_branches:set = param_group.negative_reward_branches.union(param_group.positive_reward_branches).union(param_group.mixed_reward_branches)
        unconsumed_branches:set = reward_confirmed_branches.difference(param_group.consumed_branches)

        for branch_id in unconsumed_branches:
            key = (branch.get_conn_id(), branch.get_timestamp())
            
            schedule = [] # 
            # step 1 (先统一更新)
            schedule.append({
                'obj':param_group.branches[branch_id], # 这里应该是group里面所有unconsumed_branches
                'tick_method':"apply_loss_flow",
                'kwargs':None
            })
            # step 2 （在统一计算流）
            schedule.append({
                'obj':param_group.branches[branch_id], # 
                'tick_method':"compute_loss_flow",
                'kwargs':None
            })

            loss_allocation_stepper.add(key=key, schedule=schedule)




        
        # # 2.2 如果有任意已经确定类型的branch的分支数目大于特定值，则分析branch的结果是否一致
        # branches = branch.branches

        # # 2.2.1 如果各个参数组branch
        


        # if reward_type is None:
        #     # 确定reward_type
        #     reward_type = self._determine_reward_type(branches)

        #
        

        



if TYPE_CHECKING:
    from NeuralModels import Cell,Baby,Connection
class PropagationManager:
    def __init__(self,conn_id = -1, baby:"object" = None, parent:"Connection" = None,**kwargs):
        self.groups: dict[int, ParameterGroup] = {}  # {group_id: ParameterGroup}
        
        self.history_window = 5  # 最大历史窗口数
        self.decay_factors = [0.9, 0.7, 0.5, 0.3, 0.1]  # 各历史窗口衰减因子
        self.group_id_counter = -1
        self.topology = {} # 一跳的拓扑结构
        self.conn_id = conn_id
        self.baby:"Baby|object" = baby
        self.parent:"Connection|object" = parent

        for k,v in kwargs.items():
            self.__setattr__(k,v)
        self.update_topology()

    def update_topology(self):
        if self.parent!=None:
            cellid = self.parent.__getattribute__("upstream_cell_id")
            if not cellid:
                return
            cellobj:"object" = self.baby.__getattribute__("cells")[cellid]
            self.topology = set(cellobj.__getattribute__("connectionsIn")).union(set(cellobj.__getattribute__("connectionsOut")))
            

        
    def add_group(self, initial_weights, initial_bias, initial_topology):
        """添加新的参数组"""
        self.group_id_counter += 1
        group_id = self.group_id_counter
        self.groups[group_id] = ParameterGroup(group_id)
        self.groups[group_id].windows[0] = {
            ParamWindow.WEIGHTS.value: initial_weights,
            ParamWindow.BIAS.value: initial_bias,
            Topology.item.value: initial_topology
        }
        self.groups[group_id].windows[0][ParamWindow.FINGERPRINT.value] = self._generate_fingerprint(self.groups[group_id].windows[0])

    def add_exist_group(self, group:ParameterGroup):
        """添加已存在的参数组"""
        self.group_id_counter += 1
        group_id = self.group_id_counter
        self.groups[group_id] = group
        group.group_id = group_id
        group.topology = self.topology
        group.parent = self



    def _create_window_from_link(self, link = {}, current_window = False, reference_window = None, hop = 1) -> dict:
        """根据link的拓扑结构创建窗口，只考虑一跳范围
        - link: 链接的拓扑结构
        - current_window: 是否为当前窗口
            (1) 如果为当前窗口，则创建窗口时不考虑link拓扑结构中的输出部分
            (2) 如果不为当前窗口，则创建窗口时需要考虑link拓扑结构中的输出部分

        - 以下为如何生成WEIGHTS、BIAS、TOPOLOGY三个字典    
            (1) WEIGHTS: 链接权重，采用输入输出链接的conn_id（int）作为键，1 （默认）作为值
            (2) BIAS: 链接偏置，采用输入输出链接的conn_id（int）作为键，-0.5（默认）作为值
            (3) TOPOLOGY: 链接拓扑，采用输入输出链接的conn_id（int）作为键，链接的拓扑信息（dict）作为值
            ``` Topology.item.value: {
                     Topology.WEIGHT.value: 1.0, # （因为有多个窗口，所以每个窗口占有不同的权重） 采用默认值 1 作为此拓扑结构在父结构的权重（此窗口在不同时间维度窗口的权重） 
                     Topology.INPUTS.value: {},  # 输入链接ID-链接信息映射, 链接信息可能为进一步嵌套的Topology字典，但是只考虑一跳的情况下，嵌套字典只包括WEIGHT，采用1,0作为权重，代表此节点在窗口中的权重
                     Topology.OUTPUTS.value: {}  # 输出链接ID-链接信息映射，同样在一跳情况下，只包含WEIGHT，无进一步地INPUT和OUTPUT嵌套
                 },
            ParamWindow.FINGERPRINT.value: {}，  # 窗口的指纹，由窗口内所有链接的ID-权重（初始默认1.0）映射组成的字典。```

        - reference_window: 参考窗口
            (1) 如果不为None，则创建窗口时，若链接在参考窗口中已知，需要从参考窗口中深复制对应的权重、偏置、拓扑结构等信息，或在结束后更新初始化信息。

        @attention (1) (2)中的WEIGHTS、BIAS为进行参数计算输入输出时的权重与偏置y = wx+b，而（3）TOPOLOGY的权重为拓扑结构各部分的权重，请读者不要混淆
        """

        # 初始化字典
        weights = {}
        biases = {}
        topology = {Topology.INPUTS.value: {}, Topology.OUTPUTS.value: {}}
        fingerprint = {}

        # 遍历输入链接
        for event in link.get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, []):
            conn_id = event[Event.CONN_ID.value]
            modulation_coef = ParameterGroup.get_relative_modulation(modulated_signal_type=self.parent.baby.connections[conn_id].constraints["signal_type"])
            weights[conn_id] = 1 * modulation_coef  # 默认权重为1
            biases[conn_id] = -0.5  # 默认偏置为-0.5
            topology[Topology.INPUTS.value][conn_id] = {Topology.WEIGHT.value: 1.0}  # 假设所有输入链接的拓扑权重为1

            fingerprint[conn_id] = 1.0  # 假设所有输入链接的初始权重为1

        # 如果当前窗口，则不考虑输出链接
        if not current_window:
            for event in link.get(Link.Sequence.value.item.value, {}).get(Link.Sequence.value.EVENTS.value, {}).items():
                conn_id = event[Event.CONN_ID.value]
                modulation_coef = ParameterGroup.get_relative_modulation(modulated_signal_type=self.parent.baby.connections[conn_id].constraints["signal_type"])
                weights[conn_id] = 1 * modulation_coef  # 默认权重为1
                biases[conn_id] = -0.5  # 默认偏置为-0.5
                topology[Topology.OUTPUTS.value][conn_id] = {Topology.WEIGHT.value: 1.0}  # 假设所有输出链接的拓扑权重为1

                fingerprint[conn_id] = 1.0  # 假设所有输出链接的初始权重为1

        # 如果提供了参考窗口，则从参考窗口中复制数据
        if reference_window:
            for conn_id in weights:
                if conn_id in reference_window[ParamWindow.WEIGHTS.value]:
                    weights[conn_id] = reference_window[ParamWindow.WEIGHTS.value][conn_id]
                    biases[conn_id] = reference_window[ParamWindow.BIAS.value][conn_id]
                    # 复制拓扑信息
                    if conn_id in reference_window[Topology.item.value][Topology.INPUTS.value]:
                        topology[Topology.INPUTS.value][conn_id] = copy.deepcopy(reference_window[Topology.item.value][Topology.INPUTS.value].get(conn_id, {}))
                    elif conn_id in reference_window[Topology.item.value][Topology.OUTPUTS.value] and not current_window:
                        topology[Topology.OUTPUTS.value][conn_id] = copy.deepcopy(reference_window[Topology.item.value][Topology.OUTPUTS.value].get(conn_id, {}))
                    fingerprint[conn_id] = reference_window[ParamWindow.FINGERPRINT.value].get(conn_id, 1.0)
     

        # 设置默认拓扑权重为1.0
        topology[Topology.WEIGHT.value] = 1.0


        return {
            ParamWindow.WEIGHTS.value: weights,
            ParamWindow.BIAS.value: biases,
            Topology.item.value: topology,
            ParamWindow.FINGERPRINT.value: fingerprint
        }
        
       

    def _generate_fingerprint(self, window_data:dict):
        """递归生成带权重的拓扑指纹，这里直接统计所有访问到的端口，并将其id-weight映射到字典中(只针对单个窗口)"""
        fingerprint = {        }
        topology_inputs = window_data.get(Topology.item.value, {}).get(Topology.INPUTS.value, {})
        topology_outputs = window_data.get(Topology.item.value, {}).get(Topology.OUTPUTS.value, {})
        for conn_id, conn in topology_inputs.items():
            fingerprint[conn_id] = conn[Topology.WEIGHT.value]
            self._traverse_input_conn( conn, fingerprint)

            
        for conn_id, conn in topology_outputs.items():
            fingerprint[conn_id] = conn[Topology.WEIGHT.value]
            self._traverse_output_conn(conn_id, conn, fingerprint)
                
        return fingerprint
    
    def _traverse_input_conn(self, conn, fingerprint):
        for child_conn_id, child_conn in conn.get(Topology.INPUTS.value, {}).items():
            # 保持原始权重参数（用于后续计算）
            fingerprint[child_conn_id] = child_conn[Topology.WEIGHT.value]
            # 递归处理子节点
            self._traverse_input_conn(
                child_conn,
                fingerprint
            )
    
    def _traverse_output_conn(self, conn:dict, fingerprint):
        for child_conn_id, child_conn in conn.get(Topology.OUTPUTS.value, {}).items():
            # 保持原始权重参数（用于后续计算）
            fingerprint[child_conn_id] = child_conn[Topology.WEIGHT.value]
            # 递归处理子节点
            self._traverse_output_conn(
                child_conn,
                fingerprint
            )

    def _calculate_shot_missing(self, in_conn_events = None, out_conn_events = None, target_fingerprint:dict[int:float] = {}):
        """计算单个端口的shot_weight和missing_weight
            - in_conn_events 或 out_conn_events 为输入或输出端口的事件 ，一次调用中，只有一个不为None，否则报错。
            - target_fingerprint 为目标窗口的拓扑指纹,dict[conn_id:weight]
            * 如果此端口在拓扑中不存在，missing_weight = 1.0
            * 如果此端口在拓扑中存在，则递归累加shot_weight和missing_weight。对于input端口，递归处理其上游Input端口，对于output端口，递归处理其下游Output端口。
            * 在递归处理时，如果存在上游/下游端口全部无法在target_fingerprint中找到，则停止递归后续节点。
        
        """
        shot = 0
        miss = 0
        # 确定处理的是输入事件还是输出事件
        if in_conn_events and out_conn_events is None:
            events = in_conn_events
            is_input = True
        elif out_conn_events:
            events = out_conn_events
            is_input = False
        else:
            raise ValueError("Either in_conn_events or out_conn_events must be provided")

        for event in events:
            conn_id = event[Event.CONN_ID.value]
            # 检查当前conn_id是否在目标指纹中
            if conn_id in target_fingerprint:
                # 累加当前节点的权重
                current_weight = target_fingerprint[conn_id]
                shot += current_weight
                
                # 递归处理子端口
                if is_input:
                    # 输入事件的上游触发事件的端口在inputs中
                    child_conns = event[Event.LINK.value].get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])
                else:
                    # 输出事件的下游端口在outputs中
                    child_conns = event[Event.LINK.value].get(Link.Sequence.value.item.value, {}).get(Link.Sequence.value.EVENTS.value, [])
                
                # 递归处理每个子连接
                for child_conn in child_conns:
                    child_shot, child_miss = self._calculate_shot_missing(
                        [child_conn] if is_input else None,
                        [child_conn] if not is_input else None,
                        target_fingerprint
                    )
                    shot += child_shot
                    miss += child_miss
            else:
                # 当前conn_id不在目标指纹中，累加missing
                miss += 1.0

        return shot, miss



    def _match_window(self, in_conn_events = [], out_conn_events = [], target_fingerprint:dict[int:float] = {}):
        """此函数对一个输入输出拓扑结构与一个窗口的拓扑进行匹配评分
        - in_conn_events 为输入端口事件序列， 
        - out_conn_events 为输出端口事件序列， 
        - target_fingerprint 为目标窗口的拓扑指纹。

        """
        shot_weight = 0
        miss_weight = 0
        # 查询匹配程度
        for event in in_conn_events:
            shot, miss = self._calculate_shot_missing(event, target_fingerprint)
            shot_weight += shot
            miss_weight += miss
            
        for event in out_conn_events:
            shot, miss = self._calculate_shot_missing(event, target_fingerprint)
            shot_weight += shot
            miss_weight += miss

                    
        return shot_weight, miss_weight


    def _match_group(self, links: dict[int:dict] = {}, group:ParameterGroup = None):

        """此函数对一个输入激活输出的拓扑结构事件序列与一个事件组对应的多窗口拓扑进行匹配评分
        - links 为激活序列，一个link为一组输入激活一组输出（或只有输入，未触发输出）
        - group 为事件组，其内部各个窗口存储了各自的拓扑。
        - 调用_match_window计算一个link的拓扑与一个窗口的匹配程度
        * 需要将group所有窗口的拓扑(1,2,3，...)与link的拓扑（1,2,3，...）依次进行一一对应的匹配。
        * 如果links的长度小于group的窗口数，那么无法匹配的group的窗口的拓扑匹配权重全部统计为miss。
        * 根据各个窗口的权重，计算最终的匹配程度shot/（shot+miss）
        """
        if group is None or not group.windows:
            return {"total_score": 0.0, "normalized_score": 0.0, "window_scores": []}

        total_score = 0.0
        window_scores = []
        max_group_window = max(group.windows.keys())  # 获取group的最大窗口索引
        
        for window_idx in range(max_group_window + 1):  # 遍历所有可能的窗口索引
            # 获取衰减因子（自动适配窗口数量）
            decay = self.decay_factors[window_idx] if window_idx < len(self.decay_factors) else 0.1
            
            # 获取当前窗口数据
            window_data = group.windows.get(window_idx, {})
            target_fingerprint = window_data.get(ParamWindow.FINGERPRINT.value, {})
            
            # 处理当前窗口的匹配
            if window_idx in links:
                # 存在对应窗口的链接事件
                link_info:dict = links[window_idx]
                in_conn_events = link_info.get(Link.Trigger.value, {}).get(Link.Trigger.value.EVENTS.value, [])
                out_conn_events = link_info.get(Link.Sequence.value, {}).get(Link.Sequence.value.EVENTS.value, [])
                
                # 执行窗口匹配计算
                shot, miss = self._match_window(
                    in_conn_events=in_conn_events,
                    out_conn_events=out_conn_events,
                    target_fingerprint=target_fingerprint
                )
            else:
                # 无对应链接事件，统计整个窗口权重的miss
                total_conn_weight = 0.0
                fp:dict[int:float] = window_data.get(ParamWindow.FINGERPRINT.value, {})
                
                # 累加输入端口权重
                for conn_id, weight in fp.items():
                    total_conn_weight += weight
                    
                shot, miss = 0.0, total_conn_weight
            
            # 计算窗口得分（防止除零错误）
            window_topology_weight = window_data.get(Topology.item.value, {}).get(Topology.WEIGHT.value, 1.0)
            window_score = (shot / (shot + miss + 1e-8)) * decay * window_topology_weight if (shot + miss) > 0 else 0.0
            window_scores.append({
                "window_idx": window_idx,
                "shot": shot,
                "miss": miss,
                "window_score": window_score,
                "decay": decay,
                "window_topology_weight": window_topology_weight
            })
            total_score += window_score

        # 计算归一化得分（考虑衰减因子权重）
        max_possible_score = sum(
            window_score["decay"] * window_score["window_topology_weight"] for window_score in window_scores
        )
        normalized_score = total_score / max_possible_score if max_possible_score > 0 else 0.0

        return normalized_score







    def find_optimal_group(self, links: dict[int:dict] = {}):
        """主查询接口，返回最优参数组的引用"""
        best_group = None
        best_group_id = -1
        best_normalized = -1.0


        for group_id, group in self.groups.items():
            # 执行匹配评分
            match_result = self._match_group(links=links, group=group)
            
            # 优先比较归一化得分，其次比较原始得分
            if match_result > best_normalized:
                
                best_group_id = group_id
                best_normalized = match_result
                best_group = group
                
                

        
        
        # 构建返回结果
        return {
            "best_group_id": best_group_id,
            "best_group": best_group,
            "best_score": best_normalized,
        }


    def cascade_activation(self, timestamp, link_layers: list[dict] = [], **kwargs) -> tuple[bool, ParameterGroup, BranchInfo]:
        """
        根据传入的链接信息，判断是否进行级联激活此链接，并返回是否进行级联激活的结果。
        - links 为时间维度的激活序列，一个link为一组输入激活一组输出（或只有输入，未触发输出）
        - 步骤：
            0. 根据已有缓存和当前link[0]（见步骤3）一致的参数组，判断是否需要执行后续步骤
            1. 查询最优参数组
            1.1 如果参数组为空或不存在匹配度大于0的参数组，则创建完全与当前窗口拓扑结构一致的临时参数组。
            1.2 如果存在匹配度大于0的参数组，则采用最大匹配度的参数组。
            1.3 如果最优参数组与当前links[0]的输入拓扑结构不一致，即link[0]的拓扑结构与最优参数组的第一个窗口的拓扑结构不一致，则创建完全与当前窗口拓扑结构一致的临时参数组。但采用最优参数组的权重初始化对应链接的权重。
            2. 根据最优参数组，代入links的信息，进行计算，判断是否达到激活阈值
            (判断方式： 计算第一个窗口所包含参数组与第一个link计算的得分：y = w*x + b, w为窗口中的WEIGHT字典中ID-权重映射中的权重，b为窗口中的BIAS字典中ID-权重映射中的偏置，x为link中反映信号强度的量，y为窗口的得分。最终所有窗口的得分累加，结果大于0则激活成功（达到阈值），否则激活失败（未达到阈值）。这里的阈值就是0。)
            2.1 如果达到激活阈值，最终返回真，与最优参数组。需要在外部为此参数组添加新的分支（分支和Links内部存有相互引用，在传播后彼此解耦。）
            2.2 如果未达到激活阈值，最终返回假，与None。
            3. 返回前缓存最优参数组
            3.1 缓存事件组，同样的事件组不能重复激活
            3.2 特别地，当激活成功时，需要在参数组下创建供回溯的branch，并返回branch的引用以将branch与link的Trigger绑定
        
        """

        # Step 0: Check if activation is needed based on the cache and current input topology

        # 获取输入事件组
        if not link_layers:
            return False, None, None
        current_input_events = set([(event[Event.CONN_ID.value], event[Event.ID.value]) for event in link_layers[0].get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])])




        # Step 1: Find the optimal ParameterGroup based on the provided links
        best_group_info = self.find_optimal_group(links=link_layers)
        best_group: ParameterGroup = best_group_info['best_group']
        
        if best_group is None:
            # If no best group found, create a temporary group with the current topology
            new_group = ParameterGroup(group_id=-1)
            new_group.windows[0] = self._create_window_from_link(link=link_layers[0], current_window=True)
            best_group = new_group  # Set the new temporary group as the best group

        else:
            # If a best group is found, check if it matches the current input topology (用不太精确的指纹代替拓扑，否则跑太慢了)
            # 获取links[0]的输入链接拓扑
            current_input_topology = set([event[Event.CONN_ID.value] for event in link_layers[0].get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, [])])
            best_group_input_topology = set(best_group.windows[0].get(ParamWindow.FINGERPRINT.value, {}).keys())
            
            # 判断current_input_topology是否包含任何best_group_input_topology不存在的元素（差集不为空）
            if current_input_topology.difference(best_group_input_topology):
                # If the topologies don't match, create a temporary group with the current topology
                new_group = ParameterGroup(group_id=-1)
                new_group.windows[0] = self._create_window_from_link(link=link_layers[0], current_window=True, reference_window=best_group.windows[0])
                best_group = new_group  # Set the new temporary group as the best group
        

        # Step 2: Check if the best group reaches the activation threshold
        total_score = 0
        
        # Assuming each window has a weight and bias dictionary for calculation
        weight_dict = best_group.windows[0][ParamWindow.WEIGHTS.value]
        bias_dict = best_group.windows[0][ParamWindow.BIAS.value]
        link = link_layers[0]  # Assume we're working with the first layer of link for simplicity
        
        # Calculate score for this window
        activation_strengths = {}
        for event in link.get(Link.Trigger.value.item.value, {}).get(Link.Trigger.value.EVENTS.value, []):
            conn_id = event[Event.CONN_ID.value]
            # Get signal strength of the event at current timestamp
            activation_strength = weight_dict.get(conn_id, 0) * get_signal_strength(event, timestamp-1)

            total_score += activation_strength + bias_dict.get(conn_id, -0.5)

            activation_strengths[conn_id] = activation_strength
        

        # Step 3: Determine if the activation threshold is met （TODO： 这里需要竞争性探索，所以判别式需要改）
        if total_score > 0:
            # If activation succeeds, return True with the best group
            self.cache_group(timestamp, current_input_events)
            # If activation succeeds and the group is new, add it to the list
            if best_group.group_id not in self.groups:
                self.group_id_counter += 1
                best_group.group_id = self.group_id_counter
                self.groups[best_group.group_id] = best_group
                    
            # Create a new branch for this activation
            # TODO: 
            new_branch = BranchInfo(key=(self.conn_id,timestamp),links=link_layers, match_score=total_score, reward=0, penalty=0, fingerprints=[self._generate_fingerprint(best_group.windows[0])], param_group=best_group)
            
            register_branchInfo = self.baby.__getattribute__("register_branchInfo")
            register_branchInfo(new_branch)

            best_group.branches[new_branch.get_branch_id()] = new_branch
            best_group.pending_reward_branches.add(new_branch.get_branch_id())

            # 在link中记录激活条件
            link[Link.Condition.value.item.value] = {Link.Condition.value.BRANCH.value:new_branch, Link.Condition.value.ACTIVATION_STRENGTHS.value:activation_strengths,Link.Condition.value.Y0.value:total_score}
            return True, best_group, new_branch
        
        else:
            self.cache_group(timestamp, current_input_events)
            return False, None, None

    def cache_group(self, timestamp, current_input_events):
        """Cache the activated group along with the timestamp for cooling down."""
        self.activation_cache = {'timestamp': timestamp, 'input_events': current_input_events}

    def are_same_inputs(self, current_timestamp, current_input_events):
        """Check if the topology has remained unchanged since the last activation."""
        last_input_events = self.activation_cache.get('input_events')
        
        if current_input_events == last_input_events:
            self.activation_cache.update({'timestamp': current_timestamp})            
            return True
        return False
    
#     def loss_locate(self, link:dict, reward_type, outer_loss = 0, min_branch_num = 5):
#         """
#         此函数根据传入的link来判断loss当如何作用于参数组：
#         1 认错：承认就是在此链接处导致了错误的发生，削弱激活，增强抑制 
#         2 推诿：参数组完全无法区分错误与正确情况中上游的区别，坚信自己只是执行者，相信错误来自上游，只进行部分削弱，并将剩余削弱传递给上游，基于贡献度和度分配（成为他们的outer_loss）
#         3 改进：承认此链接处导致了错误的发生，但同时此链接处也可能导致了正确的发生。通过分裂参数组分情况讨论即可。
#         4 辅助改进：承认此链接处导致了错误的发生，但此错误无法通过1,2,3的任何一种方式解决，需要添加新的链接以在此错误情况下对该链接进行抑制
#         5 不确定：无法确定以上任何结论，直接将结果拖延，直至确定结论
#         6 激荡：对于外部推诿而来的loss，如果自身依旧无法区别错误与正确并尝试推诿（情况2），则将一部分错误返回给loss来源

#         - link：需要分析的link
#         - rewardtype:该link最终的回报类型
#         - outer_loss:外部loss，由下游其他link推诿而来
#         - min_branch_num: 最小分支数目，当分支数目小于此值时，无法分析

#         # 分析过程：
#         1 获取当前link所属的参数组branch（link中存储有特定的参数组引用），并确定reward_type
#         2 当回报类型确定的branch判断是否因该参数组的调用导致了错误或失败：
#             2.1 判断branch的分支数目是否大于特定值，如果小于特定值，则会因为数据过少，无法分析（对应情况5）
#             2.2 如果branch的分支数目大于特定值，则分析branch的结果是否一致
#                 2.2.1 如果各个参数组branch中蕴含的结果均一致（均为成功或失败），则认错，削弱激活，增强抑制（对应情况1）
#                     2.2.1.1 特别地，如果参数组在根据结果修改后发生变号，则需要将此参数组移到新的conn中，并且新的conn的激活将抑制原conn的活性。（对应情况 4）
#                 2.2.2 如果各个参数组branch中蕴含的结果不一致，则判断是否可以将参数组分裂：
#                     2.2.2.1 如果众多branches根据成功失败可以分为不重合的组，则将参数组分裂，（对应情况3，可进一步进行情况1判断）
#                     2.2.2.2 如果众多branches根据成功失败将一部分分为不重合的组，另一部分则出现重合，则尽可能分裂参数组（对应情况3），不能分裂的部分留下来，直到：
#                         2.2.2.2.1 若所有branch均无法根据成功失败分为不重合的组，则推诿 （对应情况2）

                
#         """
            
#         # 1 获取branch引用并补充reward_type
#         branch:BranchInfo = link.get(Link.Condition.value.item.value, {}).get(Link.Condition.value.BRANCH.value, None)
#         if branch is None:
#             raise ValueError("Branch cannot be None. Unable to backtrack activation.")
#         branch.reward_type = reward_type
        
#         # 2 判断branch的分支数目是否大于特定值
#         param_group = branch.parent


#         if len(param_group.) < min_branch_num:
#             # 分支数目小于特定值，无法分析，返回情况5
#             return 5
        
#         # 2.2 如果有任意已经确定类型的branch的分支数目大于特定值，则分析branch的结果是否一致
#         branches = branch.branches
#         # 2.2.1 如果各个参数组branch



#         if reward_type is None:
#             # 确定reward_type
#             reward_type = self._determine_reward_type(branches)

#         #
        

        
        

        

        






#     def backpropagation(self, link):











# ## 避免链接波动干扰
# # 当一个链接的输入过小时，需要重新考量其是否可以作为参数组的拓扑指纹





