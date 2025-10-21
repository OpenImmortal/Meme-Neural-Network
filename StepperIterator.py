class StepperIterator:
    def __init__(self):
        self.workspace = {}  # {key: {'schedule': [], ...}}
        self.buffer = {}

    def add(self, key, schedule:list[dict]):
        """添加带调度计划的对象
        推荐key（conn_id/cell_id， timestamp）
        - schedule:[
            {'obj':object,
            'tick_method':str,
            'kwargs':dict},
            ...更多阶段...
        ]
        只有所有元素执行完一个阶段后，才会执行后续阶段
        """
        # 验证调度表结构
        for idx, stage in enumerate(schedule):
            if 'obj' not in stage or 'tick_method' not in stage:
                raise ValueError(f"Invalid stage {idx}，missing obj or tick_method")
            
        self.buffer[key] = {
            'schedule': [dict(s) for s in schedule],  # 防御性拷贝
        }

    def tick(self, timestamp, **kwargs):
        """执行完整调度周期"""
        # 合并缓冲区
        self.workspace.update(self.buffer)
        self.buffer.clear()
        
        removed = set()
        max_depth = max(len(v['schedule']) for v in self.workspace.values()) if self.workspace else 0

        # 分阶段执行
        for phase in range(max_depth):
            current_phase = phase  # 当前阶段索引
            phase_removal = []
            
            # 遍历所有存活对象
            for key in [k for k in self.workspace if k not in removed]:
                item = self.workspace[key]
                
                # 跳过短于当前阶段的调度
                if current_phase >= len(item['schedule']):
                    continue
                
                # 准备阶段参数
                stage = item['schedule'][current_phase]
                merged_kwargs = {
                    'timestamp': timestamp,
                    **(stage.get('kwargs') if stage.get('kwargs') else {}),
                    **(kwargs if kwargs else {})
                }
                
                try:
                    # 动态调用阶段方法
                    method = getattr(stage['obj'], stage['tick_method'])
                    if not method(**merged_kwargs):
                        phase_removal.append(key)
                except Exception as e:
                    print(f"Error processing {key} phase {current_phase}: {e}")
                    phase_removal.append(key)
            
            # 记录本阶段淘汰对象
            removed.update(phase_removal)
        
        # 清理淘汰对象
        removed_values = []
        for key in removed:
            removed_values.append(self.workspace[key]['schedule'])
            del self.workspace[key]

        return removed_values
    


class LossAllocationStepper:
    """
    代码与StepperIterator一致，用于调试所以单独复制了一遍
    """
    def __init__(self):
        self.workspace = {}  # {key: {'schedule': [], ...}}
        self.buffer = {}

    def add(self, key, schedule:list[dict]):
        """添加带调度计划的对象
        推荐key（conn_id/cell_id， timestamp）
        - schedule:[
            {'obj':object,
            'tick_method':str,
            'kwargs':dict},
            ...更多阶段...
        ]
        只有所有元素执行完一个阶段后，才会执行后续阶段
        """
        # 验证调度表结构
        for idx, stage in enumerate(schedule):
            if 'obj' not in stage or 'tick_method' not in stage:
                raise ValueError(f"Invalid stage {idx}，missing obj or tick_method")
            
        self.buffer[key] = {
            'schedule': [dict(s) for s in schedule],  # 防御性拷贝
        }

    def tick(self, timestamp, **kwargs):
        """执行完整调度周期"""
        # 合并缓冲区
        self.workspace.update(self.buffer)
        self.buffer.clear()
        
        removed = set()
        max_depth = max(len(v['schedule']) for v in self.workspace.values()) if self.workspace else 0

        # 分阶段执行
        for phase in range(max_depth):
            current_phase = phase  # 当前阶段索引
            phase_removal = []
            
            # 遍历所有存活对象
            for key in [k for k in self.workspace if k not in removed]:
                item = self.workspace[key]
                
                # 跳过短于当前阶段的调度
                if current_phase >= len(item['schedule']):
                    continue
                
                # 准备阶段参数
                stage = item['schedule'][current_phase]
                merged_kwargs = {
                    'timestamp': timestamp,
                    **(stage.get('kwargs') if stage.get('kwargs') else {}),
                    **(kwargs if kwargs else {})
                }
                
                try:
                    # 动态调用阶段方法
                    method = getattr(stage['obj'], stage['tick_method'])
                    if not method(**merged_kwargs):
                        phase_removal.append(key)
                except Exception as e:
                    print(f"Error processing {key} phase {current_phase}: {e}")
                    phase_removal.append(key)
            
            # 记录本阶段淘汰对象
            removed.update(phase_removal)
        
        # 清理淘汰对象
        removed_values = []
        for key in removed:
            removed_values.append(self.workspace[key]['schedule'])
            del self.workspace[key]

        return removed_values