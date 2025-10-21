from __future__ import annotations
from typing import TYPE_CHECKING

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from typing import List, Optional
import math
import numpy as np
from scipy.spatial.distance import cdist
from matplotlib.patches import ConnectionPatch
from matplotlib.path import Path
from matplotlib.bezier import BezierSegment

if TYPE_CHECKING:
    from NeuralModels import Cell, Connection
from ConstantEnums import GateStatus,ConnectionStatus
# 封装为可更新的可视化类
class NeuralNetworkVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots(figsize=(24, 19))
        self.pos = None  # 用于保存布局位置
        self.ax.set_title('Neural Network Visualization', fontsize=14)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.min_node_distance = 0.03  # 节点最小间距
        self.max_layout_iterations = 30  # 布局优化最大迭代次数

        
        # 定义颜色和样式映射
        self.colors = {
            "receptor": "#1F78B4",     # 蓝色
            "actuator": "#E31A1C",     # 红色
            "neuron": "#33A02C",       # 绿色
            "virtual_up": "#A6CEE3",   # 浅蓝（上游虚拟）
            "virtual_down": "#FB9A99"   # 浅红（下游虚拟）
        }

        self.conn_color_map = {
            GateStatus.OPENING.value: "#FFA500",         # 橙色 - 进行中（类似交通信号）
            GateStatus.OPENED.value: "#4CAF50",          # 绿色 - 安全/通行状态
            GateStatus.CLOSED_CHARGING.value: "#607D8B", # 灰蓝色 - 充电/等待状态
            GateStatus.CLOSED_READY.value: "#2196F3",    # 蓝色 - 就绪/可用状态
            GateStatus.OTHER.value: "#9E9E9E",            # 灰色 - 未知/其他状态
            None: "#7F7F7F" # 另一种灰色
        }
        
        self.node_styles = {
            "receptor": "s",           # 方块
            "actuator": "d",           # 菱形
            "neuron": "o",             # 圆形
            "virtual_up": "^",         # 三角形向上
            "virtual_down": "v"        # 三角形向下
        }
        
        # 创建图例
        legend_elements = [
            mpatches.Patch(facecolor=self.colors["receptor"], edgecolor='lightgray', label='Receptor'),
            mpatches.Patch(facecolor=self.colors["actuator"], edgecolor='lightgray', label='Actuator'),
            mpatches.Patch(facecolor=self.colors["neuron"], edgecolor='lightgray', label='Neuron'),
            mpatches.Patch(facecolor=self.colors["virtual_up"], edgecolor='lightgray', label='Input Interface'),
            mpatches.Patch(facecolor=self.colors["virtual_down"], edgecolor='lightgray', label='Output Interface'),
            mpatches.Patch(facecolor='white', edgecolor='lightgray', linestyle=':', label='Active Cells'),
            plt.Line2D([0], [0], marker='', color='darkgray', label='Connections')
        ]
        self.ax.legend(handles=legend_elements, loc='best', fontsize=10)
    
    def _create_layered_layout(self, G):
        """创建分层布局，神经元层按每10个细胞分为子层"""
        # 按节点类型收集节点
        node_types = {}
        for node, data in G.nodes(data=True):
            node_type = data['node_type']
            if node_type not in node_types:
                node_types[node_type] = []
            node_types[node_type].append(node)
        
        # 特殊处理神经元层：分割为多个子层
        neuron_layers = []
        if 'neuron' in node_types:
            neurons = node_types['neuron']
            # 将神经元分为最多每层10个细胞的子层
            neuron_layers = [neurons[i:i+10] for i in range(0, len(neurons), 10)]
            del node_types['neuron']  # 移除原始神经元层
        
        # 创建有序的分层结构
        layers = []
        
        # 第一层：输入接口
        if 'virtual_up' in node_types:
            layers.append(('input', node_types['virtual_up']))
        
        # 第二层：受体
        if 'receptor' in node_types:
            layers.append(('receptor', node_types['receptor']))
        
        # 第三层：神经元子层
        for i, neuron_layer in enumerate(neuron_layers):
            layers.append((f'neuron_{i}', neuron_layer))
        
        # 第四层：效应器
        if 'actuator' in node_types:
            layers.append(('actuator', node_types['actuator']))
        
        # 第五层：输出接口
        if 'virtual_down' in node_types:
            layers.append(('output', node_types['virtual_down']))
        
        # 计算每层的X坐标布局（等距分布）
        pos = {}
        layer_count = len(layers)
        if layer_count == 0:
            return {}
        
        layer_width = 1.0 / layer_count
        layer_offset = layer_width / 2
        
        for i, (layer_name, nodes) in enumerate(layers):
            x = layer_offset
            # 计算节点在层内的Y坐标布局
            layer_height = len(nodes)
            if layer_height > 1:
                y_spacing = 0.8 / (layer_height - 1)  # 留出顶部和底部空间
                y_offset = 0.1
            else:
                y_spacing = 0
                y_offset = 0.5
                
            # 将节点均匀分布在当前层
            for j, node in enumerate(nodes):
                y = y_offset + j * y_spacing
                pos[node] = (x, y)
            
            layer_offset += layer_width
        
        return pos
    def _apply_min_distance_constraint(self, pos, G):
        """应用最小距离约束，确保节点不会重叠"""
        # 转换为numpy数组以便计算
        node_ids = list(pos.keys())
        positions = np.array([pos[node] for node in node_ids])
        
        # 迭代优化节点位置
        for _ in range(self.max_layout_iterations):
            # 计算所有节点对的距离矩阵
            dist_matrix = cdist(positions, positions)
            
            # 找到距离小于阈值的节点对
            close_pairs = np.where(dist_matrix < self.min_node_distance)
            overlapping_pairs = [(i, j) for i, j in zip(*close_pairs) if i < j]
            
            if not overlapping_pairs:
                break  # 没有重叠节点，退出
                
            # 计算所有节点的排斥力向量
            repulsive_forces = np.zeros_like(positions)
            
            # 计算排斥力
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    if i == j:
                        continue
                        
                    vector = positions[i] - positions[j]
                    distance = np.linalg.norm(vector)
                    
                    if distance < self.min_node_distance:
                        # 计算排斥力 - 随距离增加而增加
                        overlap = self.min_node_distance - distance
                        force = (overlap * 0.5) * (vector / (distance + 1e-8))
                        repulsive_forces[i] += force
                        repulsive_forces[j] -= force
            
            # 应用排斥力 - 但不移动受体/效应器节点太多
            is_core_node = [not isinstance(node, str) for node in node_ids]
            displacement = 0.1 * repulsive_forces
            displacement[is_core_node] *= 0.8  # 核心节点移动较少
            positions += displacement
            
            # 限制移动范围
            positions = np.clip(positions, 0, 1)
        
        # 更新位置字典
        return {node: pos for node, pos in zip(node_ids, positions)}
    
    def _calculate_curve_point(self, start, end, rad, ratio):
        """
        计算贝塞尔曲线上的精确点位置
        :param start: 起点坐标 (x, y)
        :param end: 终点坐标 (x, y)
        :param rad: 曲线弧度值
        :param ratio: 在曲线上的比例 (0-1, 0=起点, 1=终点)
        :return: 曲线上的点坐标 (x, y)
        """
        # 计算控制点 - 根据matplotlib的arc3连接样式算法
        mid = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        ctrl_x = mid[0] + rad * (start[1] - end[1])
        ctrl_y = mid[1] + rad * (end[0] - start[0])
        
        # 定义贝塞尔曲线的控制点
        bezier = BezierSegment(np.array([
            [start[0], start[1]], 
            [ctrl_x, ctrl_y],
            [end[0], end[1]]
        ]))
        
        # 计算给定比例的点
        point = bezier.point_at_t(ratio)
        return point
    
    def _calculate_label_position(self, start, end, rad=None, endpoint_ratio=0.62):
        """
        计算标签的精确位置，距离末端指定比例
        :param start: 起点坐标 (x, y)
        :param end: 终点坐标 (x, y)
        :param rad: 曲线弧度值 (如果是直线则为None)
        :param endpoint_ratio: 距离末端的比例 (0.62表示距离末端0.38)
        :return: 标签位置坐标 (x, y)
        """
        # 对于直线连接
        if rad is None or rad == 0:
            return (
                start[0] + endpoint_ratio * (end[0] - start[0]),
                start[1] + endpoint_ratio * (end[1] - start[1])
            )
        
        # 对于曲线连接
        # 使用牛顿迭代法找到距离末端指定比例的点
        precision = 0.01
        min_t = 0.0
        max_t = 1.0
        
        for _ in range(5):  # 最多迭代5次
            t = (min_t + max_t) / 2.0
            curve_point = self._calculate_curve_point(start, end, rad, t)
            
            # 计算从当前位置到终点的距离比例
            distance_to_end = np.linalg.norm(np.array(curve_point) - np.array(end))
            total_distance = np.linalg.norm(np.array(start) - np.array(end))
            
            # 如果没有距离，则返回中点
            if total_distance == 0:
                return ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            
            # 计算当前点到终点的比例
            current_ratio = distance_to_end / total_distance
            
            # 调整t值
            if abs(current_ratio - endpoint_ratio) < precision:
                break
            elif current_ratio > endpoint_ratio:
                min_t = t
            else:
                max_t = t
                
            # 如果接近端点，提前退出
            if t < 0.01 or t > 0.99:
                break
        
        return self._calculate_curve_point(start, end, rad, t)

    def _create_fruchterman_layout(self, G, k=None):
        """创建力导向布局，带动态k值调整"""
        # 根据节点数量动态调整k值防止重叠
        n = len(G.nodes())
        if k is None:
            k = 0.1 * math.sqrt(1/n) if n > 20 else 0.1
        
        pos = nx.spring_layout(G, k=k, iterations=200, seed=42, pos=self.pos)
        return self._apply_min_distance_constraint(pos, G)
    



    def _draw_connections(self, G):
        """绘制连接线，处理同一层内的弯曲连接，并在距离末端0.38处添加标签"""
        # 计算平均层间距
        all_y = [pos[1] for pos in self.pos.values()]
        layer_spacing = np.max(all_y) - np.min(all_y) / len(self.pos) * 0.5
        
        for u, v in G.edges():
            u_pos = self.pos[u]
            v_pos = self.pos[v]
            label = G.edges[u, v].get('label', '')  # 获取连接标签
            gate_status = G.edges[u, v].get('gate_status', None)
            
            # 检查是否是同一层内的连接
            same_layer = abs(u_pos[1] - v_pos[1]) < layer_spacing
            
            # 如果是同一层内连接且水平距离较短
            horizontal_distance = abs(u_pos[0] - v_pos[0])
            if same_layer and horizontal_distance < 0.09:
                # 创建弯曲连接线
                rad_value = 0.1 if u_pos[1] > v_pos[1] else -0.1
                connection = ConnectionPatch(
                    xyA=u_pos, 
                    xyB=v_pos,
                    coordsA="data", 
                    coordsB="data",
                    arrowstyle="->", 
                    shrinkA=15, 
                    shrinkB=15,
                    mutation_scale=15,
                    linestyle="-",
                    color=self.conn_color_map[gate_status],
                    alpha=0.8,
                    connectionstyle=f"arc3,rad={rad_value}"
                )
                self.ax.add_patch(connection)
                
                # 计算精确的标签位置
                label_pos = self._calculate_label_position(
                    u_pos, v_pos, -rad_value, endpoint_ratio=1-0.62
                )
            else:
                # 绘制带箭头的直线连接
                connection = ConnectionPatch(
                    xyA=u_pos, 
                    xyB=v_pos,
                    coordsA="data", 
                    coordsB="data",
                    arrowstyle="->",
                    shrinkA=15,
                    shrinkB=15,
                    mutation_scale=15,
                    linestyle="-",
                    color=self.conn_color_map[gate_status],
                    alpha=0.8,
                    connectionstyle="arc3,rad=0"  # 0弧度表示直线
                )
                self.ax.add_patch(connection)
                
                # 计算直线连接的标签位置（距离末端0.38）
                label_pos = self._calculate_label_position(
                    u_pos, v_pos, endpoint_ratio=0.62
                )
            
            # 添加连接标签
            if label:
                self.ax.text(
                    label_pos[0], label_pos[1], 
                    label,
                    horizontalalignment='center',
                    verticalalignment='center',
                    fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.6),
                    zorder=10  # 确保标签在最上层
                )
    
    def update_network(self, cells: List[Cell], conns: List[Connection], active_cells: List[Cell]):
        """更新神经网络可视化"""
        # 清除当前图形内容但保留布局
        self.ax.clear()
        self.ax.set_title('Neural Network Visualization', fontsize=14)
        self.ax.axis('off')
        
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点（细胞）到图中
        active_ids = {cell.get_cell_id() for cell in active_cells}
        for cell in cells:
            node_type = "receptor" if cell.isReceptor else "actuator" if cell.isActuator else "neuron"
            is_active = cell.get_cell_id() in active_ids
            G.add_node(cell.get_cell_id(), 
                       node_type=node_type,
                       is_active=is_active,
                       label=str(cell.get_cell_id()))
        
        # 创建虚拟节点集合并添加
        virtual_nodes = set()
        for conn in conns:
            # 处理上游虚拟节点
            if conn.upstream_cell_id is None:
                virtual_id = f"virtual_up_{conn.get_conn_id()}"
                virtual_nodes.add(virtual_id)
                G.add_node(virtual_id, 
                           node_type="virtual_up", 
                           label=f"⥣{conn.get_conn_id()}-sg:{conn.constraints.get('signal_type', None)} t:{conn.constraints.get('ts', None)} k:{conn.constraints.get('k', None)} b:{conn.constraints.get('b', None)}",
                           is_active=False)
            
            # 处理下游虚拟节点
            if conn.downstream_cell_id is None:
                virtual_id = f"virtual_down_{conn.get_conn_id()}"
                virtual_nodes.add(virtual_id)
                G.add_node(virtual_id, 
                           node_type="virtual_down", 
                           label=f"⥥{conn.get_conn_id()}-sg:{conn.constraints.get('signal_type', None)} t:{conn.constraints.get('ts', None)} k:{conn.constraints.get('k', None)} b:{conn.constraints.get('b', None)}",
                           is_active=False)
        
        # 添加连接到图中
        for conn in conns:
            upstream = conn.upstream_cell_id or f"virtual_up_{conn.get_conn_id()}"
            downstream = conn.downstream_cell_id or f"virtual_down_{conn.get_conn_id()}"
            G.add_edge(upstream, downstream, 
                       conn_id=conn.get_conn_id(),
                       label=f"{conn.get_conn_id()}-sg:{conn.constraints.get('signal_type', None)} t:{conn.constraints.get('ts', None)} k:{conn.constraints.get('k', None)} b:{conn.constraints.get('b', None)}",
                        gate_status = conn.status.get(ConnectionStatus.GATE_STATUS.value,None)
                       )
        
        # 创建布局 - 根据节点数量选择合适的布局算法
        node_count = len(G.nodes)
        
        # 根据节点数量动态调整节点大小
        base_size = 300
        if node_count > 200:
            base_size = 100
        elif node_count > 100:
            base_size = 150
        elif node_count > 50:
            base_size = 200
            
        # 选择布局策略
        if node_count > 50:
            # 极大规模网络：使用简单的分层布局
            self.pos = self._create_layered_layout(G)
        elif node_count > 20:
            # 中等规模：使用力导向布局
            self.pos = self._create_fruchterman_layout(G)
        else:
            # 小规模：使用力导向布局保持美观
            self.pos = self._create_fruchterman_layout(G, k=0.1)

        # 应用最小距离约束
        # self.pos = self._apply_min_distance_constraint(self.pos, G)
        
        # 准备节点颜色和大小
        node_colors = []
        node_sizes = []
        node_shapes = []
        for node, data in G.nodes(data=True):
            node_colors.append(self.colors[data["node_type"]])
            node_shapes.append(self.node_styles[data["node_type"]])
            # 活跃节点增大
            size = 3 * base_size if data["is_active"] else base_size
            node_sizes.append(size)
        
        # 为不同形状的节点分别绘制
        for shape in set(node_shapes):
            node_list = [node for node in G.nodes if G.nodes[node]["node_type"] == 
                        {v: k for k, v in self.node_styles.items()}[shape]]
            
            nx.draw_networkx_nodes(
                G, self.pos,
                ax=self.ax,
                nodelist=node_list,
                node_shape=shape,
                node_size=[size for i, size in enumerate(node_sizes) if node_shapes[i] == shape],
                node_color=[color for i, color in enumerate(node_colors) if node_shapes[i] == shape],
                alpha=0.9,
                edgecolors='k',
                linewidths=2
            )
        
        # 绘制活跃节点的纹理效果（二次绘制）
        active_nodes = [node for node, data in G.nodes(data=True) if data["is_active"]]
        if active_nodes:
            nx.draw_networkx_nodes(
                G, self.pos,
                ax=self.ax,
                nodelist=active_nodes,
                node_shape="o",
                node_size=base_size * 0.5,
                node_color="white",
                alpha=0.4
            )
        
        # 绘制节点标签 - 仅在节点数量可控时显示
        if node_count <= 200:
            for node, (x, y) in self.pos.items():
                if isinstance(node, str) and "virtual" in node:
                    offset = (0, 0.015)
                else:
                    offset = (0, 0)
                self.ax.text(x + offset[0], y + offset[1], 
                        G.nodes[node]["label"],
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=8 if node_count > 20 else 9,
                        fontweight='bold')
        
        # 绘制连接
        # nx.draw_networkx_edges(
        #     G, self.pos,
        #     ax=self.ax,
        #     edge_color="#7F7F7F",
        #     width=1.0,
        #     arrowsize=12,
        #     arrowstyle='->'
        # )
        self._draw_connections(G)
        
        # 仅在节点数量少时显示连接标签
        if node_count <= 50:
            edge_labels = nx.get_edge_attributes(G, 'label')
            for (u, v), label in edge_labels.items():
                x = (self.pos[u][0] + self.pos[v][0]) / 2
                y = (self.pos[u][1] + self.pos[v][1]) / 2
                self.ax.text(x, y, label,
                        horizontalalignment='center',
                        verticalalignment='center',
                        fontsize=7,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey", alpha=0.6))
        
        # 重绘图形
        plt.draw()
        plt.pause(0.001)  # 允许图形界面更新

# 示例用法
if __name__ == "__main__":
    # 创建可视化器实例
    visualizer = NeuralNetworkVisualizer()
    
    # 初始网络状态
    def create_initial_network():
        cells = [
            Cell(1, False, True),   # 受体细胞
            Cell(2, True, False),   # 效应器细胞
            Cell(3, False, False),  # 普通神经元
            Cell(4, False, False),  # 普通神经元
            Cell(5, False, False),  # 普通神经元
        ]
        
        conns = [
            Connection(101, None, 1),           # 输入到受体
            Connection(102, 1, 3),              # 受体到神经元
            Connection(103, 3, 4),              # 神经元到神经元
            Connection(104, 4, 5),              # 神经元到神经元
            Connection(105, 5, 2),              # 神经元到效应器
            Connection(106, 2, None),           # 效应器到输出
            Connection(107, 3, 5),              # 跳过连接
        ]
        
        active_cells = [cells[2], cells[4]]  # 激活神经元3和5
        return cells, conns, active_cells
    
    # 更新网络状态
    def create_updated_network():
        cells = [
            Cell(1, False, True),
            Cell(2, True, False),
            Cell(3, False, False),
            Cell(4, False, False),
            Cell(5, False, False),
            Cell(6, False, False),  # 新增神经元
        ]
        
        conns = [
            Connection(101, None, 1),
            Connection(102, 1, 3),
            Connection(103, 3, 4),
            Connection(104, 4, 5),
            Connection(105, 5, 2),
            Connection(106, 2, None),
            Connection(107, 3, 5),
            Connection(108, 5, 6),  # 新增连接
            Connection(109, 6, 2),  # 新增连接
        ]
        
        active_cells = [cells[3], cells[5]]  # 激活神经元4和6
        return cells, conns, active_cells
    
    # 绘制初始网络
    initial_data = create_initial_network()
    visualizer.update_network(*initial_data)
    
    # 输入回车键更新网络
    input("按Enter键更新网络...")
    
    # 更新网络
    updated_data = create_updated_network()
    visualizer.update_network(*updated_data)
    
    # 保持窗口打开
    plt.show()