import numpy as np

class ProbNode:
    """概率节点，用于存储不参与划分的点"""
    def __init__(self):
        self.points = []       # 参与划分的点
        self.prob_points = []  # 不参与划分的点
        self.score = 0.0       # 总分数
        self.visit_count = 0   # 访问次数

    def add_point(self, point, score_func, is_split_point=False):
        """添加点到当前节点"""
        if is_split_point:
            self.points.append(point)
        else:
            self.prob_points.append(point)
        self.score += score_func(point)
    
    def get_total_score(self):
        return self.score

class SplitNode:
    """划分节点，负责空间划分"""
    def __init__(self, axis, value):
        self.axis = axis          # 划分维度
        self.value = value        # 划分值
        self.left = None          # 左子树
        self.right = None         # 右子树
        self.score_left = 0.0     # 左侧总分数
        self.score_right = 0.0    # 右侧总分数
        self.prob_points = []     # 当前节点关联的概率点
        self.visit_count = 0      # 访问次数

    def add_prob_point(self, point, score):
        """添加概率点并更新分数"""
        if point[self.axis] <= self.value:
            self.score_left += score
        else:
            self.score_right += score
        self.prob_points.append(point)
    
    def get_total_score(self):
        return self.score_left + self.score_right

class ProbabilityKDTree:
    def __init__(self, dimensions, score_func):
        self.dimensions = dimensions    # 空间维度
        self.score_func = score_func     # 评分函数
        self.root = None                 # 根节点
        self.axis_counts = [0] * dimensions  # 各维度划分次数统计
    
    def _select_axis(self, points):
        """选择划分维度：最少使用且方差最大的维度"""
        min_count = min(self.axis_counts)
        candidates = [i for i in range(self.dimensions) if self.axis_counts[i] == min_count]
        
        max_var = -1
        selected_axis = candidates[0]
        for axis in candidates:
            values = [p[axis] for p in points]
            var = np.var(values)
            if var > max_var or (var == max_var and axis < selected_axis):
                max_var = var
                selected_axis = axis
        return selected_axis
    
    def _median_value(self, points, axis):
        """计算指定维度的中位数"""
        values = sorted([p[axis] for p in points])
        n = len(values)
        return values[n//2] if n % 2 == 1 else (values[n//2-1] + values[n//2])/2

    def insert_split_point(self, point):
        """插入参与划分的点"""
        if not self.root:
            self.root = ProbNode()
            self.root.add_point(point, self.score_func, True)
        else:
            self.root = self._insert(point, self.root)

    def _insert(self, point, node):
        if isinstance(node, ProbNode):
            all_points = node.points + [point]
            selected_axis = self._select_axis(all_points)
            median = self._median_value(all_points, selected_axis)
            self.axis_counts[selected_axis] += 1
            
            split_node = SplitNode(selected_axis, median)
            left = ProbNode()
            right = ProbNode()
            
            for p in all_points:
                if p[selected_axis] <= median:
                    left.add_point(p, self.score_func, True)
                else:
                    right.add_point(p, self.score_func, True)
            
            for p in node.prob_points:
                score = self.score_func(p)
                if p[selected_axis] <= median:
                    split_node.score_left += score
                else:
                    split_node.score_right += score
                split_node.prob_points.append(p)
            
            split_node.left = left
            split_node.right = right
            split_node.score_left += left.get_total_score()
            split_node.score_right += right.get_total_score()
            return split_node
        
        elif isinstance(node, SplitNode):
            if point[node.axis] <= node.value:
                node.left = self._insert(point, node.left)
            else:
                node.right = self._insert(point, node.right)
            
            node.score_left = node.left.get_total_score() if node.left else 0
            node.score_right = node.right.get_total_score() if node.right else 0
            return node


    def add_prob_point(self, point):
        """添加概率点"""
        score = self.score_func(point)
        if not self.root:
            self.root = ProbNode()
            self.root.add_point(point, self.score_func)
            return
        
        current = self.root
        last_split = None
        
        while True:
            if isinstance(current, SplitNode):
                last_split = current
                if point[current.axis] <= current.value:
                    current = current.left
                else:
                    current = current.right
            else:
                break
        
        if last_split:
            last_split.add_prob_point(point, score)
        else:
            self.root.add_point(point, self.score_func)


    def get_probabilities(self, node=None):
        """获取当前节点的划分概率"""
        node = node or self.root
        if isinstance(node, ProbNode):
            return (1.0, 0.0) if node.get_total_score() else (0.0, 0.0)
        
        total = node.get_total_score()
        if total == 0:
            return (0.5, 0.5)
        return (node.score_left/total, node.score_right/total)


    def expand(self, C=1.0):
        """根据UCT算法扩展节点，返回划分条件和区域概率"""
        if not self.root:
            return {}, 0.0
        
        current_node = self.root
        conditions = {}
        parent_stack = []
        
        # UCT选择路径
        while True:
            current_node.visit_count += 1
            
            if isinstance(current_node, ProbNode):
                break
            
            if isinstance(current_node, SplitNode):
                left = current_node.left
                right = current_node.right
                
                # 计算UCT值
                left_uct = (left.get_total_score() / left.visit_count) + C * np.sqrt(np.log(current_node.visit_count) / left.visit_count) if left.visit_count != 0 else float('inf')
                right_uct = (right.get_total_score() / right.visit_count) + C * np.sqrt(np.log(current_node.visit_count) / right.visit_count) if right.visit_count != 0 else float('inf')
                
                # 选择子节点
                if left_uct >= right_uct:
                    selected = left
                    direction = 'l'
                else:
                    selected = right
                    direction = 'r'
                
                # 记录划分条件
                conditions[current_node.axis] = (current_node.value, direction)
                parent_stack.append( (current_node, direction) )
                current_node = selected
                
        
        # 处理ProbNode扩展
        if not current_node.points:
            return conditions, 0.0
        
        selected_axis = self._select_axis(current_node.points)
        median = self._median_value(current_node.points, selected_axis)
        
        # 创建新划分节点
        new_split = SplitNode(selected_axis, median)
        self.axis_counts[selected_axis] += 1
        
        # 初始化左右子节点
        left_node = ProbNode()
        right_node = ProbNode()
        for p in current_node.points:
            if p[selected_axis] <= median:
                left_node.add_point(p, self.score_func, True)
            else:
                right_node.add_point(p, self.score_func, True)
        
        # 处理概率点
        for p in current_node.prob_points:
            score = self.score_func(p)
            if p[selected_axis] <= median:
                new_split.score_left += score
            else:
                new_split.score_right += score
            new_split.prob_points.append(p)
        
        new_split.left = left_node
        new_split.right = right_node
        new_split.score_left += left_node.get_total_score()
        new_split.score_right += right_node.get_total_score()
        
        # 更新树结构
        if parent_stack:
            parent, direction = parent_stack[-1]
            if direction == 'l':
                parent.left = new_split
            else:
                parent.right = new_split
        else:
            self.root = new_split
        
        # 计算路径概率
        total_prob = 1.0
        for (node, dir) in parent_stack:
            total = node.get_total_score()
            if total == 0:
                prob = 0.5
            else:
                prob = (node.score_left / total) if dir == 'l' else (node.score_right / total)
            total_prob *= prob
        
        return conditions, total_prob

        
    def query_probability(self, point):
        """
        查询单个点的概率
        """
        if self.root is None:
            return 0.0
        return self._query_node(self.root, point)


    def _query_node(self, node, point):
        """
        递归遍历树，根据输入点返回概率乘积
        """
        # 如果为叶节点，则返回 1.0（认为该叶子区域没有再进一步的概率衰减）
        if isinstance(node, ProbNode):
            # 可以根据需求调整：例如，若没有得分，则返回 0
            return 1.0 if node.score > 0 else 0.0

        # 如果为划分节点，则首先获取该节点左右分支的概率比率
        p_left, p_right = self.get_probabilities(node)
        # 根据输入点在当前划分维度上的取值决定进入左/右子树，并将分支概率乘入
        if point[node.axis] <= node.value:
            # 注意这里假设 node.left 非空，否则需要做相应处理
            return p_left * self._query_node(node.left, point)
        else:
            return p_right * self._query_node(node.right, point)
        
        
    def _calc_splitnode_uct(self, node: SplitNode, father_node: SplitNode, C:str=1):
        node_uct = (node.get_total_score() / node.visit_count) + C * np.sqrt(np.log(father_node.visit_count) / node.visit_count) if node.visit_count != 0 else float('inf')
        return node_uct
    
    
    def print_tree(self, node=None, father_node=None, indent=0):
        """树形结构打印（用于调试）"""
        node = node or self.root
        prefix = "    " * indent
        
        if isinstance(node, ProbNode):
            print(f"{prefix}ProbNode(score={node.score}, points={node.points}, prob_points={node.prob_points})")
        else:
            if node == self.root or father_node == None:
                uct = None
            else:
                uct = self._calc_splitnode_uct(node, father_node, C=1)
            
            print(f"{prefix}SplitNode(axis={node.axis}, value={node.value:.2f}, total_score={node.get_total_score()}, probs={(self.get_probabilities(node))}, uct_score={uct}, visit_times={node.visit_count}, prob_points={node.prob_points}, left_score={node.score_left}, right_score={node.score_right})")
            
            print(f"{prefix}Left:")
            self.print_tree(node.left, node, indent+1)
            print(f"{prefix}Right:")
            self.print_tree(node.right, node, indent+1)
            
    def enhanced_print_tree(self, node=None, father_node=None, indent=0, visualize=True):
        """
        增强的树打印函数，同时支持文本打印和可视化
        
        参数:
            node: 当前节点
            father_node: 父节点
            indent: 缩进级别
            visualize: 是否同时显示图形可视化
        """
        # 文本打印部分
        node = node or self.root
        prefix = "    " * indent
        
        if isinstance(node, ProbNode):
            print(f"{prefix}ProbNode(score={node.score}, points={node.points}, prob_points={node.prob_points})")
        else:
            if node == self.root or father_node == None:
                uct = None
            else:
                uct = self._calc_splitnode_uct(node, father_node, C=1)
            
            print(f"{prefix}SplitNode(axis={node.axis}, value={node.value:.2f}, total_score={node.get_total_score()}, probs={(self.get_probabilities(node))}, uct_score={uct}, visit_times={node.visit_count}, prob_points={node.prob_points}, left_score={node.score_left}, right_score={node.score_right})")
            
            print(f"{prefix}Left:")
            self.enhanced_print_tree(node.left, node, indent+1, visualize=False)
            print(f"{prefix}Right:")
            self.enhanced_print_tree(node.right, node, indent+1, visualize=False)
        
        # 当递归到顶层并且visualize=True时，显示可视化
        if indent == 0 and visualize:
            print("Visualized...")
            visualize_tree(self)
            

    
    ###'''''''''''''###
    '''Visualization'''
    ###'''''''''''''###
    
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import FancyBboxPatch

class TreeVisualizer:
    def __init__(self, tree, figsize=(12, 8), node_size=1800, font_size=8):
        self.tree = tree  # 树对象
        self.G = nx.DiGraph()
        self.pos = {}
        self.node_count = 0
        self.node_types = {}  # 存储节点类型
        self.node_data = {}   # 存储节点数据
        self.figsize = figsize
        self.node_size = node_size
        self.font_size = font_size
    
    def _prepare_tree(self, node=None, father_node=None, x=0, y=0, level=0, horizontal_spacing=1):
        """递归遍历树，准备可视化数据"""
        if node is None:
            node = self.tree.root
        
        # 分配ID和存储节点类型
        node_id = self.node_count
        self.node_count += 1
        
        # 存储节点类型和数据
        if isinstance(node, ProbNode):
            self.node_types[node_id] = "ProbNode"
            
            # 格式化ProbNode数据
            data = {
                "score": round(node.score, 2) if hasattr(node, 'score') else "N/A",
                "points": len(node.points) if hasattr(node, 'points') else "N/A",
                "prob_points": len(node.prob_points) if hasattr(node, 'prob_points') else "N/A"
            }
        else:
            self.node_types[node_id] = "SplitNode"
            
            # 计算UCT分数
            if node == self.tree.root or father_node is None:
                uct = None
            else:
                uct = self.tree._calc_splitnode_uct(node, father_node, C=1)
            
            # 获取概率
            probs = self.tree.get_probabilities(node) if hasattr(self.tree, 'get_probabilities') else "N/A"
            
            # 格式化SplitNode数据
            data = {
                "axis": node.axis if hasattr(node, 'axis') else "N/A",
                "value": round(node.value, 2) if hasattr(node, 'value') else "N/A",
                "total_score": round(node.get_total_score(), 2) if hasattr(node, 'get_total_score') else "N/A",
                "probs": probs,
                "uct_score": round(uct, 2) if uct is not None else "N/A",
                "visit_times": node.visit_count if hasattr(node, 'visit_count') else "N/A",
                "prob_points": len(node.prob_points) if hasattr(node, 'prob_points') else "N/A",
                "left_score": round(node.score_left, 2) if hasattr(node, 'score_left') else "N/A",
                "right_score": round(node.score_right, 2) if hasattr(node, 'score_right') else "N/A"
            }
        
        self.node_data[node_id] = data
        
        # 存储节点位置
        self.pos[node_id] = (x, -level)
        
        # 添加节点到图
        self.G.add_node(node_id)
        
        # 递归处理子节点
        if hasattr(node, 'left') and node.left is not None:
            left_id = self.node_count
            left_width = self._prepare_tree(node.left, node, x-horizontal_spacing, y-1, level+1, horizontal_spacing/2)
            self.G.add_edge(node_id, left_id, side="left")
        
        if hasattr(node, 'right') and node.right is not None:
            right_id = self.node_count
            right_width = self._prepare_tree(node.right, node, x+horizontal_spacing, y-1, level+1, horizontal_spacing/2)
            self.G.add_edge(node_id, right_id, side="right")
        
        return 1
    
    def visualize(self):
        """以友好的方式可视化树结构"""
        # 准备树数据
        self._prepare_tree()
        
        # 创建图形
        plt.figure(figsize=self.figsize)
        ax = plt.gca()
        
        # 定义颜色
        prob_node_color = "#4CAF50"  # 绿色
        split_node_color = "#2196F3"  # 蓝色
        left_edge_color = "#FF9800"   # 橙色
        right_edge_color = "#9C27B0"  # 紫色
        
        # 绘制边
        for u, v, data in self.G.edges(data=True):
            edge_color = left_edge_color if data.get('side') == "left" else right_edge_color
            side_label = "L" if data.get('side') == "left" else "R"
            
            # 画边
            ax.annotate("",
                xy=self.pos[v], xycoords='data',
                xytext=self.pos[u], textcoords='data',
                arrowprops=dict(arrowstyle="-|>", 
                                color=edge_color, 
                                lw=1.5, 
                                alpha=0.7,
                                connectionstyle="arc3,rad=0.1"))
            
            # 添加边标签
            mid_x = (self.pos[u][0] + self.pos[v][0]) / 2
            mid_y = (self.pos[u][1] + self.pos[v][1]) / 2
            # 稍微偏移以避免与边重叠
            offset = 0.1
            if data.get('side') == "left":
                mid_x -= offset
            else:
                mid_x += offset
            plt.text(mid_x, mid_y, side_label, fontsize=10, 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.1'),
                    ha='center', va='center')
        
        # 绘制节点
        for node_id in self.G.nodes():
            x, y = self.pos[node_id]
            node_type = self.node_types[node_id]
            node_color = prob_node_color if node_type == "ProbNode" else split_node_color
            
            # 绘制节点圆形
            circle = plt.Circle((x, y), 0.2, color=node_color, alpha=0.7)
            ax.add_patch(circle)
            
            # 准备节点标签文本
            data = self.node_data[node_id]
            if node_type == "ProbNode":
                label = f"ProbNode\nscore={data['score']}\npoints={data['points']}\nprob_points={data['prob_points']}"
            else:
                # 对于SplitNode，创建更紧凑的标签
                label = f"SplitNode\naxis={data['axis']}, val={data['value']}\n"
                label += f"total={data['total_score']}, uct={data['uct_score']}\n"
                label += f"visits={data['visit_times']}, points={data['prob_points']}\n"
                label += f"L:{data['left_score']}, R:{data['right_score']}"
            
            # 给节点添加文本框
            text_box = ax.text(x, y-0.35, label, 
                             ha='center', va='top', 
                             fontsize=self.font_size,
                             bbox=dict(boxstyle="round,pad=0.3", 
                                      facecolor='white', 
                                      alpha=0.8,
                                      edgecolor=node_color))
            
        # 调整布局
        plt.axis('off')
        # plt.tight_layout()
        
        # 计算适当的坐标范围
        x_values = [x for x, y in self.pos.values()]
        y_values = [y for x, y in self.pos.values()]
        
        if x_values and y_values:  # 确保有节点
            x_margin = (max(x_values) - min(x_values)) * 0.2
            y_margin = (max(y_values) - min(y_values)) * 0.2
            plt.xlim(min(x_values) - x_margin, max(x_values) + x_margin)
            plt.ylim(min(y_values) - y_margin - 1, max(y_values) + y_margin)
        
        plt.title("tree visualization", fontsize=16)
        plt.show()
    
    def save(self, filename="tree_visualization.png", dpi=300):
        """保存可视化结果为图片文件"""
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"图像已保存为 {filename}")


# 示例用法
def visualize_tree(tree):
    """直接使用此函数可视化树"""
    visualizer = TreeVisualizer(tree)
    visualizer.visualize()
    # 如果需要保存图像，取消下一行的注释
    visualizer.save("my_tree.png")


# 替代原来的print_tree方法，可以同时支持文本打印和可视化
def enhanced_print_tree(self, node=None, father_node=None, indent=0, visualize=True):
    """
    增强的树打印函数，同时支持文本打印和可视化
    
    参数:
        node: 当前节点
        father_node: 父节点
        indent: 缩进级别
        visualize: 是否同时显示图形可视化
    """
    # 文本打印部分
    node = node or self.root
    prefix = "    " * indent
    
    if isinstance(node, ProbNode):
        print(f"{prefix}ProbNode(score={node.score}, points={node.points}, prob_points={node.prob_points})")
    else:
        if node == self.root or father_node == None:
            uct = None
        else:
            uct = self._calc_splitnode_uct(node, father_node, C=1)
        
        print(f"{prefix}SplitNode(axis={node.axis}, value={node.value:.2f}, total_score={node.get_total_score()}, probs={(self.get_probabilities(node))}, uct_score={uct}, visit_times={node.visit_count}, prob_points={node.prob_points}, left_score={node.score_left}, right_score={node.score_right})")
        
        print(f"{prefix}Left:")
        self.print_tree(node.left, node, indent+1, visualize=False)
        print(f"{prefix}Right:")
        self.print_tree(node.right, node, indent+1, visualize=False)
    
    # 当递归到顶层并且visualize=True时，显示可视化
    if indent == 0 and visualize:
        visualize_tree(self)

# 用这个方法替换原来的print_tree方法
# self.print_tree = enhanced_print_tree.__get__(self)

def judge_vector_with_condition(vector, condition:dict):
    for i, (dim, threshold) in enumerate(conditions.items()):
        if threshold[1] == 'l':
            distance = threshold[0] - new_point[dim]
            if distance < 0:
                return False
        elif threshold[1] == 'r':
            distance =  new_point[dim] - threshold[0]
            if distance <= 0:
                return False
    
    return True

if __name__=='__main__':
    import time
    # 创建KD树并扩展节点
    kdt = ProbabilityKDTree(3, lambda x: sum(x))
    points = [np.random.rand(5) for i in range(10)]
    for p in points:
        kdt.insert_split_point(p)
    conditions, prob = kdt.expand(C=1)
    print("扩展条件:", conditions)
    print("区域概率:", prob)
    
    find_flag = 0
    for _ in range(10):
        new_point = np.random.rand(5)
        for i, (dim, threshold) in enumerate(conditions.items()):
            if threshold[1] == 'l':
                distance = threshold[0] - new_point[dim]
            elif threshold[1] == 'r':
                distance =  new_point[dim] - threshold[0]
            if distance < 0:
                break
            else:
                find_flag=1
        if find_flag == 1:
            break
        else:
            new_point = None
            
    print(new_point)
    # kdt.insert_split_point(new_point)
    kdt.enhanced_print_tree()
        
        