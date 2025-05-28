import numpy as np

class ProbNode:
    """概率节点，用于存储不参与划分的点"""
    def __init__(self):
        self.points = []       # 参与划分的点
        self.prob_points = []  # 不参与划分的点
        self.score = 0.0       # 总分数

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
        # 获取各维度使用次数
        min_count = min(self.axis_counts)
        candidates = [i for i in range(self.dimensions) if self.axis_counts[i] == min_count]
        
        # 计算候选维度的方差
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
            # 转换概率节点为划分节点
            all_points = node.points + [point]
            
            # 选择划分维度和值
            selected_axis = self._select_axis(all_points)
            median = self._median_value(all_points, selected_axis)
            
            # 更新轴使用计数
            self.axis_counts[selected_axis] += 1
            
            # 创建新划分节点
            split_node = SplitNode(selected_axis, median)
            
            # 初始化左右子树
            left = ProbNode()
            right = ProbNode()
            
            # 分配原有参与划分的点
            for p in all_points:
                if p[selected_axis] <= median:
                    left.add_point(p, self.score_func, True)
                else:
                    right.add_point(p, self.score_func, True)
            
            # 分配原有概率点
            for p in node.prob_points:
                score = self.score_func(p)
                if p[selected_axis] <= median:
                    split_node.score_left += score
                else:
                    split_node.score_right += score
                split_node.prob_points.append(p)
            
            # 设置子树并更新分数
            split_node.left = left
            split_node.right = right
            split_node.score_left += left.get_total_score()
            split_node.score_right += right.get_total_score()
            return split_node
        
        elif isinstance(node, SplitNode):
            # 递归插入子树
            if point[node.axis] <= node.value:
                node.left = self._insert(point, node.left)
            else:
                node.right = self._insert(point, node.right)
            
            # 更新当前节点分数
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
        
        # 寻找最后一个划分节点
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
        else:  # 只有根节点且是概率节点的情况
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

    def print_tree(self, node=None, indent=0):
        """树形结构打印（用于调试）"""
        node = node or self.root
        prefix = "    " * indent
        
        if isinstance(node, ProbNode):
            print(f"{prefix}ProbNode(score={node.score}, points={node.points}, prob_points={node.prob_points})")
        else:
            print(f"{prefix}SplitNode(axis={node.axis}, value={node.value:.2f}, total_score={node.get_total_score()}, probs={(self.get_probabilities(node))}, prob_points={node.prob_points}, left_score={node.score_left}, right_score={node.score_right})")
            print(f"{prefix}Left:")
            self.print_tree(node.left, indent+1)
            print(f"{prefix}Right:")
            self.print_tree(node.right, indent+1)
        
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

if __name__=='__main__':
    import time
    # 定义评分函数（简单计数）
    def score_func(point):
        return np.sum(point)

    # 创建3维KD树
    start_time = time.time()
    kdt = ProbabilityKDTree(5, score_func)

    # 插入参与划分的点
    # 生成一个随机的50维数组，每个元素在[0,1)之间
    points = [np.random.rand(5) for i in range(10)]
    # points = [
    #     [1.2, 2.3, 3.4],
    #     [1.5, 2.6, 3.0],
    #     [1.5, 2.6, 3.0],
    #     [1.5, 2.6, 3.0],
    #     [1.9, 2.9, 3.9],
    #     [4.5, 5.6, 6.7],
    #     [2.1, 3.2, 4.3],
    #     [5.4, 6.5, 7.6]
    # ]
    for p in points:
        kdt.insert_split_point(p)
        kdt.print_tree()
        print("\n\n\n")
        
    end_time = time.time()
    print(f"执行时间: {end_time - start_time:.6f} 秒")
    

    # 添加概率点
    # prob_points = [
    #     [1.5, 2.5, 3.5],
    #     [4.0, 5.0, 6.0]
    # ]
    # for p in prob_points:
    #     kdt.add_prob_point(p)

    # 查看概率分布
    print("Root probabilities:", kdt.get_probabilities())

    # 打印树结构
    kdt.print_tree()
    
    # 测试pdf
    test_point = np.random.rand(384)
    # test_point = [1.5, 2.5, 3.5]
        # [4.0, 5.0, 6.0],
        # [2.1, 3.2, 4.3]


    # 获取概率分布
    probabilities = kdt.query_probability(test_point)

    # 打印结果
    print("Probabilities for each point:")
    print(probabilities)