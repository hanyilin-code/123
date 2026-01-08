import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import warnings

# 添加中文字体支持
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 忽略特定警告
warnings.filterwarnings('ignore', category=UserWarning)


class KillChainNetwork:
    """杀伤链网络模型类"""

    def __init__(self, n_nodes=10, topology='star', seed=42):
        """
        初始化杀伤链网络
        """
        self.n_nodes = n_nodes
        self.topology = topology
        self.seed = seed
        np.random.seed(seed)

        # 创建图
        self.G = self._create_network()

        # 为节点分配权重
        self._assign_node_weights()

        # 初始所有节点正常
        self.node_states = {node: 1 for node in self.G.nodes()}

    def _create_network(self):
        """根据拓扑类型创建网络"""
        if self.topology == 'star':
            G = nx.star_graph(self.n_nodes - 1)
        elif self.topology == 'ring':
            G = nx.cycle_graph(self.n_nodes)
        elif self.topology == 'random':
            # 使用Erdős-Rényi模型，确保连通但不要太密集
            p = 0.15  # 降低连接概率，让网络更脆弱
            G = nx.erdos_renyi_graph(self.n_nodes, p, seed=self.seed)
            # 确保连通
            while not nx.is_connected(G):
                p += 0.05
                G = nx.erdos_renyi_graph(self.n_nodes, p, seed=self.seed + len(str(p)))
        else:
            raise ValueError(f"未知拓扑类型: {self.topology}")

        return G

    def _assign_node_weights(self):
        """为节点分配随机权重"""
        for node in self.G.nodes():
            self.G.nodes[node]['weight'] = np.random.randint(1, 11)

    def get_network_info(self):
        """获取网络基本信息"""
        info = {
            '节点数': self.G.number_of_nodes(),
            '边数': self.G.number_of_edges(),
            '平均度数': np.mean([d for _, d in self.G.degree()]),
            '拓扑类型': self.topology
        }
        return info

    def visualize_network(self, highlight_nodes=None, title=None, save_path=None):
        """可视化网络"""
        plt.figure(figsize=(10, 8))

        # 节点颜色
        node_colors = []
        for node in self.G.nodes():
            if highlight_nodes and node in highlight_nodes:
                node_colors.append('red')
            elif self.node_states[node] == 0:
                node_colors.append('gray')
            else:
                node_colors.append('lightgreen')

        # 节点大小根据权重调整
        node_sizes = [300 + 50 * self.G.nodes[node]['weight'] for node in self.G.nodes()]

        # 布局
        if self.topology == 'star':
            pos = nx.spring_layout(self.G, seed=self.seed)
        elif self.topology == 'ring':
            pos = nx.circular_layout(self.G)
        else:
            pos = nx.spring_layout(self.G, seed=self.seed)

        # 绘制
        nx.draw_networkx_nodes(self.G, pos, node_color=node_colors,
                               node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(self.G, pos, alpha=0.5)

        # 添加权重标签
        labels = {node: f"{node}\nW:{self.G.nodes[node]['weight']}"
                  for node in self.G.nodes()}
        nx.draw_networkx_labels(self.G, pos, labels=labels, font_size=8)

        if title:
            plt.title(title, fontsize=14)
        else:
            plt.title(f"杀伤链网络拓扑 ({self.topology})", fontsize=14)

        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            try:
                plt.show()
            except:
                plt.savefig(f'network_{self.topology}.png', dpi=300, bbox_inches='tight')
                plt.close()
                print(f"图像已保存为 network_{self.topology}.png")

    def calculate_connectivity_efficiency(self):
        """计算网络连通效能"""
        # 只考虑正常节点
        normal_nodes = [node for node in self.G.nodes() if self.node_states[node] == 1]

        if len(normal_nodes) <= 1:
            return 0.0

        # 创建只包含正常节点的子图
        H = nx.Graph()
        H.add_nodes_from(normal_nodes)

        # 只添加两个端点都正常的边
        for u, v in self.G.edges():
            if self.node_states[u] == 1 and self.node_states[v] == 1:
                H.add_edge(u, v)

        # 计算连通分量
        components = list(nx.connected_components(H))

        # 计算连通的节点对
        connected_pairs = 0
        for comp in components:
            size = len(comp)
            connected_pairs += size * (size - 1) / 2

        # 总可能的节点对
        total_pairs = len(normal_nodes) * (len(normal_nodes) - 1) / 2

        return connected_pairs / total_pairs if total_pairs > 0 else 0.0

    def simulate_attack(self, damage_ratio=0.3):
        """模拟攻击，随机损伤一定比例的节点"""
        all_nodes = list(self.G.nodes())
        n_damage = int(len(all_nodes) * damage_ratio)
        damaged_nodes = np.random.choice(all_nodes, size=n_damage, replace=False)

        for node in damaged_nodes:
            self.node_states[node] = 0

        # 转换为普通整数列表
        return [int(node) for node in damaged_nodes]

    def repair_nodes(self, nodes_to_repair):
        """修复指定节点"""
        for node in nodes_to_repair:
            self.node_states[node] = 1


class CriticalNodeAnalyzer:
    """关键节点分析器 - 修复版"""

    def __init__(self, network: KillChainNetwork):
        self.network = network
        self.criticality_scores = {}

    def calculate_criticality(self):
        """计算所有节点的关键性分数 - 修复版"""
        # 保存当前状态
        original_states = self.network.node_states.copy()

        # 先设置所有节点为正常，计算原始效能
        for node in self.network.G.nodes():
            self.network.node_states[node] = 1

        original_efficiency = self.network.calculate_connectivity_efficiency()

        # 对每个节点进行删除测试
        for node in self.network.G.nodes():
            # 设置该节点为受损
            self.network.node_states[node] = 0

            # 计算删除后的效能
            damaged_efficiency = self.network.calculate_connectivity_efficiency()

            # 计算关键性分数
            criticality = original_efficiency - damaged_efficiency

            # 存储结果
            self.criticality_scores[node] = {
                'criticality': criticality,
                'weight': self.network.G.nodes[node]['weight']
            }

            # 恢复该节点为正常
            self.network.node_states[node] = 1

        # 恢复网络的原始状态
        self.network.node_states = original_states

        return self.criticality_scores

    def get_top_critical_nodes(self, top_k=5):
        """获取关键性最高的top_k个节点"""
        if not self.criticality_scores:
            self.calculate_criticality()

        sorted_nodes = sorted(self.criticality_scores.items(),
                              key=lambda x: x[1]['criticality'],
                              reverse=True)
        return sorted_nodes[:top_k]

    def visualize_criticality(self, save_path=None):
        """可视化关键性分数"""
        if not self.criticality_scores:
            self.calculate_criticality()

        nodes = list(self.criticality_scores.keys())
        criticalities = [self.criticality_scores[node]['criticality'] for node in nodes]
        weights = [self.criticality_scores[node]['weight'] for node in nodes]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # 关键性分数条形图
        ax1.bar(range(len(nodes)), criticalities, color='skyblue', alpha=0.7)
        ax1.set_xlabel('节点ID')
        ax1.set_ylabel('关键性分数')
        ax1.set_title('节点关键性分数')
        ax1.set_xticks(range(len(nodes)))
        ax1.set_xticklabels([str(node) for node in nodes], rotation=45)
        ax1.grid(True, alpha=0.3)

        # 关键性与权重的散点图
        scatter = ax2.scatter(weights, criticalities, s=100, alpha=0.6,
                              c=criticalities, cmap='RdYlGn_r')
        ax2.set_xlabel('节点权重')
        ax2.set_ylabel('关键性分数')
        ax2.set_title('关键性分数 vs 节点权重')
        ax2.grid(True, alpha=0.3)

        plt.colorbar(scatter, ax=ax2, label='关键性分数')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            try:
                plt.show()
            except:
                plt.savefig('criticality_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("关键性分析图像已保存为 criticality_analysis.png")


def greedy_repair_schedule(damaged_nodes, criticality_scores, repair_budget):
    """贪心修复调度算法"""
    # 获取受损节点的关键性分数
    damaged_scores = {}
    for node in damaged_nodes:
        if node in criticality_scores:
            damaged_scores[node] = criticality_scores[node]['criticality']
        else:
            damaged_scores[node] = 0

    # 按关键性降序排序
    sorted_damaged = sorted(damaged_scores.items(),
                            key=lambda x: x[1],
                            reverse=True)

    # 选择前repair_budget个节点
    n_repair = min(repair_budget, len(sorted_damaged))
    repair_nodes = [node for node, _ in sorted_damaged[:n_repair]]

    return repair_nodes


def run_experiment(topology='star', n_nodes=10, damage_ratio=0.3, repair_ratio=0.5, save_figures=False):
    """运行完整实验"""
    print(f"\n{'=' * 60}")
    print(f"实验配置: {topology}网络, {n_nodes}个节点")
    print(f"损伤比例: {damage_ratio * 100}%, 修复比例: {repair_ratio * 100}%")
    print('=' * 60)

    # 1. 创建网络
    start_time = time.time()
    network = KillChainNetwork(n_nodes=n_nodes, topology=topology)
    network_info = network.get_network_info()

    print("\n1. 网络信息:")
    for key, value in network_info.items():
        print(f"   {key}: {value}")

    # 2. 可视化原始网络
    if save_figures:
        network.visualize_network(title=f"原始{topology}网络",
                                  save_path=f"原始网络_{topology}.png")
    else:
        network.visualize_network(title=f"原始{topology}网络")

    # 3. 计算原始网络效能
    # 先保存当前状态
    original_states = network.node_states.copy()
    for node in network.G.nodes():
        network.node_states[node] = 1
    original_efficiency = network.calculate_connectivity_efficiency()
    network.node_states = original_states  # 恢复状态

    print(f"\n2. 原始网络连通效能: {original_efficiency:.4f}")

    # 4. 模拟攻击
    damaged_nodes = network.simulate_attack(damage_ratio)
    damaged_efficiency = network.calculate_connectivity_efficiency()

    print(f"\n3. 攻击后状态:")
    print(f"   受损节点: {sorted(damaged_nodes)}")
    print(f"   受损后连通效能: {damaged_efficiency:.4f}")
    if original_efficiency > 0:
        print(f"   效能下降: {(original_efficiency - damaged_efficiency) / original_efficiency * 100:.2f}%")
    else:
        print(f"   效能下降: 0.00%")

    # 可视化受损网络
    if save_figures:
        network.visualize_network(highlight_nodes=damaged_nodes,
                                  title=f"受损后{topology}网络",
                                  save_path=f"受损网络_{topology}.png")
    else:
        network.visualize_network(highlight_nodes=damaged_nodes,
                                  title=f"受损后{topology}网络")

    # 5. 关键节点分析
    print("\n4. 关键节点分析:")
    analyzer = CriticalNodeAnalyzer(network)
    criticality_scores = analyzer.calculate_criticality()

    top_critical = analyzer.get_top_critical_nodes(top_k=5)
    print("   关键性最高的5个节点:")
    for node, scores in top_critical:
        print(f"     节点{node}: 关键性={scores['criticality']:.4f}, 权重={scores['weight']}")

    if save_figures:
        analyzer.visualize_criticality(save_path=f"关键性分析_{topology}.png")
    else:
        analyzer.visualize_criticality()

    # 6. 修复决策
    print("\n5. 修复决策:")
    repair_budget = int(len(damaged_nodes) * repair_ratio)
    print(f"   修复预算: {repair_budget}个节点")

    # 贪心策略
    greedy_repair_nodes = greedy_repair_schedule(damaged_nodes, criticality_scores, repair_budget)
    print(f"   贪心策略修复节点: {sorted(greedy_repair_nodes)}")

    # 7. 评估修复效果
    print("\n6. 修复效果评估:")

    # 测试贪心策略
    test_network = copy.deepcopy(network)
    test_network.repair_nodes(greedy_repair_nodes)
    repaired_efficiency = test_network.calculate_connectivity_efficiency()

    print(f"   修复后效能: {repaired_efficiency:.4f}")

    # 计算效能恢复比例（避免除零错误）
    if original_efficiency > damaged_efficiency:
        recovery_ratio = (repaired_efficiency - damaged_efficiency) / (original_efficiency - damaged_efficiency) * 100
        print(f"   效能恢复比例: {recovery_ratio:.2f}%")
    else:
        recovery_ratio = 0
        print(f"   效能恢复比例: N/A (原始效能未下降)")

    # 8. 可视化修复效果
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 受损网络
    damaged_pos = nx.spring_layout(network.G, seed=42)
    damaged_colors = ['gray' if node in damaged_nodes else 'lightgreen'
                      for node in network.G.nodes()]
    nx.draw_networkx(network.G, damaged_pos, node_color=damaged_colors,
                     node_size=300, ax=axes[0], alpha=0.8)
    axes[0].set_title(f"受损网络\n效能: {damaged_efficiency:.3f}")
    axes[0].axis('off')

    # 修复后网络
    repaired_colors = []
    for node in network.G.nodes():
        if node in damaged_nodes and node not in greedy_repair_nodes:
            repaired_colors.append('gray')  # 仍然受损
        else:
            repaired_colors.append('lightgreen')  # 正常（包括被修复的）

    nx.draw_networkx(test_network.G, damaged_pos, node_color=repaired_colors,
                     node_size=300, ax=axes[1], alpha=0.8)
    axes[1].set_title(f"修复后网络\n效能: {repaired_efficiency:.3f}\n修复节点: {sorted(greedy_repair_nodes)}")
    axes[1].axis('off')

    plt.suptitle(f"修复效果 ({topology}网络)", fontsize=14)
    plt.tight_layout()

    if save_figures:
        plt.savefig(f"修复效果_{topology}.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        try:
            plt.show()
        except:
            plt.savefig(f"修复效果_{topology}.png", dpi=300, bbox_inches='tight')
            plt.close()
            print(f"修复效果图已保存为 修复效果_{topology}.png")

    # 9. 性能分析
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\n7. 性能分析:")
    print(f"   总执行时间: {execution_time:.4f}秒")

    # 10. 算法复杂度验证
    print(f"\n8. 算法复杂度验证:")
    n = network.G.number_of_nodes()
    m = network.G.number_of_edges()
    print(f"   网络规模: n={n}, m={m}")
    print(f"   理论复杂度: O(n×(n+m)) = O({n}×({n}+{m}))")

    return {
        'topology': topology,
        'original_efficiency': original_efficiency,
        'damaged_efficiency': damaged_efficiency,
        'repaired_efficiency': repaired_efficiency,
        'execution_time': execution_time,
        'recovery_ratio': recovery_ratio,
        'n_nodes': n,
        'n_edges': m
    }


def scalability_experiment(save_figures=False):
    """可扩展性实验：测试算法在不同规模网络上的性能"""
    print("\n" + "=" * 60)
    print("可扩展性实验：不同网络规模下的算法性能")
    print("=" * 60)

    sizes = [10, 40, 70, 100, 130]
    results = []

    for size in sizes:
        print(f"\n测试网络规模: {size}个节点")

        # 创建随机网络
        network = KillChainNetwork(n_nodes=size, topology='random', seed=42)

        # 关键节点分析
        start_time = time.time()
        analyzer = CriticalNodeAnalyzer(network)
        analyzer.calculate_criticality()
        end_time = time.time()

        exec_time = end_time - start_time

        n = network.G.number_of_nodes()
        m = network.G.number_of_edges()

        results.append({
            'n_nodes': n,
            'n_edges': m,
            'time': exec_time,
            'complexity': n * (n + m)
        })

        print(f"  执行时间: {exec_time:.4f}秒")

    # 绘制性能曲线
    fig, ax1 = plt.subplots(figsize=(10, 6))

    n_nodes = [r['n_nodes'] for r in results]
    exec_times = [r['time'] for r in results]
    complexities = [r['complexity'] for r in results]

    # 归一化复杂度，便于比较
    if max(complexities) > 0 and max(exec_times) > 0:
        normalized_complexities = [c / max(complexities) * max(exec_times)
                                   for c in complexities]
    else:
        normalized_complexities = complexities

    ax1.plot(n_nodes, exec_times, 'bo-', linewidth=2, markersize=8, label='实际执行时间')
    ax1.plot(n_nodes, normalized_complexities, 'r--', linewidth=2, label='理论复杂度(归一化)')

    ax1.set_xlabel('网络节点数', fontsize=12)
    ax1.set_ylabel('执行时间 (秒)', fontsize=12)
    ax1.set_title('算法可扩展性分析', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    plt.tight_layout()

    if save_figures:
        plt.savefig("可扩展性分析.png", dpi=300, bbox_inches='tight')
        plt.close()
    else:
        try:
            plt.show()
        except:
            plt.savefig("可扩展性分析.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("可扩展性分析图已保存为 可扩展性分析.png")

    return results


def run_simple_experiment():
    """简化版实验，保存图像到文件"""
    print("杀伤链网络韧性评估与重构算法实验")
    print("=" * 60)

    # 运行三个拓扑的实验
    topologies = ['star', 'ring', 'random']
    all_results = []

    for topology in topologies:
        n_nodes = 10 if topology != 'random' else 20
        result = run_experiment(topology=topology, n_nodes=n_nodes, save_figures=True)
        all_results.append(result)

    # 可扩展性实验
    scalability_experiment(save_figures=True)

    # 输出结果摘要
    print("\n" + "=" * 60)
    print("实验摘要")
    print("=" * 60)

    print(f"\n{'拓扑':<10} {'原始效能':<10} {'受损效能':<10} {'修复后效能':<10} {'恢复比例%':<10}")
    print("-" * 70)

    for result in all_results:
        topology = result['topology']
        original = result['original_efficiency']
        damaged = result['damaged_efficiency']
        repaired = result['repaired_efficiency']
        recovery = result['recovery_ratio']

        print(f"{topology:<10} "
              f"{original:<10.4f} "
              f"{damaged:<10.4f} "
              f"{repaired:<10.4f} "
              f"{recovery:<10.2f}")

    print("\n所有实验已完成！图像已保存为PNG文件。")

    return all_results


def run_text_only_experiment():
    """纯文本版本，完全不使用图形显示"""
    print("杀伤链网络韧性评估与重构算法实验（纯文本版）")
    print("=" * 60)

    topologies = ['star', 'ring', 'random']
    all_results = []

    for topology in topologies:
        print(f"\n{'=' * 60}")
        print(f"实验配置: {topology}网络")
        print('=' * 60)

        n_nodes = 10 if topology != 'random' else 20

        # 创建网络
        network = KillChainNetwork(n_nodes=n_nodes, topology=topology)
        network_info = network.get_network_info()

        print("\n1. 网络信息:")
        for key, value in network_info.items():
            print(f"   {key}: {value}")

        # 计算原始效能
        original_states = network.node_states.copy()
        for node in network.G.nodes():
            network.node_states[node] = 1
        original_efficiency = network.calculate_connectivity_efficiency()
        network.node_states = original_states

        print(f"\n2. 原始网络连通效能: {original_efficiency:.4f}")

        # 模拟攻击
        damaged_nodes = network.simulate_attack(0.3)
        damaged_efficiency = network.calculate_connectivity_efficiency()

        print(f"\n3. 攻击后状态:")
        print(f"   受损节点: {sorted(damaged_nodes)}")
        print(f"   受损后连通效能: {damaged_efficiency:.4f}")

        # 关键节点分析
        analyzer = CriticalNodeAnalyzer(network)
        criticality_scores = analyzer.calculate_criticality()

        top_critical = analyzer.get_top_critical_nodes(top_k=3)
        print("\n4. 关键性最高的3个节点:")
        for node, scores in top_critical:
            print(f"   节点{node}: 关键性={scores['criticality']:.4f}, 权重={scores['weight']}")

        # 修复决策
        repair_budget = int(len(damaged_nodes) * 0.5)
        greedy_repair_nodes = greedy_repair_schedule(damaged_nodes, criticality_scores, repair_budget)

        print(f"\n5. 修复决策 (预算: {repair_budget}个节点):")
        print(f"   贪心策略修复节点: {sorted(greedy_repair_nodes)}")

        # 评估效果
        test_network = copy.deepcopy(network)
        test_network.repair_nodes(greedy_repair_nodes)
        repaired_efficiency = test_network.calculate_connectivity_efficiency()

        recovery_ratio = 0
        if original_efficiency > damaged_efficiency:
            recovery_ratio = (repaired_efficiency - damaged_efficiency) / (
                        original_efficiency - damaged_efficiency) * 100

        print(f"\n6. 修复效果:")
        print(f"   修复后效能: {repaired_efficiency:.4f}")
        print(f"   效能恢复比例: {recovery_ratio:.2f}%")

        all_results.append({
            'topology': topology,
            'original': original_efficiency,
            'damaged': damaged_efficiency,
            'repaired': repaired_efficiency,
            'recovery': recovery_ratio
        })

    # 输出总结
    print("\n" + "=" * 60)
    print("实验总结")
    print("=" * 60)

    print(f"\n{'拓扑':<10} {'原始效能':<10} {'受损效能':<10} {'修复后效能':<10} {'恢复比例%':<10}")
    print("-" * 70)

    for result in all_results:
        print(f"{result['topology']:<10} "
              f"{result['original']:<10.4f} "
              f"{result['damaged']:<10.4f} "
              f"{result['repaired']:<10.4f} "
              f"{result['recovery']:<10.2f}")

    return all_results


# 主程序
if __name__ == "__main__":
    print("请选择运行模式：")
    print("1. 简化模式（保存图像到文件）")
    print("2. 纯文本模式（不显示或保存图像）")

    choice = input("请输入选择 (1/2): ").strip()

    if choice == "1":
        run_simple_experiment()
    elif choice == "2":
        run_text_only_experiment()
    else:
        print("无效选择，使用默认模式（纯文本）")

        run_text_only_experiment()
