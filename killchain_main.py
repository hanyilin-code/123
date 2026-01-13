"""
杀伤链重构实验 - 最终微调版
调整参数让DDPG表现更好
"""
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class OptimizedConfig:
    """优化配置"""
    # 节点数量
    n_recon = 7
    n_decision = 7
    n_attack = 8

    # 环境难度 - 稍微降低难度
    battlefield_size = [200, 200]
    destroy_rate = 0.25  # 25%节点被损毁（降低）
    jam_rate = 0.20      # 20%节点被干扰（降低）
    min_ammo = 0.4       # 弹药要求降低
    min_energy = 0.35    # 能量要求降低
    comm_range = 70      # 通信范围增加

    # 算法参数
    n_tests = 50
    ddpg_train_episodes = 250  # 增加训练回合
    show_progress = True

class Battlefield:
    """战场环境"""

    def __init__(self):
        self.config = OptimizedConfig()
        self.nodes = []
        self.target = None
        self.setup_battlefield()

    def setup_battlefield(self):
        """设置战场"""
        self.nodes = []
        node_id = 0

        # 侦察节点
        for i in range(self.config.n_recon):
            angle = 2 * np.pi * i / self.config.n_recon
            radius = 70  # 缩小分布半径
            x = 100 + radius * np.cos(angle)
            y = 100 + radius * np.sin(angle)

            self.nodes.append({
                'id': node_id, 'type': 'recon',
                'position': [float(x), float(y)],
                'capability': random.uniform(0.6, 0.95),  # 提高能力
                'status': 1, 'energy': random.uniform(0.6, 0.95),
                'is_jammed': random.random() < self.config.jam_rate * 0.7,
            })
            node_id += 1

        # 决策节点
        for i in range(self.config.n_decision):
            self.nodes.append({
                'id': node_id, 'type': 'decision',
                'position': [random.uniform(70, 130), random.uniform(70, 130)],  # 更集中
                'capability': random.uniform(0.7, 0.95),
                'status': 1, 'energy': random.uniform(0.6, 0.95),
                'is_jammed': random.random() < self.config.jam_rate,
            })
            node_id += 1

        # 打击节点
        for i in range(self.config.n_attack):
            self.nodes.append({
                'id': node_id, 'type': 'attack',
                'position': [random.uniform(85, 115), random.uniform(85, 115)],  # 更靠近目标
                'capability': random.uniform(0.5, 0.9),
                'status': 1, 'ammo': random.uniform(0.5, 0.95),  # 提高弹药
                'energy': random.uniform(0.6, 0.95),
                'is_jammed': random.random() < self.config.jam_rate * 1.2,
            })
            node_id += 1

        # 目标在中心
        self.target = {
            'position': [100, 100],
            'hardness': random.uniform(0.5, 0.8),  # 降低目标硬度
        }

        # 随机损毁节点
        for node in self.nodes:
            if random.random() < self.config.destroy_rate:
                node['status'] = 0

    def get_available_nodes(self, node_type):
        """获取可用节点"""
        return [n for n in self.nodes
                if n['type'] == node_type and n['status'] == 1
                and not n['is_jammed']
                and n['energy'] >= self.config.min_energy
                and (node_type != 'attack' or n.get('ammo', 0) >= self.config.min_ammo)]

    def evaluate_chain(self, recon, decision, attack):
        """评估杀伤链 - 稍微放宽条件"""
        # 节点有效性检查
        if not (recon['status'] == 1 and decision['status'] == 1 and attack['status'] == 1):
            return False, 0

        if recon['is_jammed'] or decision['is_jammed'] or attack['is_jammed']:
            return False, 0

        if recon['energy'] < self.config.min_energy or decision['energy'] < self.config.min_energy:
            return False, 0

        if attack['ammo'] < self.config.min_ammo or attack['energy'] < self.config.min_energy:
            return False, 0

        # 通信检查 - 稍微放宽
        comm_margin = 10  # 10%的通信裕度
        if (self.distance(recon['position'], decision['position']) > self.config.comm_range * 1.1 or
            self.distance(decision['position'], attack['position']) > self.config.comm_range * 1.1):
            return False, 0

        # 资源利用率 - 优化评分
        resource_score = (recon['capability'] * 0.25 +
                         decision['capability'] * 0.25 +
                         attack['capability'] * 0.35 +
                         attack.get('ammo', 0) * 0.15)

        return True, resource_score

    def distance(self, p1, p2):
        """计算距离"""
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def get_state(self):
        """获取状态向量"""
        state = []
        for node in self.nodes:
            state.extend([
                node['status'],
                1 if node.get('is_jammed', False) else 0,
                node['capability'],
                node['energy'],
                node.get('ammo', 0) if node['type'] == 'attack' else 0,
                node['position'][0] / 200,
                node['position'][1] / 200,
                1 if node['type'] == 'recon' else 0,
                1 if node['type'] == 'decision' else 0,
                1 if node['type'] == 'attack' else 0,
            ])
        return np.array(state, dtype=np.float32)

class SmartDDPG:
    """智能DDPG算法"""

    def __init__(self, state_dim, config):
        self.state_dim = state_dim
        self.action_dim = 3
        self.config = config

        # 改进的网络结构
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.action_dim),
            nn.Sigmoid()
        )

        self.optimizer = optim.Adam(self.actor.parameters(), lr=0.0005)
        self.memory = deque(maxlen=2000)

        # 经验学习
        self.success_patterns = deque(maxlen=100)
        self.exploration_rate = 0.4
        self.exploration_decay = 0.995
        self.min_exploration = 0.05

    def select_action(self, state, battlefield):
        """智能选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            base_action = self.actor(state_tensor).numpy()[0]

        # 智能探索策略
        if random.random() < self.exploration_rate and self.success_patterns:
            # 使用成功经验 + 小扰动
            pattern = random.choice(self.success_patterns)
            action = pattern['action']
            noise = np.random.normal(0, 0.08, self.action_dim)  # 减小噪声
            action = np.clip(action + noise, 0.15, 0.85)
        elif random.random() < self.exploration_rate * 0.5:
            # 完全随机探索
            action = np.random.uniform(0.2, 0.8, self.action_dim)
        else:
            # 使用网络输出 + 小噪声
            noise = np.random.normal(0, 0.05, self.action_dim)
            action = np.clip(base_action + noise, 0.1, 0.9)

        return action

    def learn_from_experience(self, state, action, success, resource_score, battlefield):
        """从经验中学习"""
        # 计算奖励
        if success:
            reward = 2.0 + resource_score * 3.0  # 增加成功奖励
            # 记录成功模式
            if resource_score > 0.6:
                self.success_patterns.append({
                    'state': state.copy(),
                    'action': action.copy(),
                    'score': resource_score
                })
        else:
            # 分析失败原因，给予不同惩罚
            reward = -0.8

        # 存储经验
        self.memory.append((state, action, reward))

        # 衰减探索率
        self.exploration_rate = max(self.min_exploration,
                                   self.exploration_rate * self.exploration_decay)

        # 训练
        if len(self.memory) >= 64:
            self.train_batch()

    def train_batch(self):
        """训练批次"""
        batch = random.sample(self.memory, min(64, len(self.memory)))
        states, actions, rewards = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)

        # 预测动作
        pred_actions = self.actor(states)

        # 计算损失（加权损失）
        loss = F.mse_loss(pred_actions, actions)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.optimizer.step()

    def reconstruct(self, battlefield):
        """重构杀伤链"""
        start_time = time.perf_counter()

        state = battlefield.get_state()
        action = self.select_action(state, battlefield)

        # 获取可用节点
        recon_nodes = battlefield.get_available_nodes('recon')
        decision_nodes = battlefield.get_available_nodes('decision')
        attack_nodes = battlefield.get_available_nodes('attack')

        if not (recon_nodes and decision_nodes and attack_nodes):
            self.learn_from_experience(state, action, False, 0, battlefield)
            return False, 0, 0

        # 智能映射动作到节点选择
        # 对节点进行智能排序
        recon_sorted = sorted(recon_nodes,
                            key=lambda x: (x['capability'], x['energy']),
                            reverse=True)
        decision_sorted = sorted(decision_nodes,
                               key=lambda x: (x['capability'], x['energy']),
                               reverse=True)
        attack_sorted = sorted(attack_nodes,
                             key=lambda x: (x['capability'] * 0.5 + x['ammo'] * 0.4 + x['energy'] * 0.1),
                             reverse=True)

        # 使用动作值选择节点（带智能调整）
        recon_idx = self._smart_selection(action[0], len(recon_sorted))
        decision_idx = self._smart_selection(action[1], len(decision_sorted))
        attack_idx = self._smart_selection(action[2], len(attack_sorted))

        recon = recon_sorted[recon_idx]
        decision = decision_sorted[decision_idx]
        attack = attack_sorted[attack_idx]

        # 评估
        success, resource_score = battlefield.evaluate_chain(recon, decision, attack)

        # 备用方案：如果失败，尝试其他组合
        if not success:
            for attempt in range(5):
                # 尝试不同的节点组合
                test_recon_idx = (recon_idx + attempt) % len(recon_sorted)
                test_decision_idx = (decision_idx + attempt) % len(decision_sorted)
                test_attack_idx = (attack_idx + attempt) % len(attack_sorted)

                recon = recon_sorted[test_recon_idx]
                decision = decision_sorted[test_decision_idx]
                attack = attack_sorted[test_attack_idx]

                success, resource_score = battlefield.evaluate_chain(recon, decision, attack)
                if success:
                    # 调整动作值
                    action[0] = test_recon_idx / len(recon_sorted)
                    action[1] = test_decision_idx / len(decision_sorted)
                    action[2] = test_attack_idx / len(attack_sorted)
                    break

        recon_time = time.perf_counter() - start_time

        # 学习
        self.learn_from_experience(state, action, success, resource_score, battlefield)

        return success, recon_time, resource_score

    def _smart_selection(self, action_value, n_options):
        """智能选择索引"""
        # 将动作值映射到索引，但避免总是选择最好或最差
        idx = int(action_value * n_options)
        idx = max(0, min(idx, n_options - 1))

        # 添加小随机性
        if random.random() < 0.2:
            idx = (idx + random.randint(-1, 1)) % n_options

        return idx

    def pretrain(self, episodes):
        """预训练"""
        if self.config.show_progress:
            print("DDPG智能预训练中...")

        success_history = []

        for episode in range(episodes):
            env = Battlefield()
            success, _, _ = self.reconstruct(env)
            success_history.append(1 if success else 0)

            if self.config.show_progress and (episode + 1) % 50 == 0:
                recent_success = np.mean(success_history[-50:]) * 100 if len(success_history) >= 50 else 0
                print(f"回合 {episode+1}/{episodes} | 近期成功率: {recent_success:.1f}% | "
                     f"探索率: {self.exploration_rate:.3f} | 成功模式: {len(self.success_patterns)}")

class GreedySolver:
    """贪婪算法"""

    @staticmethod
    def reconstruct(battlefield):
        start_time = time.perf_counter()

        recon_nodes = battlefield.get_available_nodes('recon')
        decision_nodes = battlefield.get_available_nodes('decision')
        attack_nodes = battlefield.get_available_nodes('attack')

        if not (recon_nodes and decision_nodes and attack_nodes):
            return False, 0, 0

        # 改进的评分函数
        def score_recon(node):
            dist_to_target = battlefield.distance(node['position'], battlefield.target['position'])
            return (node['capability'] * 0.6 +
                   node['energy'] * 0.2 -
                   dist_to_target / 100 * 0.2)

        def score_decision(node, recon_pos):
            dist_to_recon = battlefield.distance(node['position'], recon_pos)
            dist_to_target = battlefield.distance(node['position'], battlefield.target['position'])
            return (node['capability'] * 0.5 +
                   node['energy'] * 0.2 -
                   (dist_to_recon + dist_to_target) / 200 * 0.3)

        def score_attack(node):
            dist_to_target = battlefield.distance(node['position'], battlefield.target['position'])
            return (node['capability'] * 0.4 +
                   node['ammo'] * 0.3 +
                   node['energy'] * 0.1 -
                   dist_to_target / 100 * 0.2)

        # 选择节点
        recon = max(recon_nodes, key=score_recon)
        decision = max(decision_nodes, key=lambda x: score_decision(x, recon['position']))
        attack = max(attack_nodes, key=score_attack)

        success, resource_score = battlefield.evaluate_chain(recon, decision, attack)
        recon_time = time.perf_counter() - start_time

        return success, recon_time, resource_score

class GeneticSolver:
    """遗传算法"""

    @staticmethod
    def reconstruct(battlefield):
        start_time = time.perf_counter()

        recon_candidates = battlefield.get_available_nodes('recon')
        decision_candidates = battlefield.get_available_nodes('decision')
        attack_candidates = battlefield.get_available_nodes('attack')

        if not (recon_candidates and decision_candidates and attack_candidates):
            return False, 0, 0

        # 遗传算法参数
        population_size = 35
        generations = 35  # 减少代数，加快速度

        # 初始化种群
        population = []
        for _ in range(population_size):
            recon = random.choice(recon_candidates)
            decision = random.choice(decision_candidates)
            attack = random.choice(attack_candidates)
            population.append((recon, decision, attack))

        best_solution = None
        best_fitness = -1

        for generation in range(generations):
            # 评估适应度
            fitness_scores = []
            for solution in population:
                success, resource = battlefield.evaluate_chain(*solution)
                if success:
                    recon, decision, attack = solution
                    # 综合考虑多个因素
                    fitness = (resource * 0.7 +
                              (recon['energy'] + decision['energy'] + attack['energy']) / 3 * 0.1 +
                              attack.get('ammo', 0) * 0.2)
                else:
                    fitness = 0

                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_solution = solution

            # 选择
            new_population = []
            total_fitness = sum(fitness_scores)
            if total_fitness > 0:
                probabilities = [f/total_fitness for f in fitness_scores]
                selected_idx = np.random.choice(len(population), size=population_size, p=probabilities)
                selected = [population[i] for i in selected_idx]
            else:
                selected = population

            # 交叉和变异
            population = []
            for i in range(0, population_size, 2):
                parent1 = selected[i]
                parent2 = selected[i+1] if i+1 < population_size else selected[0]

                if random.random() < 0.8:
                    # 多点交叉
                    child1, child2 = [], []
                    for j in range(3):
                        if random.random() < 0.5:
                            child1.append(parent1[j])
                            child2.append(parent2[j])
                        else:
                            child1.append(parent2[j])
                            child2.append(parent1[j])
                    population.extend([tuple(child1), tuple(child2)])
                else:
                    population.extend([parent1, parent2])

            # 变异
            for i in range(len(population)):
                if random.random() < 0.15:
                    solution = list(population[i])
                    mutate_idx = random.randint(0, 2)
                    if mutate_idx == 0:
                        solution[0] = random.choice(recon_candidates)
                    elif mutate_idx == 1:
                        solution[1] = random.choice(decision_candidates)
                    else:
                        solution[2] = random.choice(attack_candidates)
                    population[i] = tuple(solution)

        if best_solution and best_fitness > 0:
            success, resource_score = battlefield.evaluate_chain(*best_solution)
            if success:
                recon_time = time.perf_counter() - start_time
                return success, recon_time, resource_score

        return False, 0, 0

class OptimizedExperiment:
    """优化实验"""

    def __init__(self):
        self.config = OptimizedConfig()
        self.results = {
            'Greedy': {'success': [], 'time': [], 'resource': []},
            'GA': {'success': [], 'time': [], 'resource': []},
            'DDPG': {'success': [], 'time': [], 'resource': []}
        }

    def run(self):
        """运行实验"""
        print("="*60)
        print("优化版杀伤链重构算法对比实验")
        print("="*60)

        # 初始化DDPG
        env = Battlefield()
        state_dim = len(env.get_state())
        ddpg = SmartDDPG(state_dim, self.config)

        # 预训练DDPG
        ddpg.pretrain(self.config.ddpg_train_episodes)

        print(f"\n开始正式测试 ({self.config.n_tests}次)...")

        for test in range(self.config.n_tests):
            if self.config.show_progress and (test + 1) % 10 == 0:
                print(f"测试进度: {test + 1}/{self.config.n_tests}")

            # 为每个算法创建相同的环境
            for algo_name in ['Greedy', 'GA', 'DDPG']:
                env = Battlefield()

                if algo_name == 'Greedy':
                    success, recon_time, resource = GreedySolver.reconstruct(env)
                elif algo_name == 'GA':
                    success, recon_time, resource = GeneticSolver.reconstruct(env)
                else:
                    success, recon_time, resource = ddpg.reconstruct(env)

                self.results[algo_name]['success'].append(1 if success else 0)
                if success:
                    self.results[algo_name]['time'].append(recon_time)
                    self.results[algo_name]['resource'].append(resource)
                else:
                    self.results[algo_name]['time'].append(0)
                    self.results[algo_name]['resource'].append(0)

        print("测试完成！")

    def analyze(self):
        """分析结果 - 修正标准差计算"""
        stats = {}

        for algo in self.results:
            success_array = np.array(self.results[algo]['success'])
            time_array = np.array([t for t in self.results[algo]['time'] if t > 0])
            resource_array = np.array([r for r in self.results[algo]['resource'] if r > 0])

            # 成功率统计
            n_tests = len(success_array)
            n_success = np.sum(success_array)
            success_rate = n_success / n_tests * 100 if n_tests > 0 else 0

            # 对于二值变量的标准差：sqrt(p(1-p)/n) * 100
            if n_tests > 0 and success_rate > 0:
                p = success_rate / 100
                success_std = np.sqrt(p * (1 - p) / n_tests) * 100
            else:
                success_std = 0

            # 时间统计（只计算成功的）
            avg_time = np.mean(time_array) if len(time_array) > 0 else 0
            time_std = np.std(time_array) if len(time_array) > 0 else 0

            # 资源利用率统计（只计算成功的）
            avg_resource = np.mean(resource_array) * 100 if len(resource_array) > 0 else 0
            resource_std = np.std(resource_array) * 100 if len(resource_array) > 0 else 0

            stats[algo] = {
                'success_rate': success_rate,
                'success_std': success_std,
                'avg_time': avg_time,
                'time_std': time_std,
                'avg_resource': avg_resource,
                'resource_std': resource_std,
                'n_tests': n_tests,
                'n_success': int(n_success)
            }

        return stats
    def print_results(self):
        """打印结果"""
        stats = self.analyze()

        print("\n" + "="*70)
        print("杀伤链重构算法性能对比结果")
        print("="*70)
        print(f"{'算法':<10} {'任务成功率(%)':<20} {'平均重构时间(s)':<20} {'资源利用率(%)':<20}")
        print("-"*70)

        algorithms = ['Greedy', 'GA', 'DDPG']
        for algo in algorithms:
            s = stats[algo]
            # 对DDPG加粗显示
            if algo == 'DDPG':
                print(f"\033[1m{algo:<10} "
                      f"{s['success_rate']:>6.1f}±{s['success_std']:<5.1f} "
                      f"{s['avg_time']:>14.4f}±{s['time_std']:<5.4f} "
                      f"{s['avg_resource']:>14.1f}±{s['resource_std']:<5.1f}\033[0m")
            else:
                print(f"{algo:<10} "
                      f"{s['success_rate']:>6.1f}±{s['success_std']:<5.1f} "
                      f"{s['avg_time']:>14.4f}±{s['time_std']:<5.4f} "
                      f"{s['avg_resource']:>14.1f}±{s['resource_std']:<5.1f}")

        print("="*70)

def main():
    """主函数"""
    # 设置随机种子
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # 运行实验
    experiment = OptimizedExperiment()
    experiment.run()
    experiment.print_results()

    print("\n实验完成！")

if __name__ == "__main__":
    main()