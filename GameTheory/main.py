import os
import time # 测试运行时间用得到
import numpy as np
from scipy.optimize import linprog


class NESolver:
    def __init__(self, player_actions, payoff_matrix):
        self.player_actions = player_actions
        self.payoff_matrix = payoff_matrix
        self.num_players = len(player_actions)
        self.PNE = []
        self.MNE = []
        # print(payoff_matrix, player_actions)

    def SolvePNE(self):
        '''
        求解纯策略纳什均衡
        '''
        PNE = []
        NEcandidates = []
        for id in range(self.num_players):
            nash_tuple = np.where(self.payoff_matrix[id] == np.max(self.payoff_matrix[id], axis=id, keepdims=True))
            maxs = np.max(self.payoff_matrix[id], axis=id, keepdims=True)
            # print(maxs, id)
            nash_arr = np.array([list(item) for item in nash_tuple]).T.tolist()
            # print(f"nash_arr {id}",nash_arr)
            NEcandidates.append(nash_arr)
        # print("candidates:",NEcandidates)
        for i in range(len(NEcandidates)):
            nash_arr = NEcandidates[i]  # 某个player的均衡点集
            for candidate in nash_arr:
                record_places = []  # 记录该候选项出现在哪些行中（这里利用到NE的一个前提假设：每个player对于每个位置只会计算一次NE）
                for j in range(len(NEcandidates)):
                    if candidate in NEcandidates[j]:
                        record_places.append(j)
                    else:
                        break  # 发现该点非均衡点后，直接跳出循环
                # print(candidate, ":", record_places)
                if len(record_places) == self.num_players and not candidate in PNE:
                    # 每一行中都有这个位置，证明该点是PNE
                    # print(candidate, ' added into PNE')
                    PNE.append(candidate)
                # 存在行中没有这个位置，则剪枝（删去所有这个候选项的出现）（理论上不剪枝也能成功运行）（改进：无论有没有都剪枝，去除重复）
                # （debug：删去后无法遍历所有元素，不要一边遍历一边删）
                # print("removing ", candidate, ' from ', record_places, end=' ')
                # for j in record_places:
                #     NEcandidates[j].remove(candidate)
                # print(" remain:", NEcandidates)
        # print("CurrentPNE:",PNE)
        return self.build_PNE(PNE)

    def build_PNE(self, PNEs):
        res = []
        for PNE in PNEs:
            legal_PNE = np.eye(self.player_actions[0])[PNE[0]].tolist()
            for i in range(1, len(PNE)):
                legal_PNE += np.eye(self.player_actions[i])[PNE[i]].tolist()
            legal_PNE = [int(i) for i in legal_PNE]
            res.append(legal_PNE)
        return res

    def eliminate_dominate(self):
        '''
        若有占优策略，则无法计算MNE
        仅在计算MNE前调用
        '''
        # print('before eliminate:', self.payoff_matrix, self.player_actions)
        elim_list_a = []
        for i in range(self.player_actions[0]):
            for j in range(i + 1, self.player_actions[0]):
                flag = self.payoff_matrix[0][i] > self.payoff_matrix[0][j]
                if flag.sum() == 0 and i not in elim_list_a:
                    elim_list_a.append(i)  # i弱
                elif flag.sum() == self.player_actions[1] and j not in elim_list_a:
                    elim_list_a.append(j)  # j弱

        elim_list_b = []
        for i in range(self.player_actions[1]):
            for j in range(i + 1, self.player_actions[1]):
                flag = self.payoff_matrix[1][:, i] > self.payoff_matrix[1][:, j]
                if flag.sum() == 0 and i not in elim_list_b:
                    elim_list_b.append(i)
                elif flag.sum() == self.player_actions[0] and j not in elim_list_b:
                    elim_list_b.append(j)

        # print("for player1:", elim_list_a)
        elim_list_a.sort(reverse=True)  # 从大的开始删除
        # print("for player2:", elim_list_b)
        elim_list_b.sort(reverse=True)
        for item in elim_list_a:
            self.payoff_matrix[0] = np.delete(self.payoff_matrix[0], item, axis=0)
            self.payoff_matrix[1] = np.delete(self.payoff_matrix[1], item, axis=0)  # 二者的收益矩阵都要减小
        self.player_actions[0] -= len(elim_list_a)
        for item in elim_list_b:
            self.payoff_matrix[0] = np.delete(self.payoff_matrix[0], item, axis=1)
            self.payoff_matrix[1] = np.delete(self.payoff_matrix[1], item, axis=1)
        self.player_actions[1] -= len(elim_list_b)
        # print("after eliminate:", self.payoff_matrix, self.player_actions)
        elim_list = []
        elim_list.append(elim_list_a)
        elim_list.append(elim_list_b)
        return elim_list

    def SolveMNE(self):
        '''
        求解混合策略纳什均衡
        '''
        assert self.num_players == 2
        elim_list = self.eliminate_dominate() # 不删除弱势策略，则可能导致更多MNE被忽略
        # elim_list = [[],[]]
        action_a = self.player_actions[0]
        action_b = self.player_actions[1]
        raw_MNE = []

        # 求解Player 1的MNE
        c = -self.payoff_matrix[1][:, 0]  # 定义第1列为优化目标
        # 等式约束：每一列都等于该列, 概率和为1
        A_eq, b_eq = [], []
        for j in range(action_b):
            A_eq.append(self.payoff_matrix[1][:, 0] - self.payoff_matrix[1][:, j])
            b_eq.append(0)
        A_eq.append(np.ones_like(self.payoff_matrix[1][:, 0]))
        b_eq.append(1)
        A_eq, b_eq = np.array(A_eq), np.array(b_eq)
        # 不等式约束：所有概率大于0
        A_ub = -np.eye(action_a)
        b_ub = np.zeros(action_a)
        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, method='highs')
        raw_MNE.append(sol.x if sol.x is not None else [])

        # 求解Player 2的MNE
        c = -self.payoff_matrix[0][0]
        A_eq, b_eq = [], []
        for j in range(action_a):
            A_eq.append(self.payoff_matrix[0][0] - self.payoff_matrix[0][j])
            b_eq.append(0)
        A_eq.append(np.ones_like(self.payoff_matrix[1][0]))
        b_eq.append(1)
        A_eq, b_eq = np.array(A_eq), np.array(b_eq)
        A_ub = -np.eye(action_b)
        b_ub = np.zeros(action_b)

        sol = linprog(c, A_ub, b_ub, A_eq, b_eq, method='highs')
        raw_MNE.append(sol.x if sol.x is not None else [])
        self.MNE = self.build_MNE(elim_list, raw_MNE) if len(raw_MNE[0]) > 0 and len(raw_MNE[1]) > 0 else []
        return self.MNE

    def build_MNE(self, elim_list, raw_MNE):
        MNE = [[], []]
        for i in range(2):
            MNE[i] = [-1 for _ in range(len(elim_list[i]) + len(raw_MNE[i]))]
            for j in elim_list[i]:
                MNE[i][j] = 0
            idx = 0
            for j in range(len(MNE[i])):
                if not MNE[i][j] == -1:
                    continue
                else:
                    MNE[i][j] = 0 if np.isclose(raw_MNE[i][idx], 0) else raw_MNE[i][idx]  # 去除-0.0
                    idx += 1
        res = MNE[0] + MNE[1]
        return res

    def step(self):
        self.PNE = self.SolvePNE()
        if self.num_players == 2:
            self.MNE = self.SolveMNE()
        # print(self.PNE, self.MNE)

    def compute_payoff(self, NE, payoff_line):
        return np.dot(NE, payoff_line)


def load_data(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
        # print('raw input:')
        # print(lines)
        payoff_str = lines[-1]  # 确定收益矩阵的位置
        player_info = lines[-3]  # 确定玩家个数和选择数的位置
        # 提取玩家选择数（选择数列表的长度即为玩家个数）
        idx = len(player_info) - 1
        while player_info[idx] != "{": idx -= 1
        player_actions = player_info[idx:-2].replace("{", "").replace("}", "").strip().split(" ")
        player_actions = [int(item) for item in player_actions]
        # 提取收益矩阵
        payoffs = payoff_str.strip().split()
        payoffs = [int(i) for i in payoffs]  # 字符串改为数值数组
        num_players = len(player_actions)
        payoff_matrix = []
        for player_id in range(num_players):
            id_payoff = []
            for idx in range(len(payoffs)):
                if idx % num_players == player_id:
                    id_payoff.append(payoffs[idx])
            id_payoff = np.array(id_payoff).reshape(player_actions, order='F')  # 按照题目顺序恢复原矩阵
            payoff_matrix.append(id_payoff)
        # print('player actions: ', player_actions)
        # print('payoff matrix:', payoff_matrix)
        return player_actions, payoff_matrix  # 按照player_id访问这两个数组，访问到的是同一个玩家的选择数和收益矩阵


def write_nash(solver, path):
    PNE = solver.PNE
    MNE = solver.MNE
    PNE.append(MNE)  # 此时MNE只有一个解，增加MNE解数量时需要修改此处
    result = []
    for ne in PNE:
        if len(ne) > 0:
            tmp = [int(x) if (type(x) == np.float64 or type(x) == np.float32) and x.is_integer() else x for x in ne]
            tmpstr = ','.join([str(i)[:5] for i in tmp]) + '\n'  # 最多保留五位有效字符
            if not tmpstr in result:
                result.append(tmpstr)
        else:
            PNE.remove(ne)
    print('solution :', result)
    with open(path, 'w') as f:
        for item in result:
            f.write(item)

def show_result(solver):
    PNE = solver.PNE
    MNE = solver.MNE
    PNE.append(MNE)  # 此时MNE只有一个解，增加MNE解数量时需要修改此处
    result = []
    for ne in PNE:
        if len(ne) > 0:
            tmp = [int(x) if (type(x) == np.float64 or type(x) == np.float32) and x.is_integer() else x for x in ne]
            tmpstr = ','.join([str(i)[:5] for i in tmp]) + '\n'  # 最多保留五位有效字符
            if not tmpstr in result:
                result.append(tmpstr)
        else:
            PNE.remove(ne)
    print('solution :', result)


def nash(in_path, out_path):
    # load file
    actions, payoffs = load_data(in_path)
    # get NE
    solver = NESolver(actions, payoffs)
    solver.step()
    # write file
    write_nash(solver, out_path)
    # show_result(solver) # 不写入目标文件，而是输出到终端
    pass


def load_result(path):
    # 从.ne文件中读取直接结果，可以和准备输出的内容作对比
    with open(path, 'r') as f:
        lines = f.readlines()
        print('reference:', lines)


if __name__ == '__main__':
    for f in os.listdir('input'):
        if f.endswith('.nfg'):
            nash('input/'+f, 'output/'+f.replace('nfg','ne'))

    # # 测试代码段
    # start = time.time()
    # for f in os.listdir('examples'):
    #     # if f.endswith('.ne') and f[-4] == '5':
    #     if f.endswith('.ne'):
    #         load_result('examples/' + f)
    #         nash('examples/' + f.replace('ne', 'nfg'), 'examples/' + f)
    #         # break
    # end = time.time()
    # print("Total time cost=", end-start)
