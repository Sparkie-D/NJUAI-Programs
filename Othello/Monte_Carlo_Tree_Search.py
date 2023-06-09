import random
import mychess
import myAI
import time
import math
import mydraw

'''
6.4发现的问题：
没有进行或保存随机模拟的数据，落子位置其实是随机的
'''
class Node:
    def __init__(self,position,color=None) -> None:
        self.pos=position
        self.board_state={}
        self.parent=None
        self.children=[]
        self.visit_time=1e-10
        self.win_time=0
        self.color=color
    def get_parent(self):
        return self.parent
    def get_children(self):
        return self.children
    def set_parent(self,parent):
        self.parent=parent
    def add_child(self,child):
        self.children.append(child)
    def set_visit_time(self,time):
        self.visit_time=time
    def set_state(self,board_state):
        self.board_state=board_state
    def set_color(self,color):
        self.color=color
    def get_random_child(self):
        return random.choice(self.get_children())
    def get_best_child(self):
        child_dic={node.score_counting():node for node in self.children}
        score_list=list(child_dic.keys())
        score_list.sort()
        return child_dic[score_list[-1]]
    def score_counting(self):
        bonus=[
            [  1,  -1, 0.8, 0.8, 0.8, 0.8,  -1,   1],
            [ -1,  -1, 0.4, 0.4, 0.4, 0.4,  -1,  -1],
            [0.8, 0.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
            [0.8, 0.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
            [0.8, 0.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
            [0.8, 0.4, 0.8, 0.8, 0.8, 0.8, 0.4, 0.8],
            [ -1,  -1, 0.4, 0.4, 0.4, 0.4,  -1,  -1],
            [  1,  -1, 0.8, 0.8, 0.8, 0.8,  -1,   1]
        ]
        return (self.win_time/self.visit_time)*bonus[self.pos[1]-1][self.pos[0]-1]

class MCTS:
    '''
    计划写一个以当前棋盘状态为特征来计算的MCTS，node的状态里面包含此时这颗棋落入之后的棋盘状态，
    对棋的所有估值计算都可以通过访问当前节点的棋盘状态来得到
    '''
    def __init__(self,board,board_dict:dict,AI_color) -> None:#board:Chessboard
        self.board=board
        self.color_dict=board_dict #原来的all_chessboard
        self.init_arr=[Node(pos,board_dict[pos]) for pos in board_dict.keys() if board_dict[pos]==AI_color]
        for node in self.init_arr:#初始化这些棋子的棋盘状态已知
            node.board_state=self.color_dict
        self.AI_color=AI_color
        self.all_visit_time=1
        self.round=0
        self.black=0
    def search(self,this_node):
        '''以对手下的棋为当前状态，其best_child为所要的AI落点'''
        self.simulation(this_node)
        return this_node.get_best_child()
    def set_board_state(self,node):
        '''设置当前节点的棋盘状态，当前节点不能是根'''
        res=mychess.able_to_play(self.board,node.pos,node.color,node.get_parent().board_state)
        node.board_state=node.get_parent().board_state.copy()
        mychess.turn_color(node.pos,node.color,node.board_state,res)
        node.board_state[node.pos]=node.color
    def make_tree(self,this_node,game_state):
        '''可以看做是找出当前节点的所有孩子，并给他们设置好棋盘状态'''
        choice_arr=myAI.possible_choice(self.board,this_node.board_state,this_node.color,switch=1)
        for choice in choice_arr:
            if game_state==3:
                color=this_node.color
            else:
                color=(this_node.color+1)%2
            opponent_node=Node(choice,color)
            opponent_node.set_color(color)
            opponent_node.set_parent(this_node)
            this_node.add_child(opponent_node)
            self.set_board_state(opponent_node)  
    def is_edge(self,node):
        game_state=mychess.win_judging(self.board,node.board_state,node.color)
        if game_state==0 or game_state==1:
            return True,game_state
        elif game_state==2:
            return False,2
        else :
            return False,3
    def get_winner(self,edge_node):
        '''返回值：0黑胜，1白胜'''
        return mychess.win_judging(self.board,edge_node.board_state,edge_node.color)
    def backpropagate(self,node,is_win=False):
        self.all_visit_time+=1
        while node.parent!=None:
            node.visit_time+=1
            if is_win:
                node.win_time+=1
            node=node.get_parent()
    def simulation(self,node):
        '''模拟双方落子，直到规定时间退出'''
        start_node=node
        time_end=time.time()+2                                             #秒后停止模拟
        while True:
            node=start_node
            breakout=False
            game_state=2
            while True:
                if len(node.get_children())==0: #当前节点如果没有孩子，建立孩子
                    self.make_tree(node,game_state)
                    if len(node.get_children())==0:
                        break
                '''选择孩子的策略'''
                num_arr=[i for i in node.board_state if node.board_state[i]==0 or node.board_state[i]==1]
                #node=node.get_random_child()
                node=self.rollout(node) #这个是更新后的版本，返回UCT值最大的孩子
                breakout,game_state=self.is_edge(node)
                if breakout:
                    self.round+=1
                    if self.get_winner(node)==0:
                        self.black+=1
                    break                
            self.backpropagate(node,self.get_winner(node)==self.AI_color)
            if time.time()>=time_end:
                if self.round!=0:
                    return self.black/self.round
                else :
                    return 0
    def UCT(self,node):
        '''选择孩子时使用的评估函数'''
        UCT_value=node.win_time/node.visit_time+2*math.sqrt(math.log10(self.all_visit_time)/node.visit_time)
        return UCT_value
    def rollout(self,node):
        '''选择最佳孩子'''
        child_arr=node.get_children()
        value_arr=list(map(self.UCT,child_arr))
        target=value_arr.index(max(value_arr))
        return child_arr[target]
   