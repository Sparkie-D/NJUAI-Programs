import myAI
from functools import reduce

def able_to_play(chessboard,pos,color,all_chessboard:dict) ->list:
    '''
    判断落子函数：
    若可以落子，返回可以改变颜色的方向以及距离（list），
    若不可以落子，返回False
    '''
    flag=color
    x,y=pos
    res=[False for i in range(8)] #八个方向
    '''横'''
    '''left'''
    for i in range(1,chessboard.cell_num):
        if x-i<1:
            break
        if all_chessboard[(x-i,y)]==2 or all_chessboard[(x-1,y)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x-i,y)]==(flag+1)%2:
                continue
            elif all_chessboard[(x-i,y)]==flag and i!=1:
                res[0]=i
                break
    '''right'''
    for i in range(1,chessboard.cell_num):
        if x+i>chessboard.cell_num:
            break
        if all_chessboard[(x+i,y)]==2 or all_chessboard[(x+1,y)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x+i,y)]==(flag+1)%2:
                continue
            elif all_chessboard[(x+i,y)]==flag and i!=1:
                res[1]=i
                break
    '''竖'''
    '''up'''
    for i in range(1,chessboard.cell_num):
        if y-i<1:
            break
        if all_chessboard[(x,y-i)]==2 or all_chessboard[(x,y-1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x,y-i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x,y-i)]==flag and i!=1:
                res[2]=i
                break
    '''down'''
    for i in range(1,chessboard.cell_num):
        if y+i>chessboard.cell_num:
            break
        if all_chessboard[(x,y+i)]==2 or all_chessboard[(x,y+1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x,y+i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x,y+i)]==flag and i!=1:
                res[3]=i
                break
    '''斜'''
    '''left up'''
    for i in range(1,chessboard.cell_num):
        if x-i<1 or y-i<1:
            break
        if all_chessboard[(x-i,y-i)]==2 or all_chessboard[(x-1,y-1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x-i,y-i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x-i,y-i)]==flag and i!=1:
                res[4]=i
                break
    '''left down'''
    for i in range(1,chessboard.cell_num):
        if x-i<1 or y+i>chessboard.cell_num:
            break
        if all_chessboard[(x-i,y+i)]==2 or all_chessboard[(x-1,y+1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x-i,y+i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x-i,y+i)]==flag and i!=1:
                res[5]=i
                break
    '''right up'''        
    for i in range(1,chessboard.cell_num):
        if x+i>chessboard.cell_num or y-i<1:
            break
        if all_chessboard[(x+i,y-i)]==2 or all_chessboard[(x+1,y-1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x+i,y-i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x+i,y-i)]==flag and i!=1:
                res[6]=i
                break
    '''right down'''
    for i in range(1,chessboard.cell_num):
        if x+i>chessboard.cell_num or y+i>chessboard.cell_num:
            break
        if all_chessboard[(x+i,y+i)]==2 or all_chessboard[(x+1,y+1)]!=(flag+1)%2:
            break
        else :
            if all_chessboard[(x+i,y+i)]==(flag+1)%2:
                continue
            elif all_chessboard[(x+i,y+i)]==flag and i!=1:
                res[7]=i
                break
    return res

def turn_color(pos,color,all_chessboard:dict,res:list):
    '''
    颜色反转函数
    res=[左，右，上，下，左上，左下，右上，右下]
    '''
    flag=color
    x,y=pos
    if res[0]!=False:
        '''left'''
        for i in range(1,res[0]):
            if all_chessboard[(x-i,y)]==flag:
                break
            all_chessboard[(x-i,y)]=flag
    if res[1]!=False:
        '''right'''
        for i in range(1,res[1]):
            if all_chessboard[(x+i,y)]==flag:
                break
            all_chessboard[(x+i,y)]=flag
    if res[2]!=False:
        '''up'''
        for i in range(1,res[2]):
            if all_chessboard[(x,y-i)]==flag:
                break
            all_chessboard[(x,y-i)]=flag
    if res[3]!=False:
        '''down'''
        for i in range(1,res[3]):
            if all_chessboard[(x,y+i)]==flag:
                break
            all_chessboard[(x,y+i)]=flag
    if res[4]!=False:
        '''left up'''
        for i in range(1,res[4]):
            if all_chessboard[(x-i,y-i)]==flag:
                break    
            all_chessboard[(x-i,y-i)]=flag
    if res[5]!=False:
        '''left down'''
        for i in range(1,res[5]):
            if all_chessboard[(x-i,y+i)]==flag:
                break
            all_chessboard[(x-i,y+i)]=flag
    if res[6]!=False:
        '''right up'''
        for i in range(1,res[6]):
            if all_chessboard[(x+i,y-i)]==flag:
                break
            all_chessboard[(x+i,y-i)]=flag
    if res[7]!=False:
        '''right down'''
        for i in range(1,res[7]):
            if all_chessboard[(x+i,y+i)]==flag:
                break
            all_chessboard[(x+i,y+i)]=flag

    '''判断结束条件'''

def score_counting(all_chessboard):
    '''计算得分函数'''
    black_arr=[position for position in all_chessboard.keys() if all_chessboard[position]==0 ]
    white_arr=[position for position in all_chessboard.keys() if all_chessboard[position]==1 ]
    return (len(black_arr),len(white_arr))

def win_judging(chessboard,all_chessboard,current_color):
    '''
    胜利条件：
        1、一方没有棋子，则另一方获胜
        2、棋盘上没有空闲位置，或一方没有可以落子的位置：计算棋盘上黑白棋子总数，多者获胜
    返回值：0黑胜,1白胜,2继续,3跳过
    '''         
    winner=2    #初始状态，无人胜利
    black_arr=[position for position in all_chessboard.keys() if all_chessboard[position]==0 ]
    white_arr=[position for position in all_chessboard.keys() if all_chessboard[position]==1 ]
    black,white=len(black_arr),len(white_arr)
    '''若有一方无子，游戏结束'''
    if black==0 or white==0:
        winner=current_color
        return (winner+1)%2
    '''判断对手是否存在可以落子的位置'''
    available_arr_opponent=myAI.possible_choice(chessboard,all_chessboard,current_color,1)
    available_arr=myAI.possible_choice(chessboard,all_chessboard,(current_color+1)%2,1)
    '''
    若对方没有可落子的位置，视为弃权
    若双方都没有可落子的位置，游戏结束
    '''
    if len(available_arr_opponent)==0:
        if len(available_arr)==0:  #自己也没有落子位置
            if black > white :
                return 0
            else :
                return 1 
        else :
            winner=3
    return winner    
    