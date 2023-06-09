from math import fabs
import mychess
import mydraw
import pygame
from functools import reduce

def possible_choice(chessboard,all_chessboard,chess_color,switch):
    chess_arr=[(x,y) for x,y in all_chessboard if all_chessboard[(x,y)]!=2]
    choice_arr=[]
    AI_dict={}
    score_list=[]
    for position in all_chessboard.keys():
        if position in chess_arr:
            continue
        else :
            res=mychess.able_to_play(chessboard,position,chess_color,all_chessboard)
            if res.count(False)!=8:
                choice_arr.append((position[0],position[1]))                  
                res_processed=[i for i in res if i!=False]  #找出各个方向能反转的棋子个数
                score=reduce(lambda x,y:x+y,res_processed)   #求和即为这个位置可以得的分数
                AI_dict[score]=position #相同得分的位置存一个即可
                score_list=list(AI_dict.keys())
                score_list.sort()
    if switch==1:    #state为1时找到所有可以落子的位置
        return choice_arr
    elif switch==2:  #state为2时找出AI最适合落子的位置
        return AI_dict[score_list[-1]] 
        
def AI_play(chessboard,all_chessboard:dict,chess_color:int) -> tuple:
    x,y=possible_choice(chessboard,all_chessboard,chess_color,2)
    return (x,y) #返回最大得分的位置

def AI_go(chessboard,all_chessboard,screen,flag):
    game_over=False
    AI_position=AI_play(chessboard,all_chessboard,chess_color=1)
    res=mychess.able_to_play(chessboard,AI_position,flag,all_chessboard)
    mychess.turn_color(AI_position,flag,all_chessboard,res)
    all_chessboard[AI_position]=flag
    chess_arr=[(x,y,all_chessboard[(x,y)]) for x,y in all_chessboard if all_chessboard[(x,y)]!=2]
    flag=(flag+1)%2
    '''判断是否终止(not finished)'''
    game_state=mychess.win_judging(chessboard,all_chessboard,flag)
    if game_state==0 or game_state==1:
        scores=mychess.score_counting(all_chessboard)
        mydraw.draw_chess(chessboard,screen,chess_arr)
        mydraw.draw_ending(chessboard,screen,scores)
        pygame.display.update()
        game_over=True
    elif game_state==3:
        mydraw.draw_chess(chessboard,screen,chess_arr)
        mydraw.draw_skip(chessboard,screen,flag)
        pygame.display.update()
        pygame.time.delay(1000)
        flag=(flag+1)%2
