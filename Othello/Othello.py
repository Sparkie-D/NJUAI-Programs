import pygame
import mychess
import mydraw
import myAI
import Monte_Carlo_Tree_Search as MCTS

pygame.init()
pygame.font.init()
'''
待解决的问题：
5.13
1、棋子序数出问题(done)
5.14
1、判断反转条件有误，只有第一步成功(done)
5.15
1、经debug，翻转有误，判断似乎是正确的
    经查，在翻转函数中对所有的赋值语句使用了“==”，yue
    判断功能（done）
2、翻转条件仍有误，找不出来(done)
3、AI功能仍未实现（报错）(done)
4、判断结束有误，全部棋盘占满时不能停止,无路可走时也不能停止
5.17
1、提示功能（done）
2、判断功能完善（done）
3、未解决如果点错位置闪退的情况，不知道如何重复点击(done)
5.18
1、添加文字（done）
2、添加背景图片（done）
3、重写win_judging函数(done)
初始实现完成，准备写蒙特卡洛树
5.22
1.MCTS的simulation函数总是无法停止，内层循环有问题，怀疑是make_tree函数或is_edge函数有误
'''

def Othello():
    '''棋盘初始信息'''
    class Chessboard:
        def __init__(self) -> None:
            self.space=50                                                   #棋盘和边界间距
            self.cell_size=80                                               #棋盘格大小
            self.cell_num=8                                                 #每行棋盘格个数
            self.grid_size=self.cell_size*(self.cell_num)+self.space*2-20   #游戏视窗大小 

    chessboard=Chessboard()
    cell_num=chessboard.cell_num
    cell_size=chessboard.cell_size
    space=chessboard.space
    black_p=0

    '''绘制屏幕'''
    grid_size=chessboard.grid_size                                      #总屏幕大小
    screen=pygame.display.set_mode((grid_size+150,grid_size))           #设置屏幕
    pygame.display.set_caption("Python黑白棋")                           #窗体标题设置
    background=pygame.image.load("C:\Users\ACER\Desktop\黑白棋\Othello\背景.jpg")                     #窗口背景设置

    '''游戏初始化'''
    chess_arr=[]    #棋子信息（x，y，序数）
    all_chessboard={(i,j):2 for i in range(1,cell_num+1) for j in range(1,1+cell_num)}#0黑1白2空
    all_chessboard[(4,4)],all_chessboard[(4,5)],all_chessboard[(5,4)],all_chessboard[(5,5)]=1,0,0,1 #初始化棋盘

    flag=0          #棋子颜色，0黑1白
    game_state=2    #游戏状态，0黑胜，1白胜，2继续游戏
    game_over=False #结束游戏条件
    game_round=0    #游戏回合
    AI_position=(0,0)
    AI_flag=[]
    hum_flag=[]

    '''设置AI和人类的棋子颜色'''
    if AI_black:
        AI_flag.append(0)
    if AI_white:
        AI_flag.append(1)
    if HUM_black:
        hum_flag.append(0)
    if HUM_white:
        hum_flag.append(1)

    '''下棋循环'''
    while True:
        '''循环初始化'''
        chess_arr=[(x,y,all_chessboard[(x,y)]) for x,y in all_chessboard if all_chessboard[(x,y)]!=2]
        scores=mychess.score_counting(all_chessboard)

        if game_over:
            '''停留在结束页面等待关闭'''
            while True:
                for event in pygame.event.get():
                    if event.type==pygame.QUIT:
                        pygame.quit()
                        exit()

        '''棋盘显示部分'''
        screen.fill((55,55,100))                                    #设置背景颜色（红，绿，蓝）
        screen.blit(background,(0,0))                               #画背景
        mydraw.draw_round(chessboard,screen,flag,game_round,scores) #写当前回合
        mydraw.draw_chessboard(chessboard,screen)                   #画棋盘
        mydraw.draw_chess(chessboard,screen,chess_arr)              #画棋子
        mydraw.draw_coordinate(chessboard,screen)                   #显示坐标
        mydraw.draw_this_chess(chessboard,screen,AI_position)       #显示当前落子
        # mydraw.draw_pro(chessboard,screen,flag,black_p)
        pygame.display.update()

        '''开始下棋'''
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                exit()
            
            '''HUM_play'''
            if flag in hum_flag:
                '''以下为设置提示'''
                x,y=pygame.mouse.get_pos() #获取鼠标位置
                xi=int((round(x-chessboard.space+chessboard.cell_size/2)/(chessboard.cell_size/2)+1)/2)
                yi=int((round(y-chessboard.space+chessboard.cell_size/2)/(chessboard.cell_size/2)+1)/2)
                '''提示中获得了当前鼠标位置'''
                choice_arr=myAI.possible_choice(chessboard,all_chessboard,flag,switch=1)
                if (xi,yi) in choice_arr:
                    mydraw.draw_choice(chessboard,screen,[(xi,yi)])  #提示人类玩家落子位置
                if event.type==pygame.MOUSEBUTTONUP : 
                    '''翻转判断以及执行'''
                    if (xi,yi) in choice_arr:
                        res=mychess.able_to_play(chessboard,(xi,yi),flag,all_chessboard)
                        False_count=res.count(False)
                        if False_count!=8:
                            if(flag%2==0):
                                game_round+=1
                            mychess.turn_color((xi,yi),flag,all_chessboard,res)
                            all_chessboard[(xi,yi)]=flag
                            chess_arr=[(x,y,all_chessboard[(x,y)]) for x,y in all_chessboard if all_chessboard[(x,y)]!=2] 
                            flag=(flag+1)%2
                            AI_position=(xi,yi) #为AI提供父节点
                            '''判断是否终止(not finished)'''
                            game_state=mychess.win_judging(chessboard,all_chessboard,flag)
                            if game_state==0 or game_state==1:
                                scores=mychess.score_counting(all_chessboard)
                                mydraw.draw_ending(chessboard,screen,scores)
                                pygame.display.update()
                                game_over=True
                                break
                            if game_state==3:
                                mydraw.draw_chess(chessboard,screen,chess_arr)
                                mydraw.draw_skip(chessboard,screen,game_state)
                                pygame.display.update()
                                pygame.time.delay(3000)
                                flag=(flag+1)%2
                            continue

            '''AI_play'''
            if flag in AI_flag: 
                if(flag%2==0):
                    game_round+=1
                '''贪心算法的AI'''
                #AI_position=myAI.AI_play(chessboard,all_chessboard,chess_color=0)
                '''蒙特卡洛树搜索的AI'''
                mcts=MCTS.MCTS(chessboard,all_chessboard,flag)
                if AI_position==(0,0):
                    AI_position=(4,5)
                AI_chess=MCTS.Node(AI_position)
                AI_chess.set_color(flag)
                AI_chess.set_state(all_chessboard)
                AI_node=mcts.search(AI_chess)
                AI_position=AI_node.pos

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
                    pygame.time.delay(1000)
                    game_over=True
                    break
                elif game_state==3:
                    mydraw.draw_chess(chessboard,screen,chess_arr)
                    mydraw.draw_skip(chessboard,screen,flag)
                    pygame.display.update()
                    pygame.time.delay(1000)
                    flag=(flag+1)%2
                continue  
            pygame.display.update()
        
        pygame.display.update()
    
if __name__=="__main__":
    AI_black,AI_white=False,False
    HUM_black,HUM_white=True,True
    Othello()