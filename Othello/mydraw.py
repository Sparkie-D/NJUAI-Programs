import pygame
def draw_chessboard(chessboard,screen):
    '''画棋盘'''
    for x in range(0,chessboard.cell_size*(chessboard.cell_num+1),chessboard.cell_size): 
        pygame.draw.line(screen,(200,200,200),(x+chessboard.space,0+chessboard.space),(x+chessboard.space,chessboard.cell_size*chessboard.cell_num+chessboard.space),1)
    for y in range(0,chessboard.cell_size*(chessboard.cell_num+1),chessboard.cell_size):
        pygame.draw.line(screen,(200,200,200),(0+chessboard.space,y+chessboard.space),(chessboard.cell_size*chessboard.cell_num+chessboard.space,y+chessboard.space),1)

def draw_chess(chessboard,screen,chess_arr):
    '''画棋子'''
    for tup in chess_arr: 
        x,y=tup[0],tup[1]
        number=tup[2]
        chess_color=(30,30,30) if number==0 else (200,200,200) #30黑200白 
        pygame.draw.circle(screen,chess_color,[chessboard.space+(2*x-1)*chessboard.cell_size/2,chessboard.space+(2*y-1)*chessboard.cell_size/2],35,40)

def draw_coordinate(chessboard,screen):
    def draw_row(i):
        fonts=pygame.font.Font("Othello\msyh.ttc",40)
        text_coordinate=fonts.render("%d"%i,True,(0,250,0))
        screen.blit(text_coordinate,(10,chessboard.space+(2*i-1)*chessboard.cell_size/2-30))
    def draw_col(i):
        fonts=pygame.font.Font("Othello\msyh.ttc",40)
        text_coordinate=fonts.render("%d"%i,True,(0,250,0))
        screen.blit(text_coordinate,(chessboard.space+(2*i-1)*chessboard.cell_size/2,5))
    for i in range(1,chessboard.cell_num+1):
        draw_row(i)
        draw_col(i)

def draw_this_chess(chessboard,screen,pos):
    x,y=pos
    chess_color=(230,0,0) 
    pygame.draw.circle(screen,chess_color,[chessboard.space+(2*x-1)*chessboard.cell_size/2,chessboard.space+(2*y-1)*chessboard.cell_size/2],5,40)

def draw_choice(chessboard,screen,arr):
    for tup in arr:
        x,y=tup[0],tup[1]
        pygame.draw.rect(screen,(0,0,100),((chessboard.space+(x-1)*chessboard.cell_size,chessboard.space+(y-1)*chessboard.cell_size),(chessboard.cell_size-5,chessboard.cell_size-5)))

def draw_ending(chessboard,screen,scoretuple):
    '''传入获胜方和得分元组,得到结束页面'''
    black_score,white_score=scoretuple
    win_color=0 if black_score>white_score else 1
    winner="白棋" if win_color==1 else "黑棋"
    #screen.fill((250,250,250))
    fonts2=pygame.font.Font("Othello\msyh.ttc",80)  #创建字体对象
    text_ending=fonts2.render("%s获胜"%winner,True,(255,255,30)) #创建文本
    fonts3=pygame.font.Font("Othello\msyh.ttc",40)
    text_score=fonts3.render("黑棋得分：%d   白棋得分：%d"%(black_score,white_score),True,(150,0,0))
    screen.blit(text_ending,(chessboard.grid_size/2-160,chessboard.grid_size/2-200))    #渲染到屏幕上
    screen.blit(text_score,(chessboard.grid_size/2-250,chessboard.grid_size/2)) 

def draw_skip(chessboard,screen,skipped_color):
    '''传入被跳过方，得到跳过界面'''
    skipped_player="黑棋" if skipped_color==0 else "白棋"
    fonts4=pygame.font.Font("Othello\msyh.ttc",40)
    text_skip=fonts4.render("%s无落子位置，本轮弃权"%skipped_player,True,(255,255,130))
    screen.blit(text_skip,(chessboard.grid_size/2-200,chessboard.grid_size/2-100))

def draw_round(chessboard,screen,flag,game_round,scoretuple):
    black_score,white_score=scoretuple
    fonts0=pygame.font.Font("Othello\msyh.ttc",40)
    text_rounds=fonts0.render("回合:%d"%game_round,True,(255,255,255))
    screen.blit(text_rounds,(chessboard.grid_size-20,20))
    fonts1=pygame.font.Font("Othello\msyh.ttc",40)  
    player="黑子" if flag==0 else "白子"
    text_player=fonts1.render("%s回合"%player,True,(255,255,255))
    screen.blit(text_player,(chessboard.grid_size-20,100))
    fonts2=pygame.font.Font("Othello\msyh.ttc",40)
    text_scores=fonts2.render(": %s"%black_score,True,(255,255,255))
    screen.blit(text_scores,(chessboard.grid_size+70,200))
    fonts3=pygame.font.Font("Othello\msyh.ttc",40)
    text_scores=fonts3.render(": %s"%white_score,True,(255,255,255))
    screen.blit(text_scores,(chessboard.grid_size+70,280))
    pygame.draw.circle(screen,(30,30,30),(chessboard.grid_size+40,230),20,40)
    pygame.draw.circle(screen,(200,200,200),(chessboard.grid_size+40,310),20,40)

def draw_pro(chessboard,screen,flag,black_p):
    font=pygame.font.Font("Othello\msyh.ttc",40)
    text=font.render("黑子胜率为%d"%black_p,True,(0,0,0))
    screen.blit(text,(chessboard.grid_size-100,380))