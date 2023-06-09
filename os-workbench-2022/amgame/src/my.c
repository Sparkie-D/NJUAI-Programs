#include<game.h>

#define SIDE 16

static int32_t FPS = 144;
static int32_t x = 0, y = 0;
static int32_t vx = 0, vy = 0;

static void draw_tile(int x, int y, int w, int h, uint32_t color) {
  uint32_t pixels[w * h]; // WARNING: large stack-allocated memory
  AM_GPU_FBDRAW_T event = {
    .x = x, .y = y, .w = w, .h = h, .sync = 1,
    .pixels = pixels,
  };
  for (int i = 0; i < w * h; i++) {
    pixels[i] = color;
  }
  ioe_write(AM_GPU_FBDRAW, &event);
}


int main(const char *args) {
  printf("welcome to my game !\n");
  
  ioe_init();

  int32_t next_frame = 0;

  AM_GPU_CONFIG_T info = {0};
  ioe_read(AM_GPU_CONFIG, &info);
  int32_t w = info.width;
  int32_t h = info.height;

  int32_t key = 0;

  while ((key = io_read(AM_INPUT_KEYBRD).keycode) != 1) {
    x = x + vx;
    y = y + vy;
    draw_tile(x , y , SIDE, SIDE, 0xffffff);
    while ((io_read(AM_TIMER_UPTIME).us / 1000) < next_frame) ;
    //printf("x = %d, y = %d, vx = %d,vy = %d\n", x, y, vx, vy);
    if(x < 0 || x  >= w - SIDE) vx *= -1;
    if(y < 0 || y  >= h - SIDE) vy *= -1;
    while ((key = io_read(AM_INPUT_KEYBRD).keycode) != AM_KEY_NONE) {
      //kbd_event(key);
      switch (key) {
        case 73:vy ++; break;  //up
        case 74:vy --; break;  //down
        case 75:vx --; break;  //left
        case 76:vx ++; break;  //right
        case 1 :halt(0);break;  //ESC
        default: break;
      }   
    }
    draw_tile(x , y , SIDE, SIDE, 0x000000);
    next_frame += 1000 / FPS;
    //update_screen();
  }
  return 0;
}
