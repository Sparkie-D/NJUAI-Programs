#include <common.h>

callback_t callbacks[MAX_CALLBACK_SIZE];
int cur_callback_size = 0;

static void os_init() {
  //整个系统只调用一次
  pmm->init();
  kmt->init();
}
int os_lock = UNLOCK;

static void os_run() {
  //每个处理器都调用一次
  /*
  for (const char *s = "Hello World from CPU #*\n"; *s; s++) {
    putch(*s == '*' ? '0' + cpu_current() : *s);
  }
  */
  iset(true);
  while (1) ;
}
//中断异常处理程序的唯一入口
static Context *os_trap(Event ev, Context *context){
  //中断后AM保存期存器现场到堆栈，然后调用os_trap，在函数返回后将trap返回的寄存器现场恢复到CPU
  // TODO
  Context *next = NULL;
  for(int i = 0; i < cur_callback_size; i ++){
    //for 循环体中是按seq排好序的handlers
    if(callbacks[i].event == EVENT_NULL || callbacks[i].event == ev.event){
      Context *ret = callbacks[i].handler(ev, context);
      if(ret != NULL) next = ret;
    }
  }
  return next;
}

//注册一个在中断时调用的callback
static void os_on_irq(int seq, int event, handler_t handler){
  // TODO
  callbacks[cur_callback_size ++] = (callback_t){seq, event, handler};
  for(int i = 0; i < cur_callback_size - 1; i ++){
    if(callbacks[i].seq > callbacks[cur_callback_size - 1].seq){
      callback_t tmp = callbacks[cur_callback_size - 1];
      for(int j = cur_callback_size - 2; j >= i; j --){
        callbacks[j] = callbacks[j + 1];
      }
      callbacks[i] = tmp;
      break;
    }
  }
  return;
}


MODULE_DEF(os) = {
  .init   = os_init,
  .run    = os_run,
  .trap   = os_trap, 
  .on_irq = os_on_irq,
};
