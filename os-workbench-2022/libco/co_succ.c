#ifdef DEBUG
#include "co.h"
#include <stdint.h>
#include <stdio.h>
#include <setjmp.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

//#define DEBUG
//#define CORRECTNESS
#define STACK_SIZE 1024*64
//1024b * 64 = 64kb

static inline void stack_switch_call(void *sp, void *entry, uintptr_t arg) {
  asm volatile (
#if __x86_64__
    "movq %0, %%rsp; movq %2, %%rdi; jmp *%1" : : "b"((uintptr_t)sp), "d"(entry), "a"(arg)
#else
    "movl %0, %%esp; movl %2, 4(%0); jmp *%1" : : "b"((uintptr_t)sp - 8), "d"(entry), "a"(arg)
#endif
  );
}
enum co_status {
  CO_NEW = 1, // 新创建，还未执行过
  CO_RUNNING, // 已经执行过
  CO_WAITING, // 在 co_wait 上等待
  CO_DEAD,    // 已经结束，但还未释放资源
};

struct co {
  const char *name;
  void (*func)(void *);     // co_start 指定的入口地址和参数
  void *arg;
  void *rsp;
  enum co_status status;    // 协程的状态
  struct co *    waiter;      // 是否有其他协程在等待当前协程
  struct co *    next;
  struct co *    poolnext;  //RUNNABLE进程
  struct co *    poolprev;  //NEW进程
  struct co *    prev;
  jmp_buf        context;   // 寄存器现场 (setjmp.h)
  uint8_t        stack[STACK_SIZE]__attribute__((aligned(16))); // 协程的堆栈
};

struct co * current;  // 当前正在运行的协程
struct co * head;     // 头指针
struct co * co_queue;  // 维护一个等待队列

void pool_display(){
  printf("display pool:\n");
  struct co * cur = co_queue;
  if(cur == NULL) return;
  while(1){
    printf("(%s,",cur->name);
    switch(cur->status){
      case CO_NEW:    printf("NEW)");break;
      case CO_RUNNING:printf("RUNNING)");break;
      case CO_WAITING:printf("WAITING)");break;
      case CO_DEAD:   printf("DEAD)");break;
    }
    cur = cur->poolnext;
    if(cur != co_queue) printf("->");
    else break;
  };
  printf("\n");
}

void co_push(struct co * co_this){
  //进程队列，队尾入栈，队首出栈 new节点队首入栈
  if(co_queue == NULL){
    //空池
    co_queue = co_this;
    co_queue->poolnext = co_queue;
    co_queue->poolprev = co_queue;
  }
  else if(co_queue->poolprev == NULL || co_queue->poolnext == co_queue){
    //单节点池
    co_this->poolnext  = co_queue;
    co_this->poolprev  = co_queue;
    co_queue->poolprev = co_this;
    co_queue->poolnext = co_this;
  }
  else{
    if(co_this->status == CO_RUNNING){
      struct co * prev   = co_queue->poolprev;
      co_queue->poolprev = co_this;
      prev->poolnext     = co_this;
      co_this->poolprev  = prev;
      co_this->poolnext  = co_queue;
    }
    else{
      assert(co_this->status == CO_NEW);
      struct co *next    = co_queue->poolnext;
      co_queue->poolnext = co_this;
      next->poolprev     = co_this;
      co_this->poolnext  = next;
      co_this->poolprev  = co_queue;
    }
  }
#ifdef DEBUG
  printf("push %s\n",co_this->name);
#endif
}

struct co * co_pop(){
  //从池中pull一个进程出来执行
  struct co * ret;
  if(co_queue == NULL) return NULL;
  else if(co_queue->poolnext == NULL || co_queue->poolnext == co_queue){
    ret = co_queue;
    co_queue = NULL;
  }
  else{
    //pop队首元素
    struct co * prev = co_queue->poolprev;
    struct co * next = co_queue->poolnext;
    ret = co_queue;
    prev->poolnext = next;
    next->poolprev = prev;
    co_queue = next;
  }
#ifdef DEBUG
  printf("pull %s\n",ret->name);
#endif // DEBUG
  return ret;
}

__attribute__((constructor)) void co_init() {
  head = (struct co *) malloc(sizeof(struct co));
  head->name = "head";
  head->status = CO_RUNNING; //  不需要头指针被调用运行，则初始化为RUNNING
  current = head;
  co_queue = NULL;
  //co_push(head);
  srand((unsigned)time(NULL));
}

void co_display(){
  struct co * cur = head;
  if(cur == NULL) assert(0);
  do{
    printf("(%s,",cur->name);
    switch(cur->status){
      case CO_NEW:    printf("NEW)");break;
      case CO_RUNNING:printf("RUNNING)");break;
      case CO_WAITING:printf("WAITING)");break;
      case CO_DEAD:   printf("DEAD)");break;
      default :assert(0);
    }
    cur = cur->next;
    if(cur != head) printf("->");
  }while(cur != head);
  printf("\n");
}

void func_exec() {
  /*
   * 最巧妙的函数
   * 直接解决协程运行完之后各种变量的修改问题
   * 并且stack_switch_call切换过来后，子函数的堆栈还是建立在当前函数的堆栈上的，并不影响
   * */
   current->status = CO_RUNNING;            //准备开始运行当前协程
   current->func(current->arg);             //开始运行
   current->status = CO_DEAD;               //协程运行完，待释放内存
#ifdef DEBUG
    printf("%s done\n", current->name);
#endif
   if(current->waiter != NULL){
     current->waiter->status = CO_RUNNING;  //若有等待者，则解除等待状态，可以执行
     co_push(current->waiter);              //入队
     current->waiter = NULL;
   }
   co_yield();                              //切换到其他协程执行
}

struct co* co_start(const char *name, void (*func)(void *), void *arg){
  struct co * new_co = (struct co*) malloc(sizeof(struct co)); //分配空间
  //初始化
  new_co->name = name;
  new_co->func = func;
  new_co->arg = arg;
  new_co->rsp = new_co->stack + sizeof(new_co->stack); //栈顶，高地址到低地址生长
  new_co->status = CO_NEW;
  new_co->prev = head;
  new_co->poolnext = NULL;
  new_co->poolprev = NULL;
  struct co * next = head->next;
  head->next = new_co;
  new_co->next = next;
  if(next != NULL) {
    next->prev = new_co;
  }
  if(new_co->next == NULL) {
    new_co->next = head;
    head->prev = new_co;
  }
#ifndef CORRECTNESS
  co_push(new_co);
#endif
  
#ifdef DEBUG
  co_display();
#endif
  return new_co;
}

void co_switch() {
  //寻找并切换到新创建的协程执行
#ifndef CORRCTNESS
  if(current->status == CO_RUNNING || current->status == CO_NEW) co_push(current);//当前协程入队
#ifdef DEBUG
  pool_display();
#endif
  struct co *tmp = co_pop();// 出队
#ifdef DEBUG
  pool_display();
#endif
  if(tmp == NULL) current = head;
  else current = tmp;
#endif
#ifdef DEBUG
    printf("current switched to %s\n",current->name);
#endif
}

void co_yield(){
#ifdef DEBUG
  printf("%s enter yield\n",current->name);
  co_display();
#endif
  int val = setjmp(current->context);//
  if(val == 0){
    //  从setjmp执行来的，将要寻找下一个协程执行，这个协程肯定是新创建的协程
    co_switch(); //切换协程封装函数，改变current指针的值,直接指向一个CO_NEW状态的协程
    if(current->status == CO_NEW){
      if(sizeof(void*) == 4) stack_switch_call(current->rsp, func_exec, (uintptr_t)NULL);
      else{
        asm volatile("mov %0,%%rsp"::"b"((uintptr_t)current->rsp));
        func_exec();
      }
    }
    else longjmp(current->context, 1);
  }
  // else :由longjmp跳转回来，此时寄存器已经切换好，不用做任何事
}

void co_wait(struct co *co){
#ifdef DEBUG
  printf("%s enter co_wait for %s\n",current->name, co->name);
#endif
  if(co->status == CO_DEAD){
    struct co *next = co->next, 
              *prev = co->prev;
    prev->next = next;  // prev不可能是NULL，head不会DEAD
    if(next != NULL) 
         next->prev = prev;
    free(co);
  }
  else{
    current->status = CO_WAITING;
    co->waiter = current;
    co_yield();
  }
}
#endif
