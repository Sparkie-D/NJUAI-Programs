#ifndef DEBUG
#include "co.h"
#include <stdint.h>
#include <stdio.h>
#include <setjmp.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

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
  struct co *    waiter;    // 是否有其他协程在等待当前协程
  struct co *    next;      //RUNNABLE进程
  struct co *    prev;      //NEW进程
  jmp_buf        context;   // 寄存器现场 (setjmp.h)
  uint8_t        stack[STACK_SIZE]__attribute__((aligned(16))); // 协程的堆栈
};

struct co * current;  // 当前正在运行的协程
struct co * head;     // 头指针
struct co * co_queue;  // 维护一个等待队列

void co_push(struct co * co_this){
  //进程队列，队尾入栈，队首出栈 new节点队首入栈
  if(co_queue == NULL){
    //空池
    co_queue = co_this;
    co_queue->next = co_queue;
    co_queue->prev = co_queue;
  }
  else if(co_queue->prev == NULL || co_queue->next == co_queue){
    //单节点池
    co_this->next  = co_queue;
    co_this->prev  = co_queue;
    co_queue->prev = co_this;
    co_queue->next = co_this;
  }
  else{
    if(co_this->status == CO_RUNNING){
      struct co * prev   = co_queue->prev;
      co_queue->prev = co_this;
      prev->next     = co_this;
      co_this->prev  = prev;
      co_this->next  = co_queue;
    }
    else{
      assert(co_this->status == CO_NEW);
      struct co *next    = co_queue->next;
      co_queue->next = co_this;
      next->prev     = co_this;
      co_this->next  = next;
      co_this->prev  = co_queue;
    }
  }
}

struct co * co_pop(){
  //从池中pull一个进程出来执行
  struct co * ret;
  if(co_queue == NULL) return NULL;
  else if(co_queue->next == NULL || co_queue->next == co_queue){
    ret = co_queue;
    co_queue = NULL;
  }
  else{
    //pop队首元素
    struct co * prev = co_queue->prev;
    struct co * next = co_queue->next;
    ret = co_queue;
    prev->next = next;
    next->prev = prev;
    co_queue = next;
  }

  return ret;
}

__attribute__((constructor)) void co_init() {
  head = (struct co *) malloc(sizeof(struct co));
  head->name = "head";
  head->status = CO_RUNNING; //  不需要头指针被调用运行，则初始化为RUNNING
  current = head;
  co_queue = NULL;
  srand((unsigned)time(NULL));
}


void func_exec() {

   current->status = CO_RUNNING;            //准备开始运行当前协程
   current->func(current->arg);             //开始运行
   current->status = CO_DEAD;               //协程运行完，待释放内存

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
  new_co->next = NULL;
  new_co->prev = NULL;

  co_push(new_co);

  return new_co;
}

void co_switch() {
  //寻找并切换到新创建的协程执行
  if(current->status == CO_RUNNING || current->status == CO_NEW) co_push(current);//当前协程入队
  struct co *tmp = co_pop();// 出队
  if(tmp == NULL) current = head;
  else current = tmp;
}

void co_yield(){
  int val = setjmp(current->context);//
  if(val == 0){
    co_switch(); 
    if(current->status == CO_NEW){
      if(sizeof(void*) == 4) stack_switch_call(current->rsp, func_exec, (uintptr_t)NULL);
      else{
        asm volatile("mov %0,%%rsp"::"b"((uintptr_t)current->rsp));
        func_exec();
      }
    }
    else longjmp(current->context, 1);
  }
}

void co_wait(struct co *co){
  if(co->status == CO_DEAD){
    free(co);
  }
  else{
    current->status = CO_WAITING;
    co->waiter = current;
    co_yield();
  }
}
#endif