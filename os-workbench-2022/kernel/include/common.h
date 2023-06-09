#include <kernel.h>
#include <klib.h>
#include <klib-macros.h>

//pmm相关

//#define DEBUG_PMM
#define TAB_SIZE (sizeof(struct Memory_Block))
struct Memory_Block{
  // 一个内存块： 内存 | 内存头表
  // 前一部分存放内容，后一部分存放链表 --------是否链表放开头更好？由于对齐操作，分配的内存首址在块中的相对位置就不确定了
  bool using;
  size_t size;
  uintptr_t begin;
  uintptr_t end;
  struct Memory_Block *prev;
  struct Memory_Block *next;
};
size_t    alignment(size_t);
void      block_merge(struct Memory_Block * , struct Memory_Block * );  // for pmm_free, pmm_init
uintptr_t block_partition(struct Memory_Block * , size_t, bool );       // for pmm_alloc
#ifdef    DEBUG_PMM
void      display();//输出所有块
#endif
//锁相关
#define   LOCKED 123456
#define   UNLOCK 654321
void      spin_lock(int *);
void      spin_unlock(int *);

//os相关
#define MAX_CALLBACK_SIZE 1024
typedef struct callback{
  int seq;//优先级
  int event;//事件
  handler_t handler;//处理函数
}callback_t;