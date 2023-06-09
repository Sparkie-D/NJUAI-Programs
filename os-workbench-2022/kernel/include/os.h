#include <common.h>

struct task {
  // TODO
};

struct spinlock {
  // TODO
  char *name;
  int   lock;
};

struct semaphore {
  // TODO
  char *     name;  //unique attribute
  int32_t    value; //信号量的个数
  spinlock_t lock;

};
