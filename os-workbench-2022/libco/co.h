struct co* co_start(const char *name, void (*func)(void *), void *arg);
void co_yield();
void co_wait(struct co *co);
//#define DEBUG //debug文件开关
