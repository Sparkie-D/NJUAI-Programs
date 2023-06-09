#include <os.h>
static void kmt_init(){
    //初始化一些重要的全局变量，比如current线程
}

static int kmt_create(task_t *task, const char *name, void (*entry)(void *arg), void *arg){
    // TODO
    return 0;
}

static void kmt_teardown(task_t *task){
    // TODO
} 

static void kmt_spin_init(spinlock_t *lk, const char *name){
    // TODO
}

static void kmt_spin_lock(spinlock_t *lk){
    // TODO
    iset(false);
    while(atomic_xchg(lk->lock, LOCKED) != UNLOCK); 
}

static void kmt_spin_unlock(spinlock_t *lk){
    // TODO
    atomic_xchg(lk->lock, UNLOCK);
}

static void kmt_sem_init(sem_t *sem, const char *name, int value){
    // TODO
    sem->name  = name;
    sem->value = value;
    kmt_spin_init(sem->lock, sem->name);
}

static void kmt_sem_wait(sem_t *sem){
    // TODO
    kmt_spin_lock(&sem->lock);
    sem->value --;
    if(sem->value <= 0){
        //没有资源，需要等待
        
    }
    kmt_spin_unlock(&sem->lock);
}

static void kmt_sem_signal(sem_t *sem){
    // TODO
}


MODULE_DEF(kmt) = {
    .init        = kmt_init,
    // 进程创建
    .create      = kmt_create,
    .teardown    = kmt_teardown,
    // 自旋锁
    .spin_init   = kmt_spin_init,
    .spin_lock   = kmt_spin_lock,
    .spin_unlock = kmt_spin_unlock,
    // 信号量
    .sem_init    = kmt_sem_init,
    .sem_wait    = kmt_sem_wait,
    .sem_signal  = kmt_sem_signal,
};
