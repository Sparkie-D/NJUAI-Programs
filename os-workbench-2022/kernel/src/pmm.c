#include <common.h>
#include <stdbool.h>

//全局变量表
int pmm_lock = UNLOCK;
//struct Memory_Block * root;//万恶的弱定义数据，警钟长鸣
#ifdef DEBUG_PMM
int display_lock = UNLOCK;
#endif
//atomic_xchg:设置新值返回旧值
void spin_lock(int *lock){
  //当返回值为锁值时，lock被设置为新值
  while(atomic_xchg(lock, LOCKED) != UNLOCK); 
}

void spin_unlock(int *lock){
  atomic_xchg(lock, UNLOCK);
}

static void pmm_init(){
    uintptr_t pmsize = ((uintptr_t)heap.end - (uintptr_t)heap.start);
    printf("Got %d MiB heap: [%p, %p)\n", pmsize >> 20, heap.start, heap.end);
    //初始化块
    struct Memory_Block * head = (struct Memory_Block *)(heap.end - TAB_SIZE);
    //我没有给root确定的位置！！！！！！指针飞了！
    head->prev  = NULL;
    head->next  = NULL;
    head->using = false;
    head->begin = (uintptr_t)heap.start;
    head->end   = (uintptr_t)heap.end;
    head->size  = (uintptr_t)(head->end - head->begin - TAB_SIZE);
    //初始化多个空块
    int count   = cpu_count();
    count = alignment(count);//便于对齐，注释则出现bug
    size_t smpsz= head->size / count;
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("CPU count = %d, allocate %d for each.\n",count, smpsz);
    spin_unlock(&display_lock);
#endif
    for(int i = 0; i < count - 1; i ++){
        //分配为CPU(CPU<=4)个同等大小的块，在CPU抢占内存时可以同时执行
        block_partition(head, smpsz, false);
        head = head->prev;
    }
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("************After initialization************\n");
    spin_unlock(&display_lock);
    display();
#endif
}

#ifdef DEBUG_PMM

size_t lenth(size_t index){
    int res = 0;
    do{
        res ++;
        index /= 10;
    }while(index > 0);
    return res == 0 ? 1 : res;
}

void display(){
    spin_lock(&display_lock);
    struct Memory_Block *cur = (struct Memory_Block *)(heap.end - TAB_SIZE);
    int cnt=0;
    int lnth=0;
    int align=0;
    printf("  address    index         allocation      using   align    size\n");
    while(cur != NULL){
        //规范化输出
        lnth = 7 - lenth(cnt);
        if((cur->begin % alignment(cur->require)) == 0){ align = 1;}
        else {align = 0;}
        printf("[%p]    %d",cur, cnt++);
        for(int i = 0; i < lnth; i ++) printf(" ");
        printf("%p",cur->begin);       
        if(cur->begin == 0x300000) printf(" ");
        printf("-%p   ", cur->end);
        printf(" %d       %d      %d\n", cur->using, align, cur->size);
        assert(cur->begin <= cur->end);
        cur = cur->prev;
    }
    spin_unlock(&display_lock);
}
#endif // DEBUG_PMM

size_t alignment(size_t size){
    // 对齐最小单位
    size_t res = 1;
    while(res < size) res = res << 1;
    return res;
}

uintptr_t block_partition(struct Memory_Block *block, size_t size, bool alloc){
    // 将当前块一分为二，前一半未分配，后一半分配给它（保证后一半的地址对齐）(如果要分配的话)
    assert(block->size > size);
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("Parting block %p-%p\n",block->begin, block->end);
    spin_unlock(&display_lock);
#endif // DEBUG_PMM
    // 计算最终的划分地址
    uint32_t align = alignment(size);
    uintptr_t addr = block->end - size - TAB_SIZE;
    while(addr % align != 0) addr --;
    if(addr - TAB_SIZE <= block->begin) {
#ifdef DEBUG_PMM
        spin_lock(&display_lock);
        printf("No enough space for link table! Parting failed.\n");
        spin_unlock(&display_lock);
#endif
        return 0;//对齐后不够大了
    }
    // 保存原来块的信息
    uintptr_t            blk_begin = block->begin;
    uintptr_t            blk_end   = block->end;
    struct Memory_Block *blk_prev  = block->prev;
    struct Memory_Block *blk_next  = block->next;
    // 在原来块前插入一个新块
    struct Memory_Block * befor = (struct Memory_Block *)(addr - TAB_SIZE); 
    //struct Memory_Block * after = (struct Memory_Block *)(blk_end - (sizeof(struct Memory_Block)));
    struct Memory_Block * after = block;
    // 给两个块赋值
    befor->begin = blk_begin;
    befor->end   = addr;
    befor->size  = befor->end - befor->begin - TAB_SIZE;
    befor->using = false;
    after->begin = addr;
    after->end   = blk_end;
    after->size  = after->end - after->begin - TAB_SIZE;
    if(alloc) after->using = true;
    else      after->using = false;
#ifdef DEBUG_PMM
    befor->require = 0;
    after->require = size;
#endif
    // 处理指针情况
    if(blk_prev != NULL) blk_prev->next = befor;
    befor->next = after;
    //after->next = blk_next;
    if(blk_next != NULL) blk_next->prev = after;
    after->prev = befor;
    befor->prev = blk_prev;
#ifdef DEBUG_PMM
    //spin_lock(&display_lock);
    //printf("new block 1: %p-%p, size:%d, using:%d\n",befor->begin, befor->end, befor->size, befor->using);
    //printf("new block 2: %p-%p, size:%d, using:%d\n",after->begin, after->end, after->size, after->using);
    //spin_unlock(&display_lock);
#endif
    assert(befor->begin < befor->end);
    assert(after->begin < after->end);
    return after->begin;
    
}

static void *kalloc(size_t size) {
    spin_lock(&pmm_lock);
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("************malloc memory for size %d************\n", size);
    spin_unlock(&display_lock);
#endif
    struct Memory_Block *current = (struct Memory_Block *)(heap.end - TAB_SIZE);//最后一块的地址确定，由此向前推
    void * ret = NULL;
    while(current != NULL){
        if(!current->using){
            if(current->size >= size && current->size <= size + TAB_SIZE * 2){
                //当前块全部分配给它
                current->using = true;
#ifdef DEBUG_PMM
                spin_lock(&display_lock);
                printf("Successfully allocate at %p, full block\n",current->begin);
                spin_unlock(&display_lock);
                display();
#endif
                spin_unlock(&pmm_lock);
                return (void *)current->begin;
            }
            else if(current->size > size + sizeof(current)){
                //将当前块一分为二，前一半未分配，后一半分配给它（保证后一半的地址对齐）
                current->using = true;//防止其他CPU来访问这一块，partition中会恢复不使用的部分
                uintptr_t res = block_partition(current, size, true);//这个部分可以并发进行
                if(res == 0){
                    //划分失败，重新进循环找可用块 
                    current = current->prev;
                    continue;
              }
#ifdef DEBUG_PMM
                spin_lock(&display_lock);
                printf("Successfully allocate at %p, part block\n",current->begin);
                spin_unlock(&display_lock);
                display();
#endif // DEBUG_PMM
                spin_unlock(&pmm_lock);
                return (void *)res;
            }
        }
        // 当前块小于size或正使用
        current = current->prev;
    }
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("Malloc failed.\n");
    spin_unlock(&display_lock);
    display();
#endif
    spin_unlock(&pmm_lock);
    return ret;//遍历完还没成功分配
}

void block_merge(struct Memory_Block *block1, struct Memory_Block *block2){
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("Merge block at %p-%p and %p-%p\n",block1->begin, block1->end, block2->begin, block2->end);
    spin_unlock(&display_lock);
#endif
    assert(block1->begin <= block2->begin);
    assert(!block1->using);
    assert(!block2->using);
    if(block1->begin == block2->begin) return;//是同一块
    //默认把1merge到2上
    block2->using = false;
    block2->size += block1->size + TAB_SIZE;
    block2->begin = block1->begin;
    block2->prev = block1->prev;
    if(block1->prev != NULL) block1->prev->next = block2;
}

static void kfree(void *ptr) {
    spin_lock(&pmm_lock);
    struct Memory_Block *block = (struct Memory_Block *)(heap.end - TAB_SIZE);
    while(block->prev != NULL && (uintptr_t)ptr < block->begin) block = block->prev;
#ifdef DEBUG_PMM
    spin_lock(&display_lock);
    printf("************Free memory at %p************\n", block);
    spin_unlock(&display_lock);
#endif
    if(!block->using){
#ifdef DEBUG_PMM
        spin_lock(&display_lock);
        printf("Unused address! Failed.\n");
        spin_unlock(&display_lock);
        display();
#endif
    }
    else{
        //merge空闲块
        block->using = false;//回收
        if(block->prev != NULL && !block->prev->using) block_merge(block->prev, block);
        if(block->next != NULL && !block->next->using) block_merge(block, block->next);
#ifdef DEBUG_PMM
        spin_lock(&display_lock);
        printf("Free success.\n");
        spin_unlock(&display_lock);
        display();
#endif
    }
    spin_unlock(&pmm_lock);
}

static void *kalloc_safe(size_t size) {
    bool i = ienabled();
    iset(false);
    void *ret = kalloc(size);
    if (i) iset(true);
    return ret;
}

static void kfree_safe(void *ptr) {
    int i = ienabled();
    iset(false);
    kfree(ptr);
    if (i) iset(true);
}

MODULE_DEF(pmm) = {
  .init  = pmm_init,
  .alloc = kalloc_safe,
  .free  = kfree_safe,
};
