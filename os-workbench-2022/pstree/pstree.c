#include <stdio.h>
#include <assert.h>
#include <string.h>

int atoi(char *);

typedef struct Process{ 
  char name[256];
  int index;
  int pid;
  int ppid;
  int threads;
  int display;
  int thread[20]; //这里的线程是子进程的意思，当时理解不到位
  int nchildren;  //这里的孩子是父进程为当前进程的进程，区别于子进程
  struct Process *children[100];
}Process; 

//Process *all_proc[1000];
Process all_proc[1000];
int proc_index = 0;

/*  原main函数
int main(int argc, char *argv[]) {
  for (int i = 0; i < argc; i++) {
    assert(argv[i]);
    printf("argv[%d] = %s\n", i, argv[i]);
  }
  assert(!argv[argc]);
  return 0;
}
*/

int size(char *line){
  int i = 0;
  int cnt = 0;
  while(line[i] !=' ' && line[i + 1] != ' '){
    cnt ++;
    i ++;
  }
  return cnt;
}

void read_shell_thread(char *addr, Process *cur);
/*
void subthread_sort(Process *cur) {  // basic bubble sort
  int N = cur->threads;
  if(N == 1) return;
  for(int i = 0; i < N; i ++){
    for(int j = i; j < N; j ++) {
      if(cur->thread[i] > cur->thread[j]){
        //int *curi = &cur->thread[i];
        //int *curj = &cur->thread[j];
        //int tmp = *curi;
        // *curi = *curj;
        // *curj = tmp;
        int tmp = cur->thread[i];
        cur->thread[i] = cur->thread[j];
        cur->thread[j] = tmp;
      }
    }
  }
}

*/
//给当前进程赋值
void load_procfs(int pid, Process * cur) {
  char addr[256] = "/proc/";
  char strpid[256];
  sprintf(strpid, "%d", pid);
  strcat(addr, strpid);
  char thread_addr[256];
  strcpy(thread_addr, addr);
  strcat(addr, "/status");//  生成 /proc/pid/status路径
  strcat(thread_addr, "/task");// 生成 /proc/pid/task路径，其中目录为进程的进程号 
  //printf("%s\n",addr);
  cur->display = 0;
  read_shell_thread(thread_addr, cur);
  //subthread_sort(cur);
  FILE *fp = NULL;
  fp = fopen(addr, "r");
  cur->pid = pid;
  assert(fp != NULL);
  char line[256], type[256], tmp[256];
  while(fgets(line, 256, fp) != NULL){
    assert(fp != NULL);
    sscanf(line, "%s %s", type, tmp);
    if(strcmp(type, "Name:") == 0){
      strncpy(cur->name, &line[6], strlen(line)-5);
      cur->name[strlen(cur->name) - 1] = '\0';
      int siz = strlen(line);
      //printf("size of line = %d\n",siz);
      //printf("proc_index = %d %s pid = %d",proc_index, cur->name,cur->pid);
    }
    if(strcmp(type, "PPid:") == 0){
      cur->ppid = atoi(tmp);
      //printf(" ppid = %d\n",cur->ppid);
    }
    if(strcmp(type, "Threads:") == 0){
      cur->threads = atoi(tmp);
      break;
    }
  }
  fclose(fp);
}

void create_addr(char *dest, char *addr){
  char instr[1024]="cd && cd ";
  strcat(instr,addr);
  strcat(instr," && ls -d [0-9]*"); //cd && cd addr && ls -d [0-9]*
  strcpy(dest, instr);
}

void load_info(FILE *fp){
  char out[100000];
  //fread(out, sizeof(char), 100000, fp);
  //fgets(out, sizeof(out), fp);
  int cnt = 0;
  assert(fp != NULL);
  while(fgets(out, sizeof(out), fp) != NULL){
    //此时的out中为当前读到的进程号pid
    assert(fp != NULL);
    Process cur;
    all_proc[proc_index] = cur;
    all_proc[proc_index].index = proc_index; 
    load_procfs(atoi(out), &cur);
    proc_index ++;
  }
}

void display_idx(int idx);

void load_thread(FILE *fp, Process *cur) {
  char out[100000];
  assert(fp != NULL);
  int i = 0;
  while(fgets(out, sizeof(out), fp) != NULL) {
    assert(fp != NULL);
    cur->thread[i] = atoi(out);
    i ++;
    //memset(out,'\0', 10);
  }
  //subthread_sort(cur);
  //display_idx(cur->pid);
}

void display_idx(int pid){
  Process *cur = NULL;
  for(int i = 0; i < proc_index; i ++){
    if(all_proc[i].pid == pid){
      cur = &all_proc[i];
      break;
    }
  }
  if(cur == NULL) assert(0);
  printf("threads = %d\n", cur->threads);
  for(int i = 0; i < cur->threads; i ++){
    printf("%d\t", cur->thread[i]);
    if(i % 5 == 0)  printf("\n");
  }
}

//从shell中读入目录信息并将所有已有值的进程放入all_proc数组
void read_shell_info(char *addr){
  char instr[1024];
  create_addr(instr, addr);
  FILE *fp;
  fp = popen(instr, "r");
  load_info(fp);
  pclose(fp);
}

//从shell中读入子进程信息并存入当前父进程
void read_shell_thread(char *addr, Process *cur){
  char instr[1000000];
  create_addr(instr, addr);
  FILE *fp;
  fp = popen(instr, "r");
  load_thread(fp, cur);
  //subthread_sort(cur);
  pclose(fp);
}

void print_info_p(int idx, int lvl){// 打印进程号的输出实现 注意：最后没有换行符（便于衔接输出）
  all_proc[idx].display = 1;
  for(int i = 0; i < lvl; i ++)
    printf("  ");
  //printf("%s(%d)\n", all_proc[idx]->name, all_proc[idx]->pid);  
  char line[1000];
  sprintf(line,"%s(%d)   ", all_proc[idx].name, all_proc[idx].pid); 
  printf("%s",line);
  if(all_proc[idx].threads < 2){
    printf("\n");
    return;
  }
  else{
    printf("{%s}(%d)", all_proc[idx].name, all_proc[idx].thread[1]);  
  }
  for(int i = 2; i < all_proc[idx].threads; i ++){
    printf("\n");
    for(int i = 0; i < lvl; i ++)
      printf("  ");
    for(int i = 0; i < strlen(line); i ++) printf(" ");
    printf("{%s}(%d)", all_proc[idx].name, all_proc[idx].thread[i]);  
  }
  printf("\n");
}

void print_info(int idx, int lvl){  //不打印进程号的输出实现
  all_proc[idx].display = 1;
  for(int i = 0; i < lvl; i ++)
    printf("  ");
  //printf("%s(%d)\n", all_proc[idx]->name, all_proc[idx]->pid);
  printf("%s", all_proc[idx].name);
  if(all_proc[idx].threads < 2) {
    printf("\n");
    return;
  }
  if(all_proc[idx].threads - 1 == 1)
    printf("---{%s}\n",all_proc[idx].name);
  else 
    printf("---%d*{%s}\n", all_proc[idx].threads - 1,all_proc[idx].name);
}


void display_p(int idx, int lvl) { //需要打印每个进程的进程号 (乱序)
  print_info_p(idx, lvl);
 // printf("\n");
  for(int i = idx+1; i < proc_index; i ++){
    //if(all_proc[i]->ppid == all_proc[idx]->pid)
    if(all_proc[i].ppid == all_proc[idx].pid && all_proc[i].display == 0)
      display_p(i, lvl + 1);
  }
}

void display(int idx, int lvl) {//输出一个直接孩子，乱序且无进程号
  print_info(idx, lvl);
  for(int i = idx+1; i < proc_index; i ++){
    //if(all_proc[i]->ppid == all_proc[idx]->pid)
    if(all_proc[i].ppid == all_proc[idx].pid && all_proc[i].display == 0)
      display(i, lvl + 1);
  }
}

void all_sort() {
  //存入孩子数据（区别于子进程数）
  for(int i = 0; i < proc_index; i ++) {
    all_proc[i].nchildren = 0;
    for(int j = 0; j < proc_index ; j ++) {
      if(all_proc[j].ppid == all_proc[i].pid){
        all_proc[i].children[all_proc[i].nchildren] = &all_proc[j];
        all_proc[i].nchildren ++;
      }
    }
  }
  //printf("proc_index = %d\n",proc_index);
  for(int i = 0; i < proc_index; i ++){
    if(all_proc[i].pid  == 2) continue;
    //printf("name = %s pid = %d nchildren = %d\n",all_proc[i].name,all_proc[i].pid,all_proc[i].nchildren);
    for(int j = 0; j < all_proc[i].nchildren; j ++){
      for(int k = j; k < all_proc[i].nchildren; k ++){
        //printf("children j = %d k =%d\n",j,k);
        if(all_proc[i].children[j] == NULL)assert(0);
        Process *proj = all_proc[i].children[j];
        Process *prok = all_proc[i].children[k];
        if(proj->pid > prok->pid){
          Process *tmp = all_proc[i].children[j];
          all_proc[i].children[j] = all_proc[i].children[k];
          all_proc[i].children[k] = tmp;
        }
      }
    }
  }
}

void show(int idx, int lvl){
  print_info(idx, lvl);
  for(int i = 0; i < all_proc[idx].nchildren; i ++) {
    show(all_proc[idx].children[i]->index, lvl + 1);
  }
}

void display_n(){
  all_sort();
  show(1, 0);
}

void show_np(int idx, int lvl) {
  print_info_p(idx, lvl);
  for(int i = 0; i < all_proc[idx].nchildren; i ++) {
    show_np(all_proc[idx].children[i]->index, lvl + 1);
  }
}

void display_n_p() {
 all_sort();
 show_np(1,0);
}

/*
void debug_display() {
  for(int i = 1; i < proc_index; i ++){
    print_info(i, 0); 
  }
  printf("proc_index = %d\n",proc_index);
}
*/
void display_V() {
  char s[1024] = "pstree (PSmisc) UNKNOWN\nCopyright (C) 2022 Sparky\nPSmisc comes with ABSOLUTELY NO WARRANTY.\nThis is free software, and you are welcome to redistribute it under\nthe terms of the GNU General Public License.\nFor more information about these matters, see the files named COPYING.\n";
  printf("%s", s);
}

//测试用main函数
int main(int argc, char *argv[]){
  //printf("argc = %d\n",argc);
  for(int i = 0; i < argc; i ++){
    assert(argv[i]);
    //printf("argv[%d] = %s\n", i, argv[i]);
  }
  assert(!argv[argc]);
  read_shell_info("/proc");
  //display_idx(1);
  //for(int i = 0; i < proc_index; i ++){
    //printf("pid = %d\t",all_proc[i].pid);
  //}
  if (argc == 1) display(1, 0);
  else if(argc == 2 && (strcmp(argv[1], "-V") == 0 || strcmp(argv[1], "--version") == 0)) display_V();
  else if(argc == 2 && (strcmp(argv[1], "-p") == 0 || strcmp(argv[1], "--show-pids") == 0)) display_p(1, 0);
  else if(argc == 2 && (strcmp(argv[1], "-n") == 0 || strcmp(argv[1], "--numeric-sort") == 0)) display_n();
  else if(argc == 3){
    int flag = 0;
    if(strcmp(argv[1], "-n") == 0 && strcmp(argv[2], "-p") == 0) flag = 1;
    else if(strcmp(argv[2], "-n") == 0 && strcmp(argv[1], "-p") == 0) flag = 1;
    else if(strcmp(argv[1], "--numeric-sort") == 0 && strcmp(argv[2], "--show-pids") == 0) flag = 1;
    else if(strcmp(argv[2], "--numeric-sort") == 0 && strcmp(argv[1], "--show-pids") == 0) flag = 1;
    else flag = -1;
    if(flag > 0) display_n_p();
    else assert(0);
  }
  else assert(0);
  //debug_display();
  return 0;
}
















