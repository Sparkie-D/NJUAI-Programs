import javax.print.attribute.standard.RequestingUserName;
import java.util.Collections;
import java.util.Random;
import java.util.Vector;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;
import java.util.concurrent.Semaphore;




public class parallel {
    public static final boolean DEBUG = false; // DEBUG宏，设置为true则有print输出

    public static class Parallel_MergeSort{
        // 使用PSRS技术进行MergeSort
        public static class MergeThread extends Thread{
            // 这里的内容用来接受构造方法传参
            private Thread t;
            private final int index;// 标号
//            public int[] SortTask;// 整个数组
            private final int head;// 这个线程所需要处理的部分的头
            private final int tail;// 这个线程部分的尾
            private final int p;// 选取主元个数
//            public int[] pivotArray;//存放所有主元的数组
//            public int[] pivotFinalArray;//存放剩余的p-1个主元的数组
            private final int phead;// 这个线程拥有的主元数组的头 ptail = phead + p
            public Thread Pchoose; //用来挑选主元的唯一线程
//            public Vector<Vector<Integer>> finalArray; // 最终属于这个线程的内容（获取完成后要替换掉SortTask中的内容）

            // 构造方法
            public MergeThread(String name, int index, int head, int tail, int p,  int phead, Thread Pchoose){
                super(name);
                this.index = index;
//                this.SortTask = task;
                this.head = head;
                this.tail = tail;
                this.p    = p;
//                this.pivotArray = pivotArray;
//                this.pivotFinalArray = pivotFinalArray;
                this.phead = phead;
                this.Pchoose = Pchoose;
//                this.finalArray = finalArray;
            }

            public void run() {
                series.QuickSort(SortTask, head, tail); // 首先串行排序这个部分
                if(DEBUG) {
                    mutex_println("Thread " + index + " [" + head + " ," + tail + "]");
                }
                int i = 0;
                int step = (tail - head + 1) / p; // 步长 ,每组个数
                while(i < p){  // 选出其中的主元 -> 分成p组，每组的最后一个作为主元
                    pivotArray[phead+i] = SortTask[head+step-1 + i * step];
                    i ++;
                }
                if(DEBUG) mutex_println("Thread "+index+ " finished selecting pivots, waiting for Pchoose");

//                System.out.print("pivots:");
//                series.display(pivotArray, 0, pivotArray.length-1);

                PchooseLock.release(); // +1
                try{
                    ChooseFinish.acquire();// 等待主元选择完成后再继续
                    //开始全局交换
                    Vector<Integer> tmp = new Vector<>();
                    if(DEBUG) mutex_println("Thread "+ index + " starting Global Swap");
                    int id = 0; // 主元序号
                    for(i = head; i <= tail; i ++){
                        if(id >= p-1) break;
                        else{
                            if(SortTask[i] <= pivotFinalArray[id]) finalArray.get(id).add(SortTask[i]);
                            else {
                                i--;
                                id++;
                            }
                        }
                    }
                    if(i < 0) i = 0;
                    for(;i <= tail; i ++) finalArray.get(id).add(SortTask[i]);
                    if(DEBUG){
                        try {
                            mutex.acquire();
                            for (int a = 0; a < finalArray.size(); a++) {
                                System.out.print("final " + a + " : ");
                                for (int b = 0; b < finalArray.get(a).size(); b++) {
                                    System.out.print(finalArray.get(a).get(b) + " ");
                                }
                                System.out.println();
                            }
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        } finally {
                            mutex.release();
                        }
                    }
                    finishBeforeGlobalSwap.release(); //  当前线程完成了重新赋值前的所有任务
                    if(DEBUG) mutex_println("Thread " +index + " finished Global Swap");
                    while(!finishBeforeGlobalSwap.tryAcquire());
                    // 所有线程完成全局交换后才能进行
                    if(DEBUG) mutex_println("Thread " + index + " into Reassignment of Task !");
//                    for (i = 0; i < finalArray.get(index).size(); i++) {
//                        SortTask[head + i] = finalArray.get(index).get(i); // 重置SortTask中的内容
//                    }
                    Collections.sort(finalArray.get(index));// 串行完成排序任务
//                    series.QuickSort(SortTask, head, tail);
                    MergeThreadDone.release(); // 发送信号到Pchoose整合
                    finishBeforeGlobalSwap.release(); // 防止阻塞
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }

            public void start(){
                if (t == null){
                    t = new Thread(this);
                    t.start();
                }
            }
        }


        public static class ChooseThread extends Thread{
            private Thread t;
            private int p;
//            public int[] SortTask; // 最终储存结果的地方
//            public int[] pivotArray;//存放所有主元的数组
//            public int[] pivotFinalArray;//存放挑选出的p-1个主元的数组
            public ChooseThread(int p){
                this.p = p;
//                this.pivotFinalArray = pivotFinalArray;
//                this.SortTask = task;
            }
            public void run(){
                //一共 p*p个，分成p组，选每组最后一个
                try {
                    PchooseLock.acquire(); // 获得这把锁，chooseThread开始动
//                    System.out.println(SortTask);
                    if(DEBUG) {
                        mutex_println("Pchoose working!");
                        series.display(pivotArray, 0, pivotArray.length-1);
                    }
                    series.QuickSort(pivotArray, 0, pivotArray.length-1);
                    int idx = 0;
                    int step = pivotArray.length / p;
                    assert(pivotArray.length % p ==0); // 目前不处理不整除现象
                    while (idx < p - 1) {
                        pivotFinalArray[idx] = pivotArray[step-1+ idx*step];
                        idx++;
                    }
                    if(DEBUG){
                        System.out.print("Final Pivots: ");
                        for (int i = 0; i < pivotFinalArray.length; i++) {
                            System.out.print(pivotFinalArray[i] + " ");
                        }
                        System.out.println();
                    }
                    if(DEBUG) mutex_println("Pchoose finished ");
                } catch (InterruptedException e){
                    e.printStackTrace();
                } finally {
                    ChooseFinish.release(p);
                }
                try{
                    MergeThreadDone.acquire(); // 全部子线程做完了，到这里整合
                    Vector<Integer> res = new Vector<>();
                    for(int i = 0; i < finalArray.size(); i ++){
                        for(int j = 0; j < finalArray.get(i).size(); j ++){
                            res.add(finalArray.get(i).get(j));
                        }
                    }
                    for(int i = 0; i < res.size(); i ++) SortTask[i] = res.get(i);

                }catch (InterruptedException e){
                    e.printStackTrace();
                } finally {
                    Alldone.release(p);
                }
            }

            public void start(){
                if (t == null){
                    t = new Thread(this);
                    t.start();
                }
            }
        }
        public static Semaphore mutex;

//        public static int finishBeforePchoose;
        public static Semaphore finishBeforeGlobalSwap;
        public static Semaphore PchooseLock;
        public static Semaphore ChooseFinish;
        public static Semaphore MergeThreadDone;
        public static Semaphore Alldone;
        public static int[] pivotArray;
        public static int[] pivotFinalArray;
        public static int[] SortTask;
        public static Vector<Vector<Integer>> finalArray;
//        public static int[] SortTask;
        public static void MergeSort(int[] task, int processor){
//            System.out.println("Please ensure that 0 < p < Array.length/p and Array.length % p = 0");
            mutex = new Semaphore(1);
            int p = processor; // 线程数
            PchooseLock = new Semaphore(-p+1);
            ChooseFinish = new Semaphore(0);
            Alldone = new Semaphore(0);
            MergeThreadDone = new Semaphore(-p+1);
            finishBeforeGlobalSwap = new Semaphore(-p+1);
            finalArray = new Vector<>(p);
//            System.out.println("[Sort Algorithm] Parallel Merge Sort" );
//            System.out.println("[Input scale   ] " + task.length);
//            System.out.println("[Processor Num ] " + processor);
            for(int i = 0; i < p; i ++) finalArray.add(new Vector<>());
            SortTask = task;
            Vector<MergeThread> threads = new Vector<>();
            MergeThread tmp;
            pivotArray = new int[p * p];
            pivotFinalArray = new int[p-1];
            int head, tail;
            int step = SortTask.length / p;
            Thread Pchoose = new ChooseThread(p);
//            System.out.println(Pchoose);
            Pchoose.start();
            for(int i = 0; i < p; i ++){
                //初始化线程池
                head = i*step;
                tail = (i+1) *step-1;
                tail = tail < SortTask.length ? tail : SortTask.length-1;
                int phead = i * p;
                tmp = new MergeThread("MergeThread-"+i, i, head, tail, p, phead, Pchoose);
                threads.add(tmp);
            }
            for(int i = 0; i < p; i ++) threads.get(i).start();
            while(!Alldone.tryAcquire());
//            series.display(SortTask, 0, SortTask.length-1);
        }

        public static void mutex_println(String s){
            try {
                mutex.acquire();
                System.out.println(s);

            } catch (InterruptedException e) {
                e.printStackTrace();
            } finally {
                mutex.release();
            }
        }
    }

    public static class Parallel_MergeSort_2{

        public static class MergeAction extends RecursiveAction {
            private int head; // 该线程所需要排序的起点
            private int tail; // 终点

            public MergeAction(int head, int tail) {
                this.head = head;
                this.tail = tail;
            }

            public void compute() {
                if (tail - head <= 4) {
                    series.MergeSort(SortTask, head, tail);
                } else {
                    int mid = (head + tail) / 2;
                    MergeAction left = new MergeAction(head, mid);
                    MergeAction right = new MergeAction(mid + 1, tail);
                    invokeAll(left, right);
                    series.Merge(SortTask, head, mid, tail);
                }
            }
        }

        public static int[] SortTask;
        public static Semaphore mutex;
        public static void MergeSort(int[] task) {
            mutex = new Semaphore(1);
            SortTask = task;
            MergeAction mainAction = new MergeAction(0, SortTask.length-1);
            ForkJoinPool pool = new ForkJoinPool();
            pool.invoke(mainAction);
        }

        public static void main(String[] args) {
            int[] task = {1, 3, 5, 7, 2, 6, 5, 1, 8};
            MergeSort(task);
            series.display(task, 0, task.length-1);
        }

    }

    public static class Parallel_QuickSort{

        public static class QuickThread extends Thread{
            private int head; // 该线程所需要排序的起点
            private int tail; // 终点
            private Thread t;

            public QuickThread(int head, int tail){
                this.head = head;
                this.tail = tail;
            }

            public void run(){
                if(head >= tail || head < 0 || tail >= SortTask.length) return;
                else {
                    int pivot = series.Partition(SortTask, head, tail); // 划分后的主元序号
//                    System.out.print(head <= pivot);System.out.print(pivot<=tail);
//                    series.display(SortTask, 0, SortTask.length-1);
                    QuickThread left  = new QuickThread(head, pivot-1);
                    QuickThread right = new QuickThread(pivot + 1, tail);
                    left.start();
                    right.start();
                }
            }

            public void start(){
                if (t == null){
                    t = new Thread(this);
                    t.start();
                }
            }
        }

        public static int[] SortTask;
        public static void QuickSort(int[] task) {
            SortTask = task;
            QuickThread mainThread = new QuickThread(0, SortTask.length-1);
            mainThread.start();
//            System.out.print(task + ": ");
//            series.display(task, 0, task.length-1);
        }

    }

    public static class Parallel_EnumSort {
        //用两个数组记录，一个数组记录输入数，另一个记录对应序号上的数的rank
        public static Semaphore mutex;
        public static class EnumThread  extends Thread{
            private int index;
            private int value;
            private Thread t;

            public EnumThread(int index, int value){

                this.index = index;
                this.value = value;
            }

            public void run(){
                // 计算自己分配到的数的rank
                RankList[index]  = 0;
                CountList[index] = 0;
                for(int i = 0; i < SortTask.length; i ++){
                    if(SortTask[i] < value){
                        RankList[index] ++;
                    }
                    else if(SortTask[i] == value){
                        CountList[index] ++;
                    }
                }
                RankFinish.release();
            }

            public void start(){
                if (t == null){
                    t = new Thread(this);
                    t.start();
                }
            }

        }

        public static class MainThread extends Thread{
            private Thread t;

            public MainThread(){
                return;
            }

            public void run(){
                int[] tmp = new int[SortTask.length];
                int pos = 0;
                int cnt = 0;
                try{
                    RankFinish.acquire();
                    for(int i = 0; i < SortTask.length; i ++){
                        pos = RankList[i];
                        cnt = CountList[i];
                        for(int j = 0; j < cnt; j ++){
                            tmp[pos + j] = SortTask[i];
                        }
                    }
                    for(int i = 0; i < SortTask.length; i ++) SortTask[i] = tmp[i];
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }


            public void start() {
                if (t == null) {
                    t = new Thread(this);
                    t.start();
                }
            }
        }
        public static int[] SortTask;
        public static int[] RankList;
        public static int[] CountList;
        public static Semaphore RankFinish;

        public static void EnumSort(int[] task){
            SortTask = task;
            RankList = new int[task.length];
            CountList = new int[task.length];
            RankFinish = new Semaphore(-task.length+1);
            MainThread mt = new MainThread();
            mt.start();
            EnumThread tmp;
            for(int i = 0; i < task.length; i ++){
                tmp = new EnumThread(i, task[i]);
                tmp.start();
            }
        }

        public static void main(String[] args) {
//            int[] task = {5, 3, 1, 1, 1, 2, 1, 2, 2,3};
            int[] task = new int[500];
            for(int i = 0; i < task.length; i ++) task[i] = (int)(Math.random()*10000);
//            series.display(task,0,task.length-1);
//            series.display(task, 0, task.length-1);
            series.display(task, 0, task.length-1);
            EnumSort(task);
//            System.out.print("result out: ");
//            series.display(task,0,task.length-1);
//            series.display(SortTask,0,task.length-1);
//            series.display(RankList, 0, RankList.length-1);
        }

    }


    public static void main(String[] args){
        int processor = 50;
//        int[] task = {15,46,48,93,39,6,72,91,14,36,69,40,89,61,97,12,21,54,53,97,84,58,32,27,33,72,20, 11, 88, 30};
//        Parallel_MergeSort.MergeSort(task, processor);
//        series.display(task, 0, task.length-1);
//        int NUM = 10;
//        int size = 100000;
//        int[] array = new int[size];
//        for(int i = 0; i < array.length;i++) {
//            array[i] = (int) (Math.random() * 1000);//0~1乘数1000，内容变为0~1000，强转为整型
//        }
//        long start, end;
//        start=System.currentTimeMillis();   //获取开始时间
//        for(int i = 0; i < NUM; i ++) series.MergeSort(array, 0, task.length-1);
//        end=System.currentTimeMillis(); //获取结束时间
//        System.out.println("Series Runtime： "+(end-start)+"ms");
//
//        start=System.currentTimeMillis();   //获取开始时间
//        for(int i = 0; i < NUM; i ++) Parallel_MergeSort.MergeSort(array, processor);
//        end=System.currentTimeMillis(); //获取结束时间
//        System.out.println("Parallel Runtime： "+(end-start)+"ms");
//        SortTask = task;
//        series.display(task, 0, task.length-1);
//        Parallel_QuickSort.QuickSort(task);
//        System.out.println("finished sorting " + task);
//        series.display(SortTask, 0, SortTask.length-1);
//        Parallel_EnumSort.EnumSort(task);
//        series.display(task,0,task.length-1);


    }

}
