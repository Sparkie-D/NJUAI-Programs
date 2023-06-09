import java.io.*;
import java.util.*;

public class DataProcess {
    public static int[] load(String path) throws IOException {
//        String path = "E:\\junior\\ParallelComputing\\project\\java\\random.txt";
        int[] random = null;
        try (
                FileReader reader = new FileReader(path);
                BufferedReader br = new BufferedReader(reader) // 建立一个对象，它把文件内容转成计算机能读懂的语言
        ) {
            String line;
            while ((line = br.readLine()) != null) {
                // 一次读入一行数据
                String[] s =line.split(" ");
                random = Arrays.stream(s).mapToInt(Integer::valueOf).toArray();
//                System.out.println(random);
            }
        }
         catch (IOException e) {
            e.printStackTrace();
        }
        return random;
    }

    public static void store(int[] arr, String path) throws IOException {
        FileWriter out = null;
        File file = new File(path);
        out = new FileWriter(file);
        for(int i = 0; i < arr.length; i ++){
            String text = String.valueOf(arr[i]) + " ";
            out.write(text);
        }
        out.close();
    }



    public static void main(String[] args) throws IOException {
        // 运行6种排序算法，并保存order*.txt
        String loadPath  = "E:\\junior\\ParallelComputing\\project\\java\\random.txt";
        String storePath;
        int[] random = load(loadPath);
        long start, end;
        int cnt = 10; //取平均值
        long total_time = 0;

//        // Series
        // MergeSort
        random = load(loadPath);
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order1.txt";
        for(int i = 0; i < cnt; i ++){
            start = System.nanoTime();   //获取开始时间
            series.MergeSort(random, 0, random.length - 1);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Series Merge Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);
        // QuickSort
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order2.txt";
        for(int i = 0; i < cnt; i ++){
            random = load(loadPath);
            start = System.nanoTime();   //获取开始时间
            series.QuickSort(random, 0, random.length - 1);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Series Quick Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);
        // EnumSort
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order3.txt";
        for(int i = 0; i < cnt; i ++){
            random = load(loadPath);
            start = System.nanoTime();   //获取开始时间
            series.EnumSort(random, 0, random.length - 1);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Series Enum Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);

        // Parallel
        // MergeSort
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order4.txt";
        for(int i = 0; i <cnt; i ++){
            random = load(loadPath);
            start = System.nanoTime();   //获取开始时间
            parallel.Parallel_MergeSort.MergeSort(random, 10);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Parallel Merge Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);
        // QuickSort
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order5.txt";
        for(int i = 0; i < cnt; i++){
            start = System.nanoTime();   //获取开始时间
            parallel.Parallel_QuickSort.QuickSort(random);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Parallel Quick Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);
        // EnumSort
        total_time = 0;
        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order6.txt";
        for(int i = 0; i < cnt; i ++){
            random = load(loadPath);
            start = System.nanoTime();   //获取开始时间
            parallel.Parallel_EnumSort.EnumSort(random);
            end = System.nanoTime();
            total_time += end - start;
        }
        System.out.println("Parallel Enum Sort Runtime： "+total_time/cnt+"ns");
        store(random, storePath);
        // MergeSort_2
//        total_time = 0;
//        storePath = "E:\\junior\\ParallelComputing\\project\\java\\Order\\order7.txt";
//        for(int i = 0; i < cnt; i ++){
//            random = load(loadPath);
//            start = System.nanoTime();   //获取开始时间
//            parallel.Parallel_MergeSort_2.MergeSort(random);
//            end = System.nanoTime();
//            total_time += end - start;
//        }
//        System.out.println("Parallel Merge Sort Runtime： "+total_time/cnt+"ns");
//        store(random, storePath);
    }
}
