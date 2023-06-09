import java.lang.reflect.Array;

public class series {
    public static void MergeSort(int[] Array, int head, int tail){
        if(head == tail){
            return;
        }
        else{
            int mid = (head + tail) / 2;
            MergeSort(Array, head, mid);
            MergeSort(Array, mid+1, tail);
            Merge(Array, head, mid, tail);
        }
    }

    public static void QuickSort(int[] Array, int head, int tail){
//        display(Array, head, tail);
        if(head < tail){
            int idx = Partition(Array, head, tail);
            QuickSort(Array, head, idx-1);
            QuickSort(Array, idx+1, tail);
        }
    }

    public static void EnumSort(int[]Array, int head, int tail){
        int[] res = new int[tail+1];
        int pos = 0, cnt = 0;
        for(int i = head; i <= tail; i ++){
            pos = 0;
            cnt = Count(Array, i);
            for(int j = 0; j < Array.length; j ++){
                if(Array[i] > Array[j]) {
                    pos++;
                }
            }
            for(int j = 0; j < cnt; j ++){
                res[pos + j] = Array[i];
            }
        }
        int idx = 0;
        while(idx < Array.length) Array[idx] = res[idx++];
    }
    public static void display(int[] Array, int head, int tail){
        for(int i = head; i <= tail; i ++){
            System.out.print(Array[i]);
            if(i != tail)
                System.out.print(",");
        }
        System.out.println();
    }

    public static void Merge(int[] Array, int head, int mid, int tail){
//        System.out.print("Merge:");
//        display(Array, head, mid);
//        display(Array, mid+1, tail);
//        System.out.println();
        int i = head, j = mid + 1;
        int[] res = new int[tail - head + 1];
        int idx = 0;
        while(i <= mid && j <= tail){
            if(Array[i] <= Array[j]){
                res[idx++] = Array[i++];
            }
            else{
                res[idx++] = Array[j++];
            }
        }
        while(i <= mid)  res[idx++] = Array[i++];
        while(j <= tail) res[idx++] = Array[j++];
        idx -- ;
        for(; idx>=0; idx--) Array[head+idx] = res[idx];
//        System.out.print("result:");
//        display(Array, head, tail);
//        System.out.println();
    }

    public static int Partition(int[] Array, int head, int tail){
        if(head == tail) return head;
        int pivot = Array[tail];
        int i = head - 1;
        for(int j = head; j < tail; j++){
            if(Array[j] <= pivot){
                Swap(Array, ++i, j);
            }
        }
        Swap(Array, i + 1, tail);
        return i + 1;
    }

    public static void Swap(int[] Array, int idx1, int idx2){
        int tmp = Array[idx1];
        Array[idx1] = Array[idx2];
        Array[idx2] = tmp;
    }

    public static int Count(int[] Array, int idx){
        int cnt = 0;
        for(int i = 0; i < Array.length; i ++){
            if(Array[i] == Array[idx]){
                cnt ++;
            }
        }
        return cnt;
    }

    public static void main(String[] args) {
        int[] lis = {3, 4, 2, 1, 6, 10, 203, 43, 2222, 13, 343434};
        MergeSort(lis, 0, lis.length-1);
        QuickSort(lis, 0, lis.length-1);
        EnumSort(lis, 0, lis.length-1);
        display(lis, 0, lis.length-1);
    }

}
