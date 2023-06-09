'''串行实现'''
import random

class Series:
    def MergeSort(self, lst:list):
        def merge(lst1:list, lst2:list) -> list:
            res = []
            idx1, idx2 = 0, 0
            while True:
                if lst1[idx1] > lst2[idx2]:
                    res.append(lst2[idx2])
                    idx2 += 1
                else:
                    res.append(lst1[idx1])
                    idx1 += 1
                if idx1 == len(lst1):
                    while idx2 < len(lst2):
                        res.append(lst2[idx2])
                        idx2 += 1
                    break
                if idx2 == len(lst2):
                    while idx1 < len(lst1):
                        res.append(lst1[idx1])
                        idx1 += 1
                    break
            return res

        if len(lst) == 1:
            return lst
        cut = int(len(lst)/2)
        lst1 = lst[:cut]
        lst2 = lst[cut:]
        return merge(self.MergeSort(lst1), self.MergeSort(lst2))


    def QuickSort(self, lst:list):
        ''' Inplace !!!'''
        def Swap(lst, i, j):
            tmp = lst[i]
            lst[i] = lst[j]
            lst[j] = tmp

        def InplacePartition(lst, left, right):
            pivot = lst[right]
            i = left - 1
            for j in range(left, right):
                if lst[j] <= pivot:
                    i += 1
                    Swap(lst, i, j)
            Swap(lst, i+1, right)
            return i + 1

        def sort(lst, left, right):
            if left < right:
                idx = InplacePartition(lst, left, right)
                sort(lst, left, idx-1)
                sort(lst, idx+ 1, right)

        sort(lst, 0, len(lst)-1)

    def EnumSort(self, lst):
        res = [0 for i in range(len(lst))]
        for i in range(len(lst)):
            pos = 0
            cnt = lst.count(lst[i])
            for j in range(len(lst)):
                if lst[i] > lst[j]:
                    pos += 1
            for j in range(cnt):
                res[pos + j] = lst[i]
        return res




if __name__ == '__main__':
    for i in range(1000):
        inpt = [random.randint(0, 100) for i in range(1000)]
        method = Series()
        merg = method.MergeSort(inpt)
        enum = method.EnumSort(inpt)
        quic = [i for i in inpt]
        method.QuickSort(quic)
        if not (merg == enum and merg == quic):
            print('Merge:', merg)
            print('Enum :', enum)
            print('Quick:', quic)

