import src.viterbiTrans as viterbiTrans
import src.stripWords as stripWords

if __name__ == '__main__':
    strips = 20
    print(' ' * strips , '_' * 25, ' ' * strips)
    print('*' * strips, '| 基于维特比算法的拼音输入法 |', '*' * strips)
    print(' ' * strips , '-' * 25, ' ' * strips)
    print('请选择模式：')
    print('* 1.划分长拼音（分词功能）')
    print('* 2.智能拼音输入法')
    print('-' * 46)
    mode = eval(input())
    if mode == 1:
        print("*" * strips, '分词模式','*' * strips )
        print('请输入拼音：')
        pinyin = input()
        record = []
        stripWords.dpstripPinyin(record, pinyin)
        record.reverse()
        for item in record:
            print(item, end = ' ')
        print()
    else:
        print("*" * strips, '输入法模式','*' * strips)
        print('请输入拼音：')
        pinyin = input()
        print('请稍后......')
        chances = viterbiTrans.viterbianChancedTrans(pinyin)
        for i in range(len(chances)):
            item = chances[i]
            print(i + 1, ''.join(item[1]), item[0])
