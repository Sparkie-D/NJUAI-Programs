import src.readInput as readInput
import src.stripWords as stripWords

'''
一个老旧的实现版本，没有什么参考意义，留作纪念
'''

def simpleTransPinyin(pinyin):

    charmap = readInput.loadCorpusChar()
    pinyinmap = stripWords.record_pinyin_word(charmap)
    spells = []
    stripWords.stripPinyin(spells, pinyin)
    output = ''

    for spell in spells:
        highest = ''
        record = 0
        wordslist = pinyinmap[spell]
        print(wordslist)

        for word in wordslist:
            if charmap[word] > record:
                record = charmap[word]
                highest = word
                print(highest)
        output += highest

    return output


if __name__ == '__main__':
    print(simpleTransPinyin("weizhonghuazhijueqierdushu"))
