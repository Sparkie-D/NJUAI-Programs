import xlrd
import sys
import json
import math

def loadCorpusChar():
    '''
    @return: 返回字和对应概率的键值对
    '''
    workbook = xlrd.open_workbook(r'CorpusCharacterlist.xls')
    sheetname = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheetname)
    nrows = sheet.nrows
    ncols = sheet.ncols
    # print(nrows, ncols)
    test = sheet.cell(0, 1).value
    charmap = {}
    for row in range(nrows):
        charmap[sheet.cell(row, 0).value] = int(sheet.cell(row, 1).value)
    # print(charmap)
    total = sum(charmap.values())
    for row in range(nrows):
        charmap[sheet.cell(row, 0).value] = math.log(charmap[sheet.cell(row, 0).value] / total)
    return charmap

def loadCorpusWord():
    '''
    @return: 词和概率的键值对
    '''
    workbook = xlrd.open_workbook(r'CorpusWordlist.xls')
    sheetname = workbook.sheet_names()[0]
    sheet = workbook.sheet_by_name(sheetname)
    nrows = sheet.nrows
    ncols = sheet.ncols
    # print(nrows, ncols)
    test = sheet.cell(0, 1).value
    wordmap = {}
    for row in range(nrows):
        wordmap[sheet.cell(row, 0).value] = int(sheet.cell(row, 1).value)
    # print(wordmap)
    total = sum(wordmap.values())
    for row in range(nrows):
        wordmap[sheet.cell(row, 0).value] = math.log(wordmap[sheet.cell(row, 0).value] / total)
    return wordmap

def wordMatchCount(charmap, wordmap):
    '''
    对出现在charmap中的字，统计所属词语在wordmap中的出现频率
    其中字典2是衔接字和它的出现概率之间的映射关系，总共是1
    @return: 字典1:{字:字典2}
    '''
    matchmap = {}
    for char in charmap:
        matchmap[char] = {}
        for word in wordmap.keys():
            tmp = [i for i in word]
            if char in tmp and tmp.index(char) < len(tmp) - 1:
                next = tmp[tmp.index(char) + 1]
                if next not in matchmap[char].keys():
                    matchmap[char][next] = wordmap[word]
                else:
                    matchmap[char][next] += wordmap[word]
    for item in matchmap.keys():
        submap = matchmap[item]
        if len(submap) > 0:
            total = sum(submap.values())
            for subitem in submap:
                submap[subitem] = math.log(submap[subitem] / total)
    return matchmap

def MatchWrapper():
    '''
    @return: {word:{word:prob}}
    '''
    return wordMatchCount(loadCorpusChar(), loadCorpusWord())


def PinyinWrapper():
    '''
    存储拼音对应汉字关系表的包装函数
    '''
    wordmap = loadFromFileJson(loadCorpusWord)
    return stripWords.record_pinyin_word(wordmap)

def StoreToFileJson(loadFunc):
    head = '../data/'
    filename = [k for k, v in globals().items() if v is loadFunc][0]
    filepath = head + filename + '.txt'
    js = json.dumps(loadFunc())
    file = open(filepath, 'w')
    file.write(js)
    file.close()


def loadFromFileJson(loadFunc):
    head = '../data/'
    filename = [k for k, v in globals().items() if v is loadFunc][0]
    filepath = head + filename + '.txt'
    file = open(filepath, 'r')
    js = file.read()
    dic = json.loads(js)
    file.close()
    return dic

if __name__ == '__main__':
    StoreToFileJson(loadCorpusChar)
    # StoreToFileJson(loadCorpusWord)
    # StoreToFileJson(MatchWrapper)
    # StoreToFileJson(PinyinWrapper) # 这条不在这里执行，在stripWords.py执行这句
    # 以下为测试是否装成功
    # print(loadFromFileJson(loadCorpusChar))
    # print(loadFromFileJson(loadCorpusWord))
    # print(loadFromFileJson(MatchWrapper))
    # print(loadFromFileJson(PinyinWrapper))

