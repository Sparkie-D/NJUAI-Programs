import src.readInput as readInput
import src.stripWords as stripWords
from copy import deepcopy
from src.readInput import loadCorpusChar
from src.readInput import MatchWrapper


def countProb(prevlist, word):
    '''
    @param prevlist:前一个字的序列
    @param word: 当前的字
    @return:
    返回一个概率序列，下标严格和输入字序列对应
    若无对应关系，则相应概率设置为两个字同时出现的概率
    '''
    charmap = readInput.loadFromFileJson(loadCorpusChar)
    matchmap = readInput.loadFromFileJson(MatchWrapper)

    problist = []

    if len(prevlist) == 0:
        return charmap[word]
    else:
        for char in prevlist:
            if word in matchmap[char].keys():
                problist.append(matchmap[char][word])
            else:
                problist.append(charmap[char] + charmap[word])
        return problist


def makechart(pinyin):
    charmap = readInput.loadCorpusChar()
    pinyinmap = stripWords.record_pinyin_word(charmap)
    spells = []
    stripWords.dpstripPinyin(spells, pinyin)
    spells.reverse()
    wordChart = []

    for spell in spells:
        wordslist = pinyinmap[spell]
        wordChart.append(wordslist)

    return wordChart


def changeform(wordChart):
    '''
    修改数据存放方式，将列表转化为字典，加入路径长度，路径元素的概念
    第i个拼音对应的为字典键为i的value
    '''
    charmap = readInput.loadCorpusChar()
    newchart = {}
    for i in range(len(wordChart)):
        newchart[i] = {}
        newline = newchart[i]
        wordlist = wordChart[i]

        for j in range(len(wordlist)):
            newline[wordChart[i][j]] = [charmap[wordChart[i][j]], []]
            if i == 0:
                newline[wordChart[i][j]][1].append(wordChart[i][j])

    return newchart


def showDict(dic):
    for i in dic:
        print(dic[i])


def viterbi(wordChart):
    '''
    @param wordChart:所有字可能的情况表，每一个下标对应一列读音相同的字
    @return: 和第一个字种数大小相同的列表，每个元素为一条路径，路径由下标表示，因此ret[i][j]为第i个字的第j种选择
    '''
    '''
    思路：将每个元素改成键值对的形式，键仍然为当前的字，值为一个列表，形式为[路径长度，路径元素列表]
    '''

    chart = changeform(wordChart)  # 初始化篱笆
    showDict(chart)
    for i in range(1, len(chart)):
        line = chart[i].keys()
        prev = chart[i - 1].keys()

        for word in line:
            # print(word)
            probs = countProb(prev, word)  # 仅仅根据字之间关系进行计算，没有包含路径
            # 加入路径影响
            for k in range(len(probs)):
                probs[k] += chart[i - 1][wordChart[i - 1][k]][0]
            # print(probs)

            idx = probs.index(max(probs))
            prevword = wordChart[i - 1][idx]
            # print(idx, prevword, chart[i][word])
            chart[i][word][0] = chart[i][word][0] + max(probs)  # 更新路径长
            # 更新路径元素
            for item in chart[i - 1][prevword][1]:
                chart[i][word][1].append(item)
            chart[i][word][1].append(word)

        print(chart[i])

    return chart


def viterbianTrans(pinyin):
    '''
    利用维特比算法进行拼音识别
    @param pinyin: 输入的长拼音
    @return :识别出概率最高的汉字序列
    '''
    wordchart = makechart(pinyin)
    reschart = viterbi(wordchart)
    highest = -1000
    idx = -1
    resdict = reschart[len(reschart) - 1]
    for i in resdict:
        if resdict[i][0] > highest:
            idx = i
            highest = resdict[i][0]

    return ''.join(resdict[idx][1])


def findwords(chancelist):
    reslist = []
    for item in chancelist:
        reslist.append(item[1][-1]) # 路径列表的最后一个元素
    return reslist


def mysort(mylist):
    for i in range(len(mylist)):
        for j in range(i + 1, len(mylist)):
            if mylist[j][0] > mylist[i][0]:
                tmp = mylist[i]
                mylist[i] = mylist[j]
                mylist[j] = tmp


def viterbiChances(wordChart, candidates = 5):
    '''
    改进：提供多个选择供使用者自行挑选
    @param wordChart:拼音对应字组成的表格
    @param candidates: 候选项个数
    @return: 一列结果
    '''
    chart = changeform(wordChart)
    # showDict(chart)
    chances = []
    for i in range(candidates): # chances
        chances.append([chart[0][wordChart[0][i]][0], [wordChart[0][i]]])

    for i in range(1, len(chart)):
        # print(chances)
        line = chart[i].keys()
        prev = findwords(chances)
        tmplist = [] # 储存当前循环内生成的路径列表，元素仍然为前概率后列表

        for word in line:
            # print(word)
            tmpchances = deepcopy(chances)
            probs = countProb(prev, word)  # 仅仅根据字之间关系进行计算，没有包含路径
            # 加入路径影响
            for k in range(len(probs)):
                probs[k] += tmpchances[k][0]
            # print(probs)
            for k in range(len(probs)):
                tmplist.append(tmpchances[k])
                idx = tmplist.index(tmpchances[k])
                tmplist[idx][0] = probs[k]
                tmplist[idx][1].append(word)

        mysort(tmplist)
        chances = [tmplist[i] for i in range(candidates)]
        # print(chances)

    return chances

def viterbianChancedTrans(pinyin):
    wordchart = makechart(pinyin)
    chances = viterbiChances(wordchart)
    # print(chances)
    return chances

if __name__ == '__main__':
    # pinyin = 'nanjingdaxueyingyuzhuanyedexuesheng'
    # pinyin = 'xinxingguanzhuangfeiyan'
    # pinyin = 'zhongguorenminjiefangjunhuojianjunbuduifasheleyimeifeiwangmeiguoshouduhuashengdundehedan' #中国人民解放军火箭军部队发射了一枚飞往美国首都华盛顿的核弹
    # pinyin = 'nanjingdaxuexianlinxiaoqujianglaiyidingchengweishijieyiliudaxuedeerliuxiaoqu'
    # pinyin = 'nanjingdaxueyidingchengweiquanqiuzuiniubidequanshijiediyigaoshuipingdaxue'
    # pinyin = 'womenyiqiqukandahai'
    pinyin = 'kuailedabenying'
    # pinyin = input()
    # str = viterbianTrans(pinyin)
    # pinyin = 'woaini'
    chances = viterbianChancedTrans(pinyin)
    for i in range(len(chances)):
        item = chances[i]
        print(i + 1, ''.join(item[1]), item[0])
    # str = viterbianTrans(pinyin)
