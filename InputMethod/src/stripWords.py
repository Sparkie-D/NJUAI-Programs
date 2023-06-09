import pypinyin
import json
from src.readInput import loadCorpusWord

# 声母
initials = ['b',  'p', 'm', 'f', 'd', 't', 'n' , 'l',
            'g',  'k', 'h', 'j', 'q', 'x', 'zh', 'ch',
            'sh', 'r', 'z', 'c', 's', 'y', 'w']
# 韵母
finals  = ['a' ,  'o'  , 'e'  , 'i' , 'u' , 'v' ,
           'ai',  'ei' , 'ui' , 'ao', 'ou', 'iu',
           'ue',  'an' , 'en', 'in', 'un', 'uai','ie',
           'vn',  'ang', 'eng', 'ing', 'ong', 'iang',
           'iong', 'ua', 'ia' , 'uan', 'uang', 'ian',
           'uo', 'iao']
# 整体认读音节
componds = ['yuan', 'er']

def pinyin(charac):
    '''
    把输入字转化为拼音形式，返回字符串形式的拼音
    '''
    res  = ''
    for i in pypinyin.pinyin(charac, style = pypinyin.NORMAL):
        res += ''.join(i)
    return res

def wordPinyin(word):
    '''
    把输入词转化为拼音形式，返回列表
    '''
    pinylist = []
    for char in word:
        # print(char)
        pinylist.append(pinyin(char))
    return pinylist


def oldleagal(pinyin):
    if len(pinyin) == 0 or 1 <= len(pinyin) <= 6 and (
            pinyin in componds or pinyin in finals or
            (pinyin[0] in initials and pinyin[1:] in finals) or
            (len(pinyin) > 2 and pinyin[:2] in initials and pinyin[2:] in finals)):
        return True
    else:
        return False


def leagal(pinyin):
    pinymap = loadFromFileJson(PinyinWrapper)
    # print(pinymap)
    if pinyin in pinymap.keys():
        return True
    else:
        return False


def record_pinyin_word(wordmap) -> map:
    '''
    把词频表中出现的字按读音分类，返回如下字典：
    pinyinmap[pinyin] = [word1, word2, ...]
    '''
    pinyinmap = {}
    for item in wordmap.keys():
        tmp = pinyin(item)
        if not oldleagal(tmp):
            continue
        if tmp in pinyinmap.keys():
            pinyinmap[tmp].append(item)
        else:
            pinyinmap[tmp] = [item]
    return pinyinmap


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


def PinyinWrapper():
    '''
    存储拼音对应汉字关系表的包装函数
    '''
    wordmap = loadFromFileJson(loadCorpusWord)
    return record_pinyin_word(wordmap)


def stripPinyin(record, str) ->list:
    '''
    将读入的长拼音序列转化为单个字的拼音列表，存储在record里面
    '''
    # print(str)
    if leagal(str):
        record.append(str)
        return
    elif len(str) <= 1:
        return
    else:
        for i in range(len(str), -1, -1):
            # print(str[:i], leagal((str[:i])))
            if leagal(str[:i]):
                record.append(str[:i])
                stripPinyin(record, str[i:])
                break

def dpstripPinyin(record, str) -> list:
    '''
    动态规划实现分词功能改进
    @param record:结果存储的地方
    @param str:传入的长拼音
    '''
    if len(str) == 0 :
        return True
    for i in range(len(str), -1, -1):
        # print(str[:i], leagal((str[:i])))
        if leagal(str[:i]) and dpstripPinyin(record, str[i:]):
            record.append(str[:i])
            # print(record)
            return True
    else:
        return False


if __name__ == '__main__':
    StoreToFileJson(PinyinWrapper)

