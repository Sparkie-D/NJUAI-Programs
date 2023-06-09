
def load_dataset():
    # 读取训练集
    train_X, train_Y = [], []
    with open('train.txt', 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            text = lines[i].replace('.', '').replace('?', '').strip()
            target = lines[i+1].strip()
            label = int(lines[i+2].strip())
            train_X.append((text, target))
            train_Y.append(label+1)
    # 读取测试集
    test_X = []
    with open('test.txt', 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            text = lines[i].replace('.', '').replace('?', '').strip()
            target = lines[i+1].strip()
            # label = int(lines[i+2].strip())
            test_X.append((text, target))
    return train_X, train_Y, test_X



