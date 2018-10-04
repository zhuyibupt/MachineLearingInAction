from numpy import *
from os import listdir
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

'''
这是KNN分类算法的实现代码，需要四个传入参数：
  1.需要被分类对象：输入向量inX，每个元素代表传入对象不同的属性值
  2.用于训练的样本集：dataSet(是一个二维array)，每一行代表一个对象，每一行和inX同构
  3.标签向量：labels，元素数目和矩阵dataSet的行数相同
  4.最后的参数k表示最终选择多少个临近的对象
'''
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]                    #1
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet   #2
    sqDiffMat = diffMat**2                            #3
    sqDistances = sqDiffMat.sum(axis = 1)             #4
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()          #5
    classCount = {}                                   #6
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]    #7
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1   #8
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)   #9
    return sortedClassCount[0][0]                     #10
'''
上面的代码解释：
    1.numpy的array都有shape属性，是一个元组，表示array的维数，shape[0]表示最外层有几个元素
    2.tile方法就是重构array的方法，这一句就是让inX先重构shape为(dataSetSize, 1)，然后和样本集的每一个相减，得到新的array为diffMat（每个元素还是数组）
    3.对diffMat每个数值平方得到新数组sqDiffMat（元素还是数组）
    4.axis = 1 代表按照 横轴 相加，也就是将一个点的坐标之差的平方加起来，得到新数组sqDistances（元素是inX和样本集每个点的距离的平方）
    5.argsort方法就是将distances按照元素从小到大顺序，把元素对应的index拿出来
    6.初始化一个字典，用来记录每个label已经出现了几次
    7.这里的i按照distances从小到大取，取出的label返回
    8.如果label已存在直接自加1，不存在的返回0之后自加1，字典的get方法，先返回第一个参数对应的value，如果不存在第一个参数的key，就返回第二个参数
    9.根据字典中的元素sort，但是标准为字典元素的第二个部分，也就是value，也就是label出现的次数，从大到小排序
    10.返回出现次数最多的，也就是第一个元素的key
总结：
    先计算 输入向量inX 到 训练样本集dataSet 每一个元素(每一行)的距离
    然后找出最近的 K 个元素，记录它们的label
    最后取出label次数最多的那个label作为inX的label返回出来
'''


'''
    由于匹配对象的信息都放在file中，四列分别为飞行里程数、玩视频游戏所耗时间百分比、每周消费的冰淇淋公升数 和 分类标签，为了使用KNN分类函数，需要得到 训练样本集 和 标签向量，所以需要读入文件，并将前三列的值作为训练样本集，最后一列作为标签向量
    传入参数为文件的path
'''
def file2matrix(filename):
    love_dictionary={'largeDoses':3, 'smallDoses':2, 'didntLike':1}   #1
    fr = open(filename)                         #2
    arrayOLines = fr.readlines()                #3
    numberOfLines = len(arrayOLines)            #4
    returnMat = zeros((numberOfLines,3))        #5
    classLabelVector = []                       #6   
    index = 0
    for line in arrayOLines:                    #7
        line = line.strip()                     #8
        listFromLine = line.split('\t')         #9
        returnMat[index,:] = listFromLine[0:3]  #10
        if(listFromLine[-1].isdigit()):         #11
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
'''
代码解释：
    1.原文件中的标签是字符串，为了存储更小以及方便计算，这里给出一个标签映射为int数字的字典
    2.打开文件，open(name[, mode[, buffering]])
          i. name : 一个包含了你要访问的文件名称的字符串值。
          ii. mode : mode 决定了打开文件的模式：只读，写入，追加等。所有可取值见如下的完全列表。这个参数是非强制的，默认文件访问模式为只读(r)。
          iii. buffering : 如果 buffering 的值被设为 0，就不会有寄存。如果 buffering 的值取 1，访问文件时会寄存行。如果将 buffering 的值设为大于 1 的整数，表明这就是寄存区的缓冲大小。如果取负值，寄存区的缓冲大小则为系统默认。
    3.返回全部行，如果传入参数n：.readlines([n])，返回前n行的列表, n 未指定则返回全部行
    4.记录文本全部的行数，也即已知的所有匹配对象的人数
    5.zeros(shape[, dtype = float] [, order = 'C'])：创建给定类型的矩阵，并初始化为0，这里就是创建一个 文件行数、3列的矩阵
          i. shape：可以是int类型数据，或者是int类型的序列。表示新的数组的大小，比如zeros(3)，zeros((2,3))等价于zeros([2,3])
          ii. dtype：数组数据类型，默认为float
          iii. order：在内存中的排列方式，有C语言和Fortran语言两种排列方式
    6.初始化 类标签向量(是一个数组)
    7.访问文件的每一行
    8.去除每一行的首尾空格，如果传入参数n：.strip(n)，表示去除每一行的首尾特定字符n
    9.返回字符串的列表，方法str.split(str="" [, num=string.count(str)])
          i. 根据指定分隔符str对字符串进行切片
          ii. num表示最终形成 num+1 个子串，默认是指定分隔符str的数量
    10.文件中每一行都会被切成4个字符串的list，分别表示飞行里程数、视频游戏时间、冰淇淋数、类型，每次循环，都将文件中一行[0:3]，也就是前三个子串赋值给returnMat，returnMat里面是每个人的三个具体数值
    11.这个if else，表示判断每一行最后那个标签是不是数字，如果是数字，就将这个数字添加到类标签向量中，如果不是数字，就将这个标签在love_dictionary字典中对应的数字提取出来并添加，string.isdigit()，检测字符串是否只由数字组成，dict.get(key)，从字典中get出key对应的value
'''


'''
    由于最后要使用KNN的分类算法，但是我们的代码其实是计算坐标点的距离，所以如果不同属性值的量级差距过大，就会有问题，所以这里写一个归一化样本集的方法
'''
def autoNorm(dataSet):
    minValues = dataSet.min(0)  #1
    maxValues = dataSet.max(0)
    diffs = maxValues - minValues
    normDataSet = zeros(shape(dataSet))  #2
    normDataSet = (dataSet - tile(minValues, (dataSet.shape[0], 1))) / tile(diffs, (dataSet.shape[0], 1))  
    return normDataSet, diffs, minValues, maxValues
'''
代码解释：
    1.array.min或者max，但是传入参数代表axis，0代表纵轴，也就是找出每列的最大最小值
    2.先初始化一个和原训练样本集dataSet相同shape的归一化样本集，然后：
        i. 根据指定分隔符str对字符串进行切片
        ii. num表示最终形成 num+1 个子串，默认是指定分隔符str的数量
'''


'''
    这是测试KNN分类方法的准确率的方法，传入参数testRate表示样本集中取测试集的比例，k就是最终比对多少个临近邻居
'''
def datingClassTest(testRate, k):
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')    #1
    normMat, ranges, minV, maxV = autoNorm(datingDataMat)             #2
    m = normMat.shape[0]                      #3
    testNum = int(m*testRate)                 #4
    errorCount = 0.0 
    for i in range(testNum):
        res = classify0(normMat[i, :], normMat[testNum:, :], datingLabels[testNum:], k)  #5
        print("the result of my classifier is %d, the real category is %d" % (res, datingLabels[i]))
        if (res != datingLabels[i]): 
            errorCount += 1.0
    print("if the test rate is: %.2f and the k-value is: %d, the total correct rate is: %.4f" % (testRate, k, 1-errorCount/testNum))
'''
代码解释：
    1.获得训练样本集和标签向量
    2.获得归一化样本集等
    3.获得文本行数，也就是样本数
    4.获得测试集大小，testRate是测试集比例，也就是样本集中准备选取多少比例进行测试，m*testRate也就是测试集有多大
    5.得到分类类别结果：
      a. normMat[i, :] ：在第i次循环中，取出归一化样本集的第i行数据
      b. normMat[testNum:, :] ：取出从testNum开始到最后的所有样本数据作为输入的样本数据
      c. datingLabels[testNum:] ：取出从testNum开始到最后的所有标签数据作为输入的标签数据
'''

'''
这是提供给海伦的程序，可以通过输入一个人的三种属性，判断这个人属于哪种类型
'''
def personClassifier():
    resultList = ['not at all', 'in small doses', 'in large doses']
    flightMiles = float(input("frequent flier miles earned per year?"))            #1
    videoGameTime = float(input("percentage of time spent playing vedio game?"))
    iceCream = float(input("liters of ice cream consumed per week?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')                 #2
    normMat, ranges, minV, maxV = autoNorm(datingDataMat)                          #3
    inArr = array([flightMiles, videoGameTime, iceCream])                          #4
    res = classify0((inArr - minV) / ranges, datingDataMat, datingLabels, 3)
    print("you will probably like this person " + resultList[res-1])
'''
代码解释：
    1.手动输入参数
    2.获得训练样本集和标签向量
    3.获得归一化样本集等
    4.获得输入向量
'''
    
    
'''
生成一个shape为(1, 1024)的向量，然后将数字对应的数据集导入向量中并返回
'''
def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = line[j]
    return returnVect


'''
    测试手写识别代码算法的准确率，传入参数k表示选取临近邻居的个数
    在写入这些代码之前，我们必须确保将from os import listdir写入文件的起始部分，这段代码的主要功能是从os模块中导入函数listdir，它可以列出给定目录的文件名。
'''
def handWritingClass(k):
    hwLabels = []
    trainingFileList = listdir("digits/trainingDigits") #1
    m = len(trainingFileList)  #2
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileName = trainingFileList[i]  #3
        realNum = int((fileName.split('.')[0]).split('_')[0])  #4
        hwLabels.append(realNum)  #5
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileName)  #6
    testFileList = listdir("digits/testDigits")
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileName = testFileList[i]
        realNum = int((fileName.split('.')[0]).split('_')[0])
        testVect = img2vector('digits/testDigits/%s' % fileName)
        res = classify0(testVect, trainingMat, hwLabels, k)  #7
        print("the result is %d meanwhile the real number is %d" % (res, realNum))
        if(res != realNum):
            errorCount += 1
    print("While k-value is %d, we test %d numbers and recognise %d numbers correctly." % (k, mTest, mTest-errorCount))
    print("The correct rate is %f" % float(1-errorCount/mTest))
'''
代码解释：
    1.获得训练集的文档名称list
    2.获得训练集的行数
    3.获得文件名
    4.通过文件名获得真实数字
    5.将真实数字放入 标签向量 中
    6.将图片转为向量，存入 训练样本集 中
    7.通过KNN，将转好的向量变为数字结果res
'''