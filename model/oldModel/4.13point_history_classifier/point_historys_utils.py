#用来分隔数据 重新排列数据的程序
Scsvfile='model/point_history_classifier/point_history.csv'
Ocsvdir='model/point_history_classifier/point_historys/'
filename='point_history'
classnum=6

def main():
    print("开始运行")
    spiltCSV(Scsvfile,Ocsvdir)
    #formCSV(Scsvfile,Ocsvdir)
    

def spiltCSV(Scsvfile,Ocsvdir):
    with open(Scsvfile,'r') as f:
        print("打开f文件成功")
        line = f.readline()
        while line:
            num=line[0]
            print(num)
            with open(Ocsvdir+filename+str(num)+'.csv','a',newline="") as of:
                of.write(line)
            line=f.readline()


def formCSV(Scsvfile,Ocsvdir):
    with open (Scsvfile,'w',newline="")as of:
        print("打开of文件成功")
        for i in range(classnum):
            print(i)
            with open(Ocsvdir+filename+str(i)+'.csv','r') as f:
                line=f.readline()
                while line:
                    of.write(line)
                    line=f.readline()


if __name__ == '__main__':
    main()
