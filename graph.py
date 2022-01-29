"""colors
angry:red(#ff0000)
disgust:green(#008000)
fear:purple(#800080)
happy:yellow(#ffff00)
neutral:black(#000000)
sad:blue(#0000ff)
surprise:lime(#00ff00)
"""

import matplotlib.pyplot as plt

def graphmake(filename):
    colors=["#ff0000","#008000","#800080","#ffff00","#000000","#0000ff","#00ff00"]

    name=filename
    f = open("./emomemo/"+name+".txt", 'r')
    emolist = f.readlines()
    f.close()
    #print(emolist)
    emolistALL=[]
    for i in range(len(emolist)):
        emolist_temp=[]
        emos_S=emolist[i].rstrip("\n")#'0.0036976836,1.8513228e-10,0.37736663,7.271934e-05,0.16216876,0.45669422,3.752166e-09'
        emos=emos_S.split(",")
        for j in range(7):
            emolist_temp.append(float(emos[j]))
        emolistALL.append(emolist_temp)
    #print(emolistALL)

    y=emolistALL
    y_ang,y_dis,y_fea,y_hap,y_neu,y_sad,y_sur=[],[],[],[],[],[],[]
    ylist=[y_ang,y_dis,y_fea,y_hap,y_neu,y_sad,y_sur]

    for i in range(7):
        for j in range(len(emolistALL)):
            ylist[i].append(y[j][i])

    x=list(range(len(emolistALL)))# [0.0,0.1,0.2,0.3,0.4,...]
    x_float=[]
    for i in range(len(x)):
        x_float.append(x[i]/10)
    # make a figure
    fig = plt.figure()
    # set ax to the figure
    ax = fig.add_subplot(1,1,1)
    # plot on axes
    """
    if second == 100:
        for i in range(7):
            ax.plot(x_float,ylist[i],"-",c=colors[i],linewidth=1)
    elif second == 1000:
        for i in range(7):
            ax.plot(x,ylist[i],"-",c=colors[i],linewidth=1)
"""
    for i in range(7):
        ax.plot(x_float,ylist[i],"-",c=colors[i],linewidth=1)

    # 汎用要素
    ax.grid(True)
    ax.set_xlabel('time [sec]')
    ax.set_ylabel('exist rate')
    ax.set_title('detected face emotions')
    ax.legend(['angry','disgust','fear','happy','neutral','sad','surprise'])

    # show
    plt.show()

memoname = input("mamoname (without .txt) >> ")
graphmake(memoname)
