import json
import socket
import time
import struct
import numpy as np
import math

HOST = "192.168.1.100"    # The remote host
PORT = 30003        # The same port as used by the server
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((HOST, PORT))
def caijishuju():
    # for j in range(2600):
    #     print('**********************************************************************************')
    #     print(j)
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 初始化一个TCP类型的Socket
        # s.connect((HOST, PORT))  # 与服务端建立连接
    dic = {'MessageSize': 'i', 'Time': 'd', 'q target': '6d', 'qd target': '6d', 'qdd target': '6d',
           'I target': '6d',
           'M target': '6d', 'q actual': '6d', 'qd actual': '6d', 'I actual': '6d', 'I control': '6d',
           'Tool vector actual': '6d', 'TCP speed actual': '6d', 'TCP force': '6d',
           'Tool vector target': '6d',
           'TCP speed target': '6d', 'Digital input bits': 'd', 'Motor temperatures': '6d',
           'Controller Timer': 'd',
           'Test value': 'd', 'Robot Mode': 'd', 'Joint Modes': '6d', 'Safety Mode': 'd', 'empty1': '6d',
           'Tool Accelerometer values': '3d',
           'empty2': '6d', 'Speed scaling': 'd', 'Linear momentum norm': 'd', 'SoftwareOnly': 'd',
           'softwareOnly2': 'd', 'V main': 'd',
           'V robot': 'd', 'I robot': 'd', 'V actual': '6d', 'Digital outputs': 'd', 'Program state': 'd',
           'Elbow position': '3d', 'Elbow velocity': '3d'}

    data = s.recv(1116)  # 等待接收信息  1108代表返回数据字节总数  二进制形式？？？   %后边可以print(names)，看一下最一开始'MessageSize': 'i'里边的值是多少，填到里边，二楼是1116，一楼是1108
    # 按照字典中的格式解析，解析之后将解析的数据再放入字典中
    names = []

    ii = range(len(dic))
    for key, i in zip(dic, ii):
        #key是字符串类型，按顺序为MessageSize，Time，q target。。。。。依此类推
        #i为0，1，2，3，。。。。。。。37
        fmtsize = struct.calcsize(dic[key])
        # dic[key]的值为键值对应的那个value值，字符串类型，就是i,d,6d,6d,6d。。。。。。。
        # fmtsize则为每个value对应的字节数，4，8，48，48，48。。。。。。。4。。
        data1, data = data[0:fmtsize], data[fmtsize:]
        # print(len(data1))
        fmt = "!" + dic[key]  # !表示我们要使用网络字节顺序解析，因为我们的数据是从网络中接收到的，在网络上传送的时候它是网络字节顺序的
        # fmt就是!i，!d ，!6d，!6d，!6d
        names.append(struct.unpack(fmt, data1))
        # print(names)
        dic[key] = dic[key], struct.unpack(fmt, data1)  # ？？？

    q = dic["q actual"]  # Actual joint positions实际关节位置 (q actual里面都有啥？？？)
    # print(q)
    qd = dic["qd actual"]
    # print(qd)
    i_actual = dic["I actual"]
    # print(i_actual)
    i_target = dic["I target"]
    # print(i_target)
    m_target = dic["M target"]
    # print(m_target)
    t = dic["Time"]

    return np.array(q[1])



with open("new.json") as f:
    tra = json.load(f)
    tra=np.squeeze(np.array(tra))
    tra=np.concatenate([np.zeros([1,6]),tra])
    dt=0.5
    total_time=np.linspace(start=0,stop=dt*len(tra),num=len(tra)+1)
    
    strL = b"movej([0,0,0,0,0,0],a=1.2, v=0.25, t=0, r=0.023)\n"
    s.send(strL)

    while 1:
        q=caijishuju()
        if np.linalg.norm(q)<.2:
            break
    print('zero completed')
    start_time=time.time()
    step=0

    start_time=time.time()
    cur_time=0
    dqd=np.zeros(6)
    i=0
    while (time.time()-start_time)<dt*len(tra)-1:
        temporal_difference=(time.time()-start_time)-cur_time
        cur_time=time.time()-start_time
        step=math.floor(cur_time/dt)
        remainder=cur_time%dt
        q=caijishuju()
        qd=tra[step,:]+(tra[step+1,:]-tra[step,:])*remainder
        ddqd=(qd-q)*0.2-dqd*0.2
        dqd=dqd+ddqd*temporal_difference
        # dqd 速度
        # ddqd*temporal_difference   速度差
        # ddqd  加速度
        # qd 目标位置
        # q当前位置
        
    #     if math.floor((time.time()-start_time)/0.5)>step:
            
    #         step=math.floor((time.time()-start_time)/0.5)
    #         print(step)
    #         strL = "movej(%s, a=1.2, v=0.25, t=0, r=0.023)\n" % list(tra[step,:])
    #         strL = strL.encode()
    
    #         s.send(strL)
         
            
         
            
         # strL = "movej(%s, a=1.2, v=0.25, t=0, r=0.023)\n" % list(tra[step,:])



        strL = "speedj(qd=%s,a=1,t=1)\n" % list(dqd)
    
        strL = strL.encode()
       
        s.send(strL)   

s.close()
       

        


