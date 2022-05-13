import numpy as np
import sympy as sp
import os
from queue import PriorityQueue
import matplotlib.pyplot as plt
import cv2
import pygame
import math

# %% [markdown]
# Declaring map

# %%
#node size ratio (unit/node)

def map():
    global x_size, y_size, map_def, Rat
    Rat=1
    x_size = int(400/Rat) #x and y  as per gloabl coordinate system and not matrix
    y_size = int(250/Rat)
    map_def = np.zeros((y_size,x_size,360))
    map_def[:,:,:] = -2

# %% [markdown]
# Declaring Node 

# %%

class Node:
    def __init__(self, node_id, node_loc, parent_id, c2c , c2g):
        self.parent_id = parent_id #
        self.node_id = node_id #unique node id for each node
        self.node_loc = node_loc ## [x,y,angle]
        self.c2c= round(c2c,0)
        self.c2g=round(c2g,2)
        self.total_cost = round(c2c+c2g,2)
    

# %% [markdown]
# Declaring Obstacle space

# %%
"""Circle"""

#x is column and y is row for the matrix
def circle_obs():
    global x_c, y_c, x_nc, y_nc, map_obs1, allp_map,x_centre,y_centre,radius
    x_c = []
    y_c = []
    x_nc = []
    y_nc = []
    y_centre = int((249 - 65)/Rat)
    x_centre = int((399 - 100)/Rat)
    radius_c = int(55/Rat) #radius of circle including clearance
    radius = int(40/Rat)
    map_obs1 = np.copy(map_def) # a deep copy #change this to map_obs1
    for x in range(int(240/Rat),int(360/Rat)):
        for y in range(int(120/Rat), int(240/Rat)):
            if (np.sqrt((x-x_centre)**2 + (y-y_centre)**2)) < radius_c:
                map_obs1[y, x,:] = -1 #obstacle space updated as -1
                x_c.append(x) #to be used for plotting
                y_c.append(y)
                allp_map.append([x,y]) #to be used for visualisation on pygame
            if (np.sqrt((x-x_centre)**2 + (y-y_centre)**2)) < radius:
                x_nc.append(x) #list of obstacle without clearance
                y_nc.append(y)

# %%
"""Triangles"""

def triang_obs():
    global x_t1, y_t1, x_t1_nc, y_t1_nc, map_obs2, x_t2_nc, y_t2_nc, x_t2, y_t2, allp_map, Rat
    x_t1 = []
    y_t1 = []
    x_t2 = []
    y_t2 = []
    x_t1_nc = []
    y_t1_nc = []
    x_t2_nc = []
    y_t2_nc = []
    map_obs2 = np.copy(map_obs1)
    for x in range(int(170/Rat)):
        for y in range(int(40*Rat),int(240/Rat)):
            if (y - (6/7)*x - ((779/7) + 15)/Rat)<=0 and (y - (-85/69)*x - ((15880/69) -18)/Rat)>=0 and (y - (-16/5)*x - ((2169/5) +25)/Rat)<=0 :
                map_obs2[y,x,:] = -1
                x_t1.append(x)
                y_t1.append(y)
                allp_map.append([x,y])
            if (y - (6/7)*x - ((779/7))/Rat)<=0 and (y - (-85/69)*x - ((15880/69))/Rat)>=0 and (y - (-16/5)*x - ((2169/5) +10)/Rat)<=0 :
                map_obs2[y,x,:] = -1
                x_t1_nc.append(x)
                y_t1_nc.append(y)
                allp_map.append([x,y])
            if (y - (6/7)*x - ((779/7) - 15)/Rat)>=0 and (y - (-85/69)*x - ((15880/69) -18)/Rat)>=0 and (y - (25/79)*x - ((13661/79) +15)/Rat)<=0:
                map_obs2[y,x,:] = -1
                x_t2.append(x)
                y_t2.append(y)
                allp_map.append([x,y])
            if (y - (6/7)*x - ((779/7) )/Rat)>=0 and (y - (-85/69)*x - ((15880/69))/Rat)>=0 and (y - (25/79)*x - ((13661/79))/Rat)<=0:
                map_obs2[y,x,:] = -1
                x_t2_nc.append(x)
                y_t2_nc.append(y)
                allp_map.append([x,y])

# %%
"""Hexagon"""

def hex_obs():
    global map_obs4, x_h, y_h, x_h_nc, y_h_nc, allp_map
    x_h = []
    y_h = []
    x_h_nc = []
    y_h_nc = []
    map_obs4 = np.copy(map_obs2)
    for x in range(int(148/Rat), int(252/Rat)):
        for y in range(int(30/Rat),int(165/Rat)):
            if (y - (5/8)*x + 81/Rat)>=0 and (y + (5/8)*x - 281/Rat)<=0 and (y - (5/8)*x - 31/Rat)<=0 and (y + (5/8)*x - 169/Rat)>=0 and (x - 249/Rat)<=0 and (x - 150/Rat)>=0:
                map_obs4[y,x,:] = -1
                x_h.append(x)
                y_h.append(y)
                allp_map.append([x,y])
            if (y - (4/7)*x + (380/7)/Rat)>=0 and (y + (4/7)*x - 1780/7/Rat)<=0 and (y - (4/7)*x - (180/7)/Rat)<=0 and (y + (4/7)*x - (1220/7)/Rat)>=0 and (x - 234/Rat)<=0 and (x - 165/Rat)>=0:
                map_obs4[y,x,:] = -1
                x_h_nc.append(x)
                y_h_nc.append(y)
                allp_map.append([x,y])

# %%

def c2g(initial, final):
    return np.round(np.linalg.norm(np.array(initial)-np.array(final)),2)

# %%
"""considering a radius of 1.5 for goal space:"""

def goal_space(goal_x, goal_y):
    centre_x = goal_x
    centre_y = goal_y
    goal_list = []
    for i in range(goal_x-int(30/Rat),goal_x+int(30/Rat)):
        for j in range(goal_y-int(30/Rat), goal_y+int(30/Rat)):
            if ((i-centre_x)**2 + (j-centre_y)**2)<=(15/Rat):
                goal_list.append([i,j])
    return goal_list 

# %% [markdown]
# 

# %%

def pop(Closed_list, Open_list, All_list): ### pop from open list and add to closed list and check for goal
    dat = Open_list.get()
    #dat[0] is c2c+c2g, dat[1] is Node_id--> starts from 1, list starts from 0 
    Closed_list.append(All_list[dat[1]-1]) #indexing the node from the all_list  
    
    print('closed list updated')

    ##check for goal
    if All_list[dat[1]-1].node_loc[0:2] in goal_list:
        goal_id=dat[1]-1
        print('this will be done at', All_list[dat[1]-1].node_loc )
        print('the goal cost is', All_list[dat[1]-1].total_cost)
        return "Goal found", goal_id
    return 1, All_list[dat[1]-1]

# %%
def move_zero(node, All_list, Open_list,action):
    ## r= radius of wheel
    ## L= wheel distance
    r = 0.038*1000
    L = 0.354*1000
    ul=action[0]
    ur=action[1]
    x_dot=[]
    t=0
    dt=0.1
    D=0
    Xn,Yn,thetan=node.node_loc
    thetan=3.14*thetan/180 ## degree to radian
    while t<2:
        Xs=Xn
        Ys=Yn
        t=t+dt
        d_xn=(0.5*r * (ul+ur) * math.cos(thetan) * dt)/Rat
        d_yn=(0.5*r * (ul+ur) * math.sin(thetan) * dt)/Rat
        d_theta=((r / L) * (ur-ul) * dt)
        Xn=Xn+d_xn
        Yn=Yn+d_yn
        thetan+= d_theta
        D=D+math.sqrt(math.pow((0.5*r * (ul+ur) * math.cos(thetan) * dt),2)+math.pow((0.5*r * (ul+ur) * math.sin(thetan) * dt),2))
        ## check if path in obstacle
        # print(Xn,Yn,thetan)
        if (np.round(Yn)>=250 or np.round(Yn)<=0 or 0>= np.round(Xn) or np.round(Xn)>=400 or (Xs==Xn and Ys==Yn)):
            print('Boundaries')
            return None
        elif map_obs4[int(np.round(Yn)),int(np.round(Xn)),int(np.round((thetan*180/3.14))%360)]==-1:
            print('Path in obstacle')
            return None
        
        else:
            Xn=int(np.round(Xn))
            Yn=int(np.round(Yn))
            thetan=int(np.round((180*(thetan)/3.14))%360)
            if map_obs4[Yn,Xn,thetan]==-2:
                param(All_list, Open_list, map_obs4, Xn,Yn,thetan,D, node)
                print('node created')
                return None
            else:
                print('node exists, cost check')
                cost_update(node, Xn ,Yn,thetan, All_list,  Closed_list, Open_list, map_obs4, D)  
                return None

def movement(node, All_list, Open_list, rpm1, rpm2):
    ##8 actions

    ##move right with rpm1
    move_zero(node, All_list, Open_list,[0,rpm1])
    print('moving right rpm1')
    ## move left with rpm1
    move_zero(node, All_list, Open_list,[rpm1,0])
    print('moving left rpm1')
    ## move straight with rpm1
    move_zero(node, All_list, Open_list,[rpm1,rpm1])
    print('moving straight rpm1')
    ## move right with rpm2
    move_zero(node, All_list, Open_list,[0,rpm2])
    print('moving right rpm2')
    ## move left with rpm2
    move_zero(node, All_list, Open_list,[rpm2,0])
    print('moving left rpm2')
    ## move straight with rpm2
    move_zero(node, All_list, Open_list,[rpm2,rpm2])
    print('moving straight rpm2')
    ## move right if rpm2>rpm1
    move_zero(node, All_list, Open_list,[rpm1,rpm2])
    print('moving s1 rpm2')
    ## move left if rpm2>rpm1
    move_zero(node, All_list, Open_list,[rpm2,rpm1])
    print('moving s2 rpm2')

    pass

# %%



# %%

def param(All_list, Open_list, map_obs4, Xn, Yn, thetan,D, node):
     id = All_list[-1].node_id + 1
     map_obs4[Yn,Xn,thetan] = id  ## id is updated in map
     c2_g=c2g([Xn,Yn],goal_location[0:2])
     c2c=node.c2c+D
     cost = c2c+c2_g  #cost_dir is a dictionary
     parent = node.node_id
     loc = [Xn,Yn,thetan]
     print('node at id', id, 'cost is', cost, 'loc', loc)
     All_list.append(Node(id, loc, parent, c2c,c2_g)) ## node is created
     tup_new = [cost, id]
     Open_list.put(tup_new) ## list of [cost, id] ###gives the [cost,id] in priority queue open list

# %%

def cost_update(nod, Xn ,Yn, thetan, All_list,  Closed_list, Open_list, map_obs4, D):
     index = int(map_obs4[Yn, Xn, thetan])
     if round((nod.c2c + D),1)< round(All_list[index-1].c2c,1): #since index/node_id is starting from 1
            All_list[index-1].c2c = round((nod.c2c + D),1)
            All_list[index-1].parent_id = nod.node_id
            print('updated cost of node', index, 'is', All_list[index-1].total_cost)
            if Open_list.qsize() > 0:
                for j in Open_list.queue:
                    if j[1] ==index:
                            j[0] = round((nod.c2c + D + All_list[index-1].c2g),1)
                            # i.parent_id = nod.node_id

def backtrack(A_list, x, y,z, goal_id):
    ind = int(map_obs4[All_list[goal_id].node_loc[1], All_list[goal_id].node_loc[0],All_list[goal_id].node_loc[2]])
    x.append(A_list[ind-1].node_loc[0])
    y.append(A_list[ind-1].node_loc[1])
    z.append(A_list[ind-1].node_loc[2])
    id = A_list[ind-1].parent_id
    while(id>0):
        x.append(A_list[id-1].node_loc[0])
        y.append(A_list[id-1].node_loc[1])
        z.append(A_list[id-1].node_loc[2])
        id = A_list[id - 1].parent_id

def calc_linear(x_a,y_a,z_a):
    v_list=[]
    w=[]
    t_list=[]
    t=4
    v_i = 0.2
    w_old = 0
    w_i = 0
    for i in range(1,len(x_a),4):
        dist=(((x_a[i-1]-x_a[i])**2+(y_a[i-1]-y_a[i])**2)**(1/2))/1000
        # d=dist/t
        v=dist/t
        o_x=x_a[i-1]
        n_x=x_a[i]
        o_y=y_a[i-1]
        n_y=y_a[i]
        d = 0
        w_old = 0
        delt = 0
        
        
        if v>0.22:
            v = 0.22
            delt = dist/v
        else:
            delt = t

        w_new = math.radians(z_a[i]-z_a[i-1])/delt


        if w_new>2:
            w_new = 2

        elif w_new<-2:
            w_new = -2

        v_list.append(v)
        w.append(w_new)
        t_list.append(delt)

        v_list.append(0)
        w.append(0)
        t_list.append(1)
        
        
    
    return v_list,w,t_list

        






def display_path(goal_id):
    global x_a, y_a
    # if (map_obs4[All_list[goal_id].node_loc[0], All_list[goal_id].node_loc[1],All_list[goal_id].node_loc[2]]!=-1 and map_obs4[All_list[goal_id].node_loc[0], All_list[goal_id].node_loc[1],All_list[goal_id].node_loc[2]]!=-2 ):
    x_a = []
    y_a = []
    z_a = []
    backtrack(All_list, x_a, y_a, z_a, goal_id)
    lin_vel,ang_vel,t_list=calc_linear(x_a, y_a, z_a)

    # y_a.reverse()
    # x_a.reverse()
    plt.title('Backtracked Path')
    plt.axis([0, 400/Rat, 0, 250/Rat])
    plt.plot(x_a,y_a)
    op_path=[[lin_vel[i],ang_vel[i],t_list[i]] for i in range(len(lin_vel))]
    with open('output.txt', 'w') as file:
        for element in op_path:
            file.write(",".join((str(x) for x in element)) + '\n')

    # path=[[x_a,ang_vel] for i in range(len(lin_vel))]
    # with open('output.txt', 'w') as file:
    #     for element in op_path:
    #         file.write(",\n".join((str(x) for x in element)) + '\n')


    plt.plot(x_nc, y_nc)
    plt.plot(x_t1_nc,y_t1_nc)
    plt.plot(x_t2_nc,y_t2_nc)
    plt.plot(x_h_nc,y_h_nc)
    plt.show()
    # else:
    #     print('cannot be back tracked')
    #     exit(0)

    x_plot = []
    y_plot = []


    #explored path
    for i in All_list:
        
        y_plot.append(i.node_loc[1])
        x_plot.append(i.node_loc[0])
    
    plt.title('Explored Path')
    plt.axis([0, 400/Rat, 0, 250/Rat])
    plt.scatter(x_plot,y_plot)
    plt.plot(x_nc, y_nc)
    plt.plot(x_t1_nc,y_t1_nc)
    plt.plot(x_t2_nc,y_t2_nc)
    plt.plot(x_h_nc,y_h_nc)
    plt.show()


def display():

        # Put animation here

    pygame.init()

    display_width = int(400/Rat)
    display_height = int(250/Rat)
    display_h = 250
    n = 2
    m = n
    s = n/Rat
    gameDisplay = pygame.display.set_mode((n*display_width,n*display_height))
    pygame.display.set_caption('Visited Nodes and Backtrack- Animation')
    black = (0,0,0)
    white = (255,255,255)
    Y = (255, 0,0)
    B = (0,0,255)
    G=(0,255,0)

    clock = pygame.time.Clock()
    done = True
    while done:
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:  
                done = False  
        
        gameDisplay.fill(black)
        pygame.draw.circle(gameDisplay, B, (n*x_centre, n*(display_height-1-y_centre)), n*radius)
        pygame.draw.polygon(gameDisplay, B, [(s*114,s*(display_h-1-209)),(s*35,s*(display_h-1-184)), (s*104,s*(display_h-1-99)), (s*79,s*(display_h-1-179))])
        pygame.draw.polygon(gameDisplay, B, [(s*199,s*(display_h-1-59.6)),(s*234,s*(display_h-1-78.8)), (s*234,s*(display_h-1-119)), (s*200,s*(display_h-1-139)),(s*164,s*(display_h-1-119)),(s*164,s*(display_h-1-78.8))])
        pygame.draw.circle(gameDisplay, G, (n*goal_location[0], n*(display_height-1-goal_location[1])), n*5)
        pygame.draw.circle(gameDisplay, Y, (n*start_location[0], n*(display_height-1-start_location[1])), n*5)
        for i in range(len(All_list)):
                p_id = All_list[i].parent_id
                if(i>0):
                    x_start = All_list[p_id -1].node_loc[0]
                    y1_start = All_list[p_id -1].node_loc[1]
                    y_start = abs(display_height-1-y1_start)
                    # print(x_start, y_start)
                else:
                    x_start = All_list[p_id].node_loc[0]
                    y1_start = All_list[p_id].node_loc[1]
                    y_start = abs(display_height-1-y1_start)
                
                x = All_list[i].node_loc[0]
                y1 = All_list[i].node_loc[1]
                y = abs(display_height-1-y1)
                pygame.draw.line(gameDisplay, white, (n*x_start,n*y_start),(m*x,m*y), 1)
                pygame.display.flip()
                # pygame.time.wait(4)

        # ind = int(map_obs4[All_list[goal_id].node_loc[0], All_list[goal_id].node_loc[1],All_list[goal_id].node_loc[2]])
        for i in range(len(x_a)-1):
                pygame.draw.line(gameDisplay, Y, (n*x_a[i],n*(display_height-y_a[i])),(n*x_a[i+1],n*(display_height-y_a[i+1])), 5)
                pygame.display.flip()
                # pygame.time.wait(4)
        done = False
        
        pygame.time.wait(4000)
        pygame.quit()

# %%

def initial():
    global All_list, Closed_list, goal_location, Open_list, allp_map, goal_list, L,rpm1,rpm2,Rat,start_location
    allp_map = []
    All_list = []
    Closed_list = []
    goal_list = []
    Rat=1
    first_node_id = 1
    start_cords_x = int(input("please enter the starting x coordinate between 10 and 390: \n"))
    start_cords_y = int(input("please enter the starting y coordinate between 10 and 240: \n"))
    start_angle= int(input("please enter the starting angle from 0 to 360: \n"))


    if start_cords_x<10 or start_cords_x>(400-10) or start_cords_y<10 or start_cords_y>(250-10):
        print("Either wrong input or the start node is in obstacle space")
        exit(0)
    else:
        start_location = [int((start_cords_x-1)/Rat), int((start_cords_y-1)/Rat), start_angle]
        
    rpm1=int(input("please enter the first rotations per minute value: \n"))/60
    rpm2=int(input("please enter the second rotations per minute value: \n"))/60
    r = 0.038*1000
    L = 0.354*1000
    goal_cords_x =int(input("please enter the goal x coordinate between 10 and 390: \n"))
    goal_cords_y =int(input("please enter the goal y coordinate between 10 and 240: \n"))
                           
    
    if goal_cords_x<10 or goal_cords_x>(400-10) or goal_cords_y<10 or goal_cords_y>(250-10):
        print("Either wrong input or the goal node is in obstacle space")
        exit(0)
    else:
        goal_location = [int((goal_cords_x-1)/Rat), (int(goal_cords_y-1)/Rat)]


    goal_list = goal_space(int((goal_cords_x-1)/Rat), int((goal_cords_y-1)/Rat))



    ### Initialising map
    map()
    print('map is defined, initialising obstacle space')
    circle_obs()
    triang_obs()
    hex_obs()
    # if map_obs4[int((start_cords_x-1)/Rat), int((start_cords_y-1)/Rat), start_angle]==-1 or map_obs4[int((goal_cords_y-1)/Rat), int((goal_cords_x-1)/Rat), goal_k]==-1:
    #     print("start or goal node in obstacle space")
    #     exit(0)
    # else:
    map_obs4[int((start_cords_y-1)/Rat), int((start_cords_x-1)/Rat), start_angle] = first_node_id
    first_parent_id = 0
    initial_c2g=c2g(start_location[0:2], goal_location[0:2])
    first_cost = 0+initial_c2g
    All_list.append(Node(first_node_id, start_location, first_parent_id,0,initial_c2g))
    tup = [first_cost, first_node_id] # just the cost and node id, access the node using all visited and node id
    Open_list = PriorityQueue()
    Open_list.put(tup) 



def main():
    # %%

    Rat=1
    initial()

    # %%
    goal_list

    # %%

    ##goal_cost = np.inf
    while(1):
        if (Open_list.qsize()>0):
            print('inside while')
            r,node =pop(Closed_list, Open_list, All_list) ## pop works
            print('all list length is', len(All_list))
            if type(r) == str:
                print('done/GOAL FOUND')
                goal_id=node
                display_path(goal_id)
                display()
                break
            else:
                print('popped')
                movement(node, All_list, Open_list, rpm1,rpm2)
        else:
            print('open list empty')
            break