### PROBLEM II #####################################################################################################
import datetime

class File:
    def __init__(self, name):
        assert isinstance(name, str)
        self.name = name
        self.date = datetime.datetime.now()

    def __repr__(self):
        return f"Filename : {self.name}\nContent : {self.content}\nGenerated at {self.date:%Y-%m-%d %H:%M:%S}"


class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

    @property
    def data(self):
        return self._data

    @property
    def next(self):
        return self._next

    @data.setter
    def data(self, new_data):
        self._data = new_data

    @next.setter
    def next(self, new_next):
        self._next = new_next
        
class UnorderedList:
    def __init__(self):
        self.head = None
        self.metric = None
            
    def append(self, item):
        current = self.head
        temp = Node(item)
        if current == None:
            self.head = temp
        else:
            while current.next != None:
                current = current.next

            current.next = temp
    
    def merge_sort_helper(self, new_head):
        ### CODE HERE ###
        if new_head.next==None: #list의 길이가 1일때는 정렬할 필요 없이 new_head를 반환하면 된다.
            return new_head
        
        left_half=new_head
        mid_point = self.get_middle(new_head)
        right_half = mid_point.next #코드를 단순화하기 위해 get_middle은 짝수길이의 list의 경우 중간 노드들 중 앞의 것을 반환하게 했다.
        mid_point.next = None #new_head부터 mid_point까지가 왼쪽 절반 리스트, right_head(=mid_point.next)부터 끝까지가 오른쪽 절반이다.
        
        left_head = self.merge_sort_helper(left_half) #쪼갠 두 리스트에 대해 merge sort를 수행해준다.
        right_head = self.merge_sort_helper(right_half)
        
        return self.merge_two_list(left_head,right_head) #sorting된 두 리스트를 따로 구현해준 merge_two_list로 합쳐준다.
        #################
        
    def merge_two_list(self,left_head,right_head):
        temp_head=Node(None) # 두 리스트를 합치기 위해 가상의 temp_head를 만들어주었다.
        current=temp_head #이 노드를 current로 설정해주고 current.next를 계속 붙여주는 방식으로 코드를 구현할 것이다.
        
        while left_head!=None and right_head!=None: #합치는 두 리스트가 끝까지 도달하지 않았을 때
            if self.metric=='date': #metric이 date인 경우 date로 비교해준다.
                if left_head.data.date<right_head.data.date:
                    current.next=left_head
                    left_head=left_head.next
                else:
                    current.next=right_head
                    right_head=right_head.next
            else: #metric이 name인 경우 name으로 비교해준다.
                if left_head.data.name<right_head.data.name:
                    current.next=left_head
                    left_head=left_head.next
                else:
                    current.next=right_head
                    right_head=right_head.next
                
                
            current=current.next
            
        if left_head!=None: #while문이 모두 돌고도 left_head나 right_head가 남아있다면 이를 뒤에 그대로 추가해준다.
            current.next=left_head
        elif right_head!=None:
            current.next=right_head
            
        return temp_head.next
    
    
    def get_middle(self, new_head): #중간지점의 노드를 반환하는 함수이다.
        current=new_head
        cnt=0
        while current!=None: #cnt를 통해 linked list의 길이를 구해준다.
            current=current.next
            cnt+=1
        current2=new_head
        for _ in range((cnt-1)//2): #편의상 짝수 길이의 리스트에서 get_middle은 중간의 두 노드 중 앞의 노드를 반환하도록 하였다.
            current2=current2.next
        return current2
    
    def merge_sort(self, metric):
        assert isinstance(metric, str)
        self.metric = metric
        head = self.merge_sort_helper(self.head)
        self.head = head

    def __repr__(self):
        result = "["
        current = self.head
        if current != None:
            result += current.data.name
            current = current.next
            while current != None:
                result += ", " + current.data.name
                current = current.next
        result += "]"
        return result
####################################################################################################################


### PROBLEM III ####################################################################################################
#problem 3-1
class HashTable_Chain:
    def __init__(self, size):
        self.size = size
        self.slots = [[] for _ in range(size)]
        self.data = [[] for _ in range(size)]
        self.num_collision = 0
        self.num_element = 0
          
    def hash_func(self, key): #hash_function은 key를 self.size로 나눈 값으로 설정해준다.
        ### CODE HERE ###
        return key % self.size
        #################
    
    def _resize(self):
        ### CODE HERE ###
        key_list=self.keys() #기존의 key,data를 저장해둔다.
        data_list=self.values()
        
        self.size=self.size*2 #self.size를 두배로 늘려준 이후 slots와 data를 새로 만들어준다.
        self.slots = [[] for _ in range(self.size)]
        self.data = [[] for _ in range(self.size)]
        
        for i in range(len(key_list)): #기존의 key,data를 다시 넣어준다.
            self.put(key_list[i],data_list[i])
        
        #################
        
    def put(self, key, data):
        ### CODE HERE ###
        if self.num_element==self.size: #num_element가 self.size와 동일한 경우 resize를 실행해준다.
            self._resize()
        hash_value=self.hash_func(key)
        if len(self.slots[hash_value])!=0: #내가 찾은 슬롯이 비어있지 않은 경우
            self.num_collision+=1 #collision값을 하나 늘려준다.
        found=False
        location=None
        for i in range(len(self.slots[hash_value])):
            if self.slots[hash_value][i]==key:
                found=True
                location=i
        if found: #해당 slot내부의 chain에 이미 key가 있을 경우 data만 바꿔준다.
            self.data[hash_value][location]=data
        else: #key가 없을 경우 key, data를 모두 append해준다.
            self.slots[hash_value].append(key)
            self.data[hash_value].append(data)
        self.num_element+=1 #num_element를 늘려준다.
        #################
        
    def get(self, key):
        ### CODE HERE ###
        hash_value=self.hash_func(key)
        found=False
        location=None
        for i in range(len(self.slots[hash_value])):#hash_value를 통해 얻은 chain에서 key를 검색해준다.
            if self.slots[hash_value][i]==key:
                found=True
                location=i
        if found:
            return self.data[hash_value][location]
        else:
            print('There is no corresponding key')
        #################
            
    def remove(self, key):
        ### CODE HERE ###
        hash_value=self.hash_func(key)
        found=False
        location=None
        for i in range(len(self.slots[hash_value])): #hash_value를 통해 접근한 slot에서 for문을 통해 key를 찾아준다.
            if self.slots[hash_value][i]==key: #key를 찾은 경우
                found=True
                location=i
        if found: #key,data쌍을 삭제시켜주는 경우 리스트 슬라이싱을 이용했다.
            self.slots[hash_value]=self.slots[hash_value][:location]+self.slots[hash_value][location+1:]
            self.data[hash_value]=self.data[hash_value][:location]+self.data[hash_value][location+1:]
            self.num_element-=1
        else: #key를 찾지 못해 에러메시지 반환
            print('There is no corresponding key')
        #################
    
    
    def __getitem__(self, key):
        ### CODE HERE ###
        return self.get(key)
        #################
    
    def __setitem__(self, key, data):
        ### CODE HERE ###
        self.put(key,data)

        #################
        
    def __delitem__(self, key):
        ### CODE HERE ###
        self.remove(key)
        #################
        
    def __len__(self):
        ### CODE HERE ###
        return self.num_element #len은 Table_size가 아니라 key,data쌍 개수를 반환한다
        #################
    
    def __contains__(self, key): #모든 slot을 확인해 key가 있으면 True를, 아니면 False를 반환한다.
        ### CODE HERE ###
        found=False
        for i in range(len(self.slots)):
            for j in range(len(self.slots[i])):
                if self.slots[i][j]==key:
                    found=True
                    
        return found
        #################
    
    def keys(self):
        ### CODE HERE ###
        result=[]
        for i in range(len(self.slots)):
            for j in range(len(self.slots[i])):
                result.append(self.slots[i][j])
        return result
        #################
    def values(self):
        ### CODE HERE ###
        result=[]
        for i in range(len(self.slots)):
            for j in range(len(self.slots[i])):
                result.append(self.data[i][j])
        return result
        #################
    

class HashTable_DoubleHash:
    def __init__(self, size):
        self.size = size
        self.slots = [None] * self.size
        self.data = [None] * self.size
        self.num_collision = 0
        self.num_element = 0
    
    def hash_function1(self, key):
        ### CODE HERE ###
        return key % self.size
        #################
    
    def hash_function2(self, key):
        ### CODE HERE ###
        return (key+(self.size//2))%self.size #서로소 조건을 위해 hash_function2는 반드시 self.size보다 작은 값을 반환하도록 했다.

        #################
    def _resize(self):
        ### CODE HERE ###
        #resize에는 조건문이 필요 없고 put하면서 num_element가 많으면 resize를 실행하도록 한다.
        key_list=self.keys()
        data_list=self.values()
        
        self.size=self.size*2
        self.slots = [None] * self.size
        self.data = [None] * self.size
        
        for i in range(len(key_list)):
            self.put(key_list[i],data_list[i])
        #################
                
    def put(self, key, data):
        ### CODE HERE ###

        if self.num_element>=0.75*self.size: #num_element가 self.size의 0.75배보다 크면 resize를 실행해준다.
            self._resize()
            
        hash_value=self.hash_function1(key)
        if self.slots[hash_value]==None or self.slots[hash_value]==key or self.slots[hash_value]=='delete':
            self.slots[hash_value]=key
            self.data[hash_value]=data
        else:
            cnt=1 #위의 탈출조건을 만족하지 않는 경우 hash_value를 계속 바꿔가면서 while문을 수행해준다.
            while self.slots[hash_value]!=None and self.slots[hash_value]!=key and self.slots[hash_value]!='delete':
                hash_value=(self.hash_function1(key)+cnt*self.hash_function2(key))%self.size
                self.num_collision+=1 #한번 충돌이 일어날때마다 num_collision을 하나씩 증가시켜준다.
                cnt+=1
            #key를 넣을 수 있는 slot을 찾을 때까지 계속 hash_value를 바꾼다. 해당 slot을 찾을 경우 while문이 종료된다.
            self.slots[hash_value]=key
            self.data[hash_value]=data
        self.num_element+=1
        
        #################
            
                    
    def get(self, key):
        ### CODE HERE ###
        hash_value=self.hash_function1(key)
        if self.slots[hash_value]==key:
            return self.data[hash_value]
        else:
            cnt=1
            found=True
            while self.slots[hash_value]!=key:
                if cnt==self.size or self.slots[hash_value]==None: #self.size만큼 반복했거나 while문 동작중 None인 slot을 발견하면
                    found=False #못 찾은 것으로 판단하고 while문을 탈출한다. 이후 error메시지를 출력한다.
                    break
                else: #위의 조건이 아닌 경우 계속 hash_value를 바꿔가면서 while문을 동작한다.
                    hash_value=(self.hash_function1(key)+cnt*self.hash_function2(key))%self.size
                    cnt+=1
                    
            if found: #해당 key를 찾았으면 key에 대응하는 data를 반환한다.
                return self.data[hash_value]
            else:
                print('There is no corresponding key')
                
        #################
    
    def remove(self, key):
        ### CODE HERE ###
        hash_value=self.hash_function1(key)
        if self.slots[hash_value]==key: #바로 key를 찾을 경우 delete로 표기해주고 삭제해준다.
            self.slots[hash_value]='delete'
            self.data[hash_value]='delete'
            self.num_element-=1
        else:
            cnt=1
            found=True
            while self.slots[hash_value]!=key: #찾지 못할 경우 key를 찾을 때까지 while문을 수행해준다.
                if cnt==self.size or self.slots[hash_value]==None:#None인 slot을 만나거나 self.size만큼 반복했을 경우
                    found=False #못찾은 것으로 판단하고 while문을 탈출한다. 이후 error 메시지를 출력한다.
                    break
                else:
                    hash_value=(self.hash_function1(key)+cnt*self.hash_function2(key))%self.size
                    cnt+=1
                    
            if found: #찾았다면 key,data쌍을 delete로 바꿔주고 num_element를 하나 줄여준다.
                self.num_element-=1
                self.slots[hash_value]='delete'
                self.data[hash_value]='delete'
                
            else:
                print('There is no corresponding key')
                
        #################
    

    def __getitem__(self, key): 
        ### CODE HERE ###
        return self.get(key)
        #################
    
    def __setitem__(self, key, data): 
        ### CODE HERE ###
        self.put(key,data)
        #################
    
    def __delitem__(self, key):
        ### CODE HERE ###
        self.remove(key)
        #################
        
    def __len__(self):
        ### CODE HERE ###
        return self.num_element
        #################
    
    def __contains__(self, key):
        ### CODE HERE ###
        found=False
        for i in range(len(self.slots)):
            if self.slots[i]==key:
                found=True
                break
                    
        return found
        #################
    
    def keys(self):
        ### CODE HERE ###
        result=[]
        for i in range(len(self.slots)):
            if self.slots[i]!=None and self.slots[i]!='delete':
                result.append(self.slots[i])
        return result
        #################
    
    def values(self):
        ### CODE HERE ###
        result=[]
        for i in range(len(self.slots)):
            if self.data[i]!=None and self.data[i]!='delete':
                result.append(self.data[i])
        return result
        #################

def plot_collision(): 
    import random
    import matplotlib.pyplot as plt
    ### CODE HERE ###
    chaining=HashTable_Chain(size=307)
    double=HashTable_DoubleHash(size=307) #Table size를 소수로 설정해야 값이 제대로 나온다. 307은 소수이다.
    num_list=[i for i in range(2000)] #임의로 2000내에서 614개의 숫자를 고르도록 했다.
    keys=random.sample(num_list,614)
    data=random.sample(num_list,614)
    c_coll=[]
    d_coll=[]
    for i in range(614):
        chaining.put(keys[i],data[i])
        double.put(keys[i],data[i])
        c_coll.append(chaining.num_collision) #i가 증가함에 따라 늘어나는 collision값을 각각 리스트에 넣어준다.
        d_coll.append(double.num_collision)

    plt.figure(figsize=(10,10)) #
    plt.xlabel('# of elements',size=7)
    plt.ylabel('# of collision',size=7)
    plt.plot(c_coll)
    plt.plot(d_coll)
    plt.legend(['Chaining','Double Hashing'],loc='best',fontsize=10)
    
    plt.show()
    #################


def plot_chain_length():
    import random
    import matplotlib.pyplot as plt
    ### CODE HERE ###
    chaining=HashTable_Chain(size=983)
    num_list=[i for i in range(50000)] #임의로 50000내에서 983개의 숫자를 고르도록 했다.
    keys=random.sample(num_list,983)
    data=random.sample(num_list,983)
    length=[]
    for i in range(983):
        chaining.put(keys[i],data[i])
    for j in range(chaining.size):
        length.append(len(chaining.data[j])) #length에 각 chain의 길이를 넣어준다.
    a=max(length)
    plt.hist(length, range=(0,a), bins=a) #가장 긴 chain의 길이까지로 범위를 설정하고 histogram을 그려준다.
    plt.xlabel('Length of chains',fontsize=10)
    plt.show()
    #################
    
####################################################################################################################
