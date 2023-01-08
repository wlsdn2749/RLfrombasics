import random
import numpy as np
import time
class GridWorld():
    '''
        init : 환경이 시작될때 초기화
        step : 에이전트가 a라는 매개변수를 받아 액션을 하면 x,y가 변화함으로써 state가 바뀜
               현재 에이전트의 상태(Tuple), 보상, 도착 여부를 리턴하게 된다. 
        move : 방향에 따라 x, y값을 1씩 조정하되, 벽에 닿으면 다시 제자리로 오게된다.
        is_done : 에이전트가 목적지 x=y=3에 도착하게 되면 True값 반환 아니면 False
        reset : x,y값을 0으로 초기화 하고 x,y값을 반환
    '''
    def __init__(self):
        self.x=0
        self.y=0
    
    def step(self, a):
        # 0번 액션: 왼쪽, 1번 액션: 위, 2번 액션: 오른쪽, 3번 액션: 아래쪽
        if a==0:
            self.move_left()
        elif a==1:
            self.move_up()
        elif a==2:
            self.move_right()
        elif a==3:
            self.move_down()

        reward = -1 # 보상은 항상 -1로 고정
        done = self.is_done()
        return (self.x, self.y), reward, done

    def move_right(self):
        self.y += 1  
        if self.y > 3:
            self.y = 3
      
    def move_left(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0
      
    def move_up(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0
  
    def move_down(self):
        self.x += 1
        if self.x > 3:
            self.x = 3

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True
        else :
            return False

    def get_state(self):
        return (self.x, self.y)
      
    def reset(self):
        self.x = 0
        self.y = 0
        return (self.x, self.y)

class Agent():
    '''
        select_action : coin값에 따라 0.25확률로 동서남북으로 움직이는 액션을 반환하는 함수
    '''
    def __init__(self):
        pass        

    def select_action(self):
        coin = random.random()
        if coin < 0.25:
            action = 0
        elif coin < 0.5:
            action = 1
        elif coin < 0.75:
            action = 2
        else:
            action = 3
        return action


def main():
    '''
        env     : 환경
        agent   : 에이전트
        data    : 4x4 행렬 (state map?)
        gamma   : 할인 계수 (discount factor)  
        reward  : 스텝을 이동할때마다 받는 보상
        alpha   : 경험을 통해 리턴을 업데이트
    '''
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.001
    
    
    epochs = [100, 1000, 5000, 25000, 50000]
    for epoch in epochs:
        data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] # epoch별로 달라지는 값을 보기위해 학습된 데이터를 초기화
        start = time.time()
        for k in range(epoch):
            done = False
            history = []

            while not done:
                action = agent.select_action()
                (x,y), reward, done = env.step(action)
                history.append((x,y,reward)) # 최신본이 뒤로 가게 되기 떄문에
            env.reset()

            cum_reward = 0
            for transition in history[::-1]: # 역으로 search를 해준다.
                x, y, reward = transition
                data[x][y] = data[x][y] + alpha*(cum_reward-data[x][y])
                cum_reward = reward + gamma*cum_reward  # 책에 오타가 있어 수정하였습니다 R_t+1 + gamma G_t+1

        print("epoch : " , epoch)   # epoch 별로 수행  
        for row in data:
            print(row)
        end = time.time()
        print(epoch, ":" , round(end-start, 3))
        print("\n\n\n")
        
        
if __name__ == '__main__':
    main()
    
    
