import random
import numpy as np
import time 
class GridWorld():
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
    #TD
    env = GridWorld()
    agent = Agent()
    data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    gamma = 1.0
    reward = -1
    alpha = 0.01

    epochs = [100, 1000, 5000, 25000, 50000]
    for epoch in epochs:
        data = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]] # epoch별로 달라지는 값을 보기위해 학습된 데이터를 초기화
        start = time.time()
        for k in range(epoch):
            done = False
            while not done:
                x, y = env.get_state()
                action = agent.select_action()
                (x_prime, y_prime), reward, done = env.step(action)
                x_prime, y_prime = env.get_state()
                
                data[x][y] = data[x][y] + alpha*(reward+gamma*data[x_prime][y_prime]-data[x][y])
                #V(s_t) = V(s_t) + alpha(r_t+1 + gamma*V(s_t+1) - V(s_t)
            env.reset()
            
        
        print("epoch : " , epoch)   # epoch 별로 수행  
        for row in data:
            print(row)
        end = time.time()
        print(epoch, ":" , round(end-start, 3))
        print("\n\n\n")

if __name__ == '__main__':
    main()