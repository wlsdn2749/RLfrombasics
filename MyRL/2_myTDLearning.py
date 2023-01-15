import numpy as np
import random

class GridWorld():
    ########################################################
    # TO-DO
    # 환경을 나타내는 class, 4x4 gridworld의 환경을 담당한다.
    # 환경의 x값과 y값은 현재 agent의 위치를 의미한다
    # 첫 환경의 x,y는 0으로 시작하고
    # 오른쪽으로 움직이면 x값을 +1 
    # 아래로 움직이면 y값을 +1
    # 왼쪽은 x을 -1
    # 위는 y를 -1 
    # 만약 4x4의 환경을 넘어가는 행동을한다면 제자리로 돌아오게 된다.

    # choose_action함수는 action을 인자로 받아 agent가 하는 행동을 의미한다.
    # action의 값에 따라 agent가 상하좌우로 이동하고
    # 이동시에 받는 보상은 항상 -1로 
    # 현재 상태에서 액션을 함으로써 다음 상태, 보상, 목적지 도착 여부를 반환하게 된다.
    # reset 함수는 환경을 초기화 시킨다.
    # is_done 함수는 환경이 목적지에 도착했는지 안했는지를 반환한다.
    # get_state 함수는 환경의 현재 state를 반환한다.
    ########################################################
    def __init__(self):
        self.x = 0
        self.y = 0
    
    def right_action(self):
        self.x += 1
        if self.x > 3:
            self.x = 3
    
    def down_action(self):
        self.y += 1
        if self.y > 3:
            self.y = 3
    
    def left_action(self):
        self.x -= 1
        if self.x < 0:
            self.x = 0

    def up_action(self):
        self.y -= 1
        if self.y < 0:
            self.y = 0  
    
    def choose_action(self, action):
        if action == 0:
            self.right_action()
        if action == 1:
            self.down_action()
        if action == 2:
            self.left_action()
        if action == 3:
            self.up_action()

        reward = -1

        done = self.is_done()
        return (self.x, self.y), reward, done

    def reset(self):
        self.x = 0
        self.y = 0

    def is_done(self):
        if self.x == 3 and self.y == 3:
            return True 
        else: 
            return False

    def get_state(self):
        return ( self.x, self.y )

class Agent():
    ########################################################
    # TO-DO
    # select_action 함수는 agent가 하는 행동을 의미한다.
    # 특정 policy에 따라 agent는 행동을 하게 되고
    # 여기서는 4방향 uniform random으로 동작하게 된다.
    # 어떻게 움직이냐에 따라 움직임값 상수를 리턴하게 된다 (상,하,좌,우 -> 0,1,2,3) 순서
    ########################################################
    def select_action(self):
        probs = random.random()
        if probs < 0.25:
            return 0
        if probs < 0.50:
            return 1
        if probs < 0.75:
            return 2
        if probs < 1:
            return 3

def main():
    env = GridWorld()
    agent = Agent()
    alpha = 0.01 # TD러닝은 1-step마다 일어나기 때문 MC보다 높은 alpha값을 주었음.
    gamma = 1.0 
    grid = np.array(np.zeros((4,4)))
       
    # alpha = 0.001
    # gamma = 1.0
    # data = 4x4
    # policy = 0.25 uniform distribution
    ########################################################
    # TO-DO
    # 횟수 지정 보통 10000 for1
        # 완료 여부를 초기화 시킨다.
        # episode가 끝날때까지 반복 for2
        # 1. agent는 action을 한다. action값을 변수에 저장한다
        # 2. 현재 스테이트를 환경에서 가져와서 저장한다.
        # 3. 특정 state에서 action을 하여 반환되는 다음 스테이트, 보상, 목적지 도착 여부를 저장하고
        # 4. why? Temporal-Difference Method는 따로 저장하는 메모리가 필요없다. 
        # 5. V(S_t) = V(S_t) + alpha*(R_t + gamma*V(S_t+1) - V(S_t)) 공식을 사용하여 V(S_t)를 업데이트 시킨다.
        # 환경을 초기화 시킨다.
    ########################################################

    for iter in range(10000):
        done = False
        while not done:
            action = agent.select_action()
            x, y = env.get_state()
            (x_prime, y_prime), reward, done = env.choose_action(action)
            grid[x][y] = grid[x][y] + alpha * (reward + gamma*grid[x_prime][y_prime] - grid[x][y])
        env.reset()
        
    for row in grid:
        print( row )

if __name__ == "__main__":
    main()