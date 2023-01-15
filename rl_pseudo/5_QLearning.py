import random
import numpy as np

class GridWorld():
    ########################################################
    # Q-Learning
    # TO-DO
    # 환경을 나타내는 class, 7x5 gridworld의 환경을 담당한다.
    # 환경의 x값과 y값은 현재 agent의 위치를 의미한다
    # 첫 환경의 x,y는 0으로 시작하고
    # 오른쪽으로 움직이면 x값을 +1 
    # 아래로 움직이면 y값을 +1
    # 왼쪽은 x을 -1
    # 위는 y를 -1 
    # 만약 7x5의 환경을 넘어가는 행동을한다면 제자리로 돌아오게 된다.

    # choose_action함수는 action을 인자로 받아 agent가 하는 행동을 의미한다.
    # action의 값에 따라 agent가 상하좌우로 이동하고
    # 이동시에 받는 보상은 항상 -1로 
    # 현재 상태에서 액션을 함으로써 다음 상태, 보상, 목적지 도착 여부를 반환하게 된다.
    # reset 함수는 환경을 초기화 시킨다.
    # is_done 함수는 환경이 목적지에 도착했는지 안했는지를 반환한다.
    # get_state 함수는 환경의 현재 state를 반환한다.
    ########################################################
    pass

class QAgent():
    ########################################################
    # TO-DO
    # QAgent class는 Agent를 나타낸다
    # __init__() 함수는 7X5환경의 4방향의 값을 가지는 q_table을 0으로 초기화한다.
    # 계산에 필요한 eps와 alpha를 각각 정한다. ( 0.9 , 0.01)
    # select_action 함수는 action을 선택하는 policy를 나타낸다 우리는 delayed e-greedy를 사용할 것이기 떄문에
    # 랜덤값을 구해 처음엔 탐색이 많이 일어나게끔 학습하고, 나중엔 그리디가 많이 일어나게끔 한다.
    # action을 
    #   랜덤하게 정하는 방법은, 0~3의 랜덤한 int값을 구해서 방향값을 정하면 된다.
    #   그리디 하게 정하는 방법은, 해당하는 셀의 q_value를 구해 가장 큰 q_value를 가지고 있는 방향을 action으로 구한다
    # update_table함수는 q_table을 TD방법으로 업데이트하는 방법(Q-learning)
    # 단 여기서 TD Target 을 r+ np.amax(q_table(x',y',:))로 해서 greedy한 정책을 가져가야한다.
    # target policy와 behaviour policy가 서로 다르다, off-policy
    # anneal_eps 함수는 delayed e-greedy를 사용하기에 e를 0.03씩 감소시켜 최소 0.2까지 감소시키게 끔 만들어준다
    # show_table은 q_table의 어느액션이 가장 값이 높았는지를 반환해준다.
    ########################################################
    pass
      
def main():
    env = GridWorld()
    agent = QAgent()

    for n_epi in range(1000): # 총 1,000 에피소드 동안 학습
        done = False
        
        s = env.reset()
        while not done: # 한 에피소드가 끝날 때 까지
            a = agent.select_action(s)
            s_prime, r, done = env.step(a)
            agent.update_table((s,a,r,s_prime)) # SARSA는 history가 필요없다.
            s = s_prime
        
        agent.anneal_eps()

    agent.show_table() # 학습이 끝난 결과를 출력

if __name__ == '__main__':
    main()