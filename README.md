# RLfrombasics

<img src="https://user-images.githubusercontent.com/8207326/93460041-a7096500-f91d-11ea-9797-583677d2c898.jpg" height="400"></img>

"바닥부터 배우는 강화학습"에 수록된 코드를 모아놓은 레포입니다.
누군가에게 도움이 되는 책이었으면 좋겠습니다.
감사합니다.

This repo provides all the codes from the book "RLfrombasics"
Hope this book is useful to somebody
Thankyou :)

# Typo(오타)

1. 챕터 5 : ch5_mclearning.py 코드 97라인  <br>
(수정 전) cum_reward = cum_reward + gamma * reward <br>
(수정 후) cum_reward = reward + gamma * cum_reward <br>
Thanks to goodjian7

2. 챕터 3 : 67, 69 페이지 OX 퀴즈 <br>
(수정 전) r_t+1 + gamma * r_t+1 + ... <br>
(수정 후) r_t+2 + gamma * r_t+2 + ... <br>
Thanks to namdori61



### 1~5 's base line code is same
#### Difference is ..
- MCLearning is base
- TDLearning = not history, instead of G_t
using next s', a'
- MCControl = not Value Iteration, using Q_value iteration
- SARSA (TDControl) = not history, instead of G_t
using next s', a'
- Q-learning = Off-Policy, using target greedy-policy maxQ(s', a') is different from behaviour eps-greedy policy Q(s,a)