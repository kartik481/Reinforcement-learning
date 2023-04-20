
############################################################################################################
##########################            RL2023 Assignment Answer Sheet              ##########################
############################################################################################################

# **PROVIDE YOUR ANSWERS TO THE ASSIGNMENT QUESTIONS IN THE FUNCTIONS BELOW.**

############################################################################################################
# Question 2
############################################################################################################

def question2_1() -> str:
    """
    (Multiple choice question):
    For the Q-learning algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_2() -> str:
    """
    (Multiple choice question):
    For the First-visit Monte Carlo algorithm, which value of gamma leads to the best average evaluation return?
    a) 0.99
    b) 0.8
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_3() -> str:
    """
    (Multiple choice question):
    Between the two algorithms (Q-Learning and First-Visit MC), whose average evaluation return is impacted by gamma in
    a greater way?
    a) Q-Learning
    b) First-Visit Monte Carlo
    return: (str): your answer as a string. accepted strings: "a" or "b"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a" or "b"
    return answer


def question2_4() -> str:
    """
    (Short answer question):
    Provide a short explanation (<100 words) as to why the value of gamma affects more the evaluation returns achieved
    by [Q-learning / First-Visit Monte Carlo] when compared to the other algorithm.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Monte Carlo(MC) calculates the expected return by adding the discounted rewards for the entire episode, whereas Q-learning calculates the Q-values by bootstrapping, or estimating the maximum Q-value for the next state. The learning rate(alpha) determines how much weight to give to the newly observed reward in the update, reducing the effect. The discount factor has a direct impact on expected returns in MC because it determines the weighting of future rewards. The discount factor determines the weighting of future Q-values in Q-learning, but it is not as directly related to expected returns as it is in Monte Carlo."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 3
############################################################################################################

def question3_1() -> str:
    """
    (Multiple choice question):
    In Reinforce, which learning rate achieves the highest mean returns at the end of training?
    a) 6e-1
    b) 6e-2
    c) 6e-3
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_2() -> str:
    """
    (Multiple choice question):
    When training DQN using a linear decay strategy for epsilon, which exploration fraction achieves the highest mean
    returns at the end of training?
    a) 0.75
    b) 0.25
    c) 0.01
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_3() -> str:
    """
    (Multiple choice question):
    When training DQN using an exponential decay strategy for epsilon, which exploration fraction achieves the highest
    mean returns at the end of training?
    a) 1.0
    b) 0.75
    c) 0.001
    return: (str): your answer as a string. accepted strings: "a", "b" or "c"
    """
    answer = "c"  # TYPE YOUR ANSWER HERE "a", "b" or "c"
    return answer


def question3_4() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 1.0?
    a) 0.0
    b) 1.0
    c) epsilon_min
    d) approximately 0.0057
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "b"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_5() -> str:
    """
    (Multiple choice question):
    What would the value of epsilon be at the end of training when employing an exponential decay strategy
    with epsilon decay set to 0.990?
    a) 0.990
    b) 1.0
    c) epsilon_min
    d) approximately 0.0014
    e) it depends on the number of training timesteps
    return: (str): your answer as a string. accepted strings: "a", "b", "c", "d" or "e"
    """
    answer = "a"  # TYPE YOUR ANSWER HERE "a", "b", "c", "d" or "e"
    return answer


def question3_6() -> str:
    """
    (Short answer question):
    Based on your answer to question3_5(), briefly  explain why a decay strategy based on an exploration fraction
    parameter (such as in the linear decay strategy you implemented) may be more generally applicable across
    different environments  than a decay strategy based on a decay rate parameter (such as in the exponential decay
    strategy you implemented).
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "Determining the optimal exploration rate and decay rates for each environment may be difficult if you use a fixed decay rate. Various environments might have different optimal exploration and decay rates. When dealing with different environments that have different optimal exploration rates and a decay method based on an exploration fraction parameter might offer additional flexibility in adjusting the exploration rate. This is in contrasts with a decay approach based on a fixed decay rate parameter which might not be the best option as seen in above question after reaching a certain threshold the exploration rate remains constant."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


def question3_7() -> str:
    """
    (Short answer question):
    In DQN, explain why the loss is not behaving as in typical supervised learning approaches
    (where we usually see a fairly steady decrease of the loss throughout training)
    return: answer (str): your answer as a string (150 words max)
    """
    answer = "Because in supervised learning the target values are remain stationary throughout the training. But in DQN(which is an off-policy algorithm) the target value is frequently updated using the Bellman equation, which also takes into account the estimated Q-values from past experiences resulting in a non-stationary target values leading to fluctuations in the loss function. As a result unlike in supervised learning, the loss usually does not decrease."  # TYPE YOUR ANSWER HERE (150 words max)
    return answer


def question3_8() -> str:
    """
    (Short answer question):
    Provide an explanation for the spikes which can be observed at regular intervals throughout
    the DQN training process.
    return: answer (str): your answer as a string (100 words max)
    """
    answer = "This happens when the agent learns a new behaviour that outperforms its previous one but forgets the information it previously learned. This is due to the fact that DQN is based on gradient descent optimisation, which updates network weights based on the most recent experiences. As a result, as the agent gains new experiences, the Q-values for previously learned state-action pairs may change dramatically, resulting in spikes in loss graph. Also, to reduce the correlation between experiences, experience replay stores previous experiences in a buffer and samples from it at random."  # TYPE YOUR ANSWER HERE (100 words max)
    return answer


############################################################################################################
# Question 5
############################################################################################################

def question5_1() -> str:
    """
    (Short answer question):
    Provide a short description (200 words max) describing your hyperparameter turning and scheduling process to get
    the best performance of your agents
    return: answer (str): your answer as a string (200 words max)
    """
    answer = "Considering I was already easily getting expected rewards above 150 in Question 4 by adjusting the hidden sizes of the neural networks. I therefore used the question 4 hyper-parameters as a baseline and performed a grid search by taking limited samples(not random but handpicked) in the area around the question 4 hyper-parameters. Nevertheless, I did not adjust every hyper-parameter, such as gamma (0.99), because a greater value of gamma suggests that the agent places more significance on future rewards but we want to solve the environment as a whole, not simply put a focus on short-term rewards. And by taking this strategy, I obtained the best hyper-parameter with the expected returns greater than 280 by training this configuration overnight. I additionally employed the epsilon-greedy scheduling technique, which at first encourages the agent to take action by adding some noise but eventually as agent learns about the environment it will take more deterministic actions. And to check if the results were consistent, I trained it over different seeds and evaluated it over 100 independent episodes."  # TYPE YOUR ANSWER HERE (200 words max)
    return answer