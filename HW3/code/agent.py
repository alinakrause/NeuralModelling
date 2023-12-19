import numpy as np
from scipy import stats # for gaussian noise
from environment import Environment, Environment_TwoStepAgent
import random



def allmax(a):
    if len(a) == 0:
        return []
    all_ = [0]
    max_ = a[0]
    for i in range(1, len(a)):
        if a[i] > max_:
            all_ = [i]
            max_ = a[i]
        elif a[i] == max_:
            all_.append(i)
    return all_


class DynaAgent(Environment):

    def __init__(self, alpha, gamma, epsilon,greedy):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha   -- learning rate \in (0, 1]
            gamma   -- discount factor \in (0, 1)
            epsilon -- controls the influence of the exploration bonus
        '''

        self.alpha   = alpha
        self.gamma   = gamma 
        self.epsilon = epsilon
        self.greedy = greedy

        return None

    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''

        Environment.__init__(self, env_config)

        return None

    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.Q = np.zeros((self.num_states, self.num_actions))

        return None

    def _init_experience_buffer(self):

        '''
        Initialise the experience buffer
        '''

        self.experience_buffer = np.zeros((self.num_states*self.num_actions, 4), dtype=int)
        for s in range(self.num_states):
            for a in range(self.num_actions):
                self.experience_buffer[s*self.num_actions+a] = [s, a, 0, s]

        return None

    def _init_history(self):

        '''
        Initialise the history
        '''

        self.history = np.empty((0, 4), dtype=int)

        return None
    
    def _init_action_count(self):

        '''
        Initialise the action count
        '''

        self.action_count = np.zeros((self.num_states, self.num_actions), dtype=int)

        return None

    def _update_experience_buffer(self, s, a, r, s1):

        '''
        Update the experience buffer (world model)
        Input arguments:
            s  -- initial state
            a  -- chosen action
            r  -- received reward
            s1 -- next state
        '''

        # complete the code
        self.experience_buffer[s*self.num_actions+a,:] = np.asarray((s,a,r,s1))

        return None

    def _update_qvals(self, s, a, r, s1, bonus=False):

        '''
        Update the Q-value table
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
            bonus -- True / False whether to use exploration bonus or not
        '''

        e = self.Q[s1,np.argmax(self.Q[s1,:])]

        self.Q[s,a] = self.Q[s,a] + self.alpha*(r + (self.epsilon*bonus*np.sqrt(self.action_count[s,a])) + self.gamma * e - self.Q[s,a])

        
        # complete the code

        return None

    def _update_action_count(self, s, a):

        '''
        Update the action count
        Input arguments:
            Input arguments:
            s  -- initial state
            a  -- chosen action
        '''

        # time since last visited
        # everything +1 ago
        self.action_count = self.action_count + 1
        # last visited 0 timesteps ago
        self.action_count[s,a] = 0

        # complete the code

        return None

    def _update_history(self, s, a, r, s1):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([s, a, r, s1])))

        return None



    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''

        # complete the code
        q_values = self.Q[s,:] + self.epsilon*np.sqrt(self.action_count[s,:])
        q_values = random.choice(allmax(q_values))

        if self.greedy:
            epsilon_greedy = 0.1
            p = np.random.random()
            p = np.random.random() < epsilon_greedy
            if p:
                q_values = np.random.randint(0,4)


        return q_values
    



    def _plan(self, num_planning_updates):

        '''
        Planning computations
        Input arguments:
            num_planning_updates -- number of planning updates to execute
        '''

        # complete the code
        for i in range(num_planning_updates):
            #num states
            rows = len(self.experience_buffer)

            # pick a state
            select_ind = np.random.randint(rows)
            s,a,r,s1 = self.experience_buffer[select_ind]
            
            self._update_qvals(s, a, r, s1, bonus=True)

        return None

    def get_performace(self):

        '''
        Returns cumulative reward collected prior to each move
        '''

        return np.cumsum(self.history[:, 2])

    def simulate(self, num_trials, reset_agent=True, num_planning_updates=None):

        '''
        Main simulation function
        Input arguments:
            num_trials           -- number of trials (i.e., moves) to simulate
            reset_agent          -- whether to reset all knowledge and begin at the start state
            num_planning_updates -- number of planning updates to execute after every move
        '''

        if reset_agent:
            self._init_q_values()
            self._init_experience_buffer()
            self._init_action_count()
            self._init_history()

            self.s = self.start_state

        for _ in range(num_trials):

            # choose action
            a  = self._policy(self.s)
            # get new state
            s1 = np.random.choice(np.arange(self.num_states), p=self.T[self.s, a, :])
            # receive reward
            r  = self.R[self.s, a]
            # learning
            self._update_qvals(self.s, a, r, s1, bonus=False)
            # update world model 
            self._update_experience_buffer(self.s, a, r, s1)
            # reset action count
            self._update_action_count(self.s, a)
            # update history
            self._update_history(self.s, a, r, s1)
            # plan
            if num_planning_updates is not None:
                self._plan(num_planning_updates)

            if s1 == self.goal_state:
                self.s = self.start_state
            else:
                self.s = s1

        return None
    
class TwoStepAgent(Environment_TwoStepAgent):

    def __init__(self, alpha1, alpha2, beta1, beta2, lam, w, p):

        '''
        Initialise the agent class instance
        Input arguments:
            alpha1 -- learning rate for the first stage \in (0, 1]
            alpha2 -- learning rate for the second stage \in (0, 1]
            beta1  -- inverse temperature for the first stage
            beta2  -- inverse temperature for the second stage
            lam    -- eligibility trace parameter
            w      -- mixing weight for MF vs MB \in [0, 1] 
            p      -- perseveration strength
        '''

        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.beta1  = beta1
        self.beta2  = beta2
        self.lam    = lam
        self.w      = w
        self.p      = p

        self.rewards = np.random.uniform(0.25,0.75,4)


        self.transition_matrix = [[0.7,0.3], [0.3,0.7]]
        self.transition = self.transition_matrix[0] 
        self.transition_count_b = 0
        self.transition_count_c = 0

        self.last_a = -1

        return None
        
    def init_env(self, env_config):

        '''
        Initialise the environment
        Input arguments:
            **env_config -- dictionary with environment parameters
        '''
        Environment_TwoStepAgent.__init__(self, env_config)
        return None
    
    
    def _init_q_values(self):

        '''
        Initialise the Q-value table
        '''

        self.QTD = np.zeros((self.num_states* self.num_actions))
        self.QMB = np.zeros((self.num_states* self.num_actions))
        self.Qnet = np.zeros((self.num_states* self.num_actions))
        return None
    

    def _init_history(self):

        '''
        Initialise history to later compute stay probabilities
        '''

        self.history = np.empty((0, 3), dtype=int)

        return None
    
    def _update_history(self, a, s1, r):

        '''
        Update the history
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''

        self.history = np.vstack((self.history, np.array([a, s1, r])))
        return None


    def add_noise(self):
        '''
        Update rewards using a Gaussian random walk and keep them within boundaries.
        '''
        # Update rewards with a Gaussian random walk
        self.rewards += np.random.normal(loc=0, scale=0.025, size=4)
        self.rewards = np.clip(self.rewards, 0.25, 0.75)
        return None




    def _Q_td(self, s, a, r, s1, a1, trace):

        '''
        Update the Q-value table of td
        Input arguments:
            s     -- initial state
            a     -- chosen action
            r     -- received reward
            s1    -- next state
        '''
        
        if not s == 0:
            alpha = self.alpha2
            delta = r - self.QTD[s * self.num_actions + a]
            self.QTD[s * self.num_actions + a] += alpha * delta
        else:
            alpha = self.alpha1
            delta = r - self.QTD[s1 * self.num_actions + a1] if trace else r + self.QTD[s1 * self.num_actions + a1] - self.QTD[s * self.num_actions + a]
            self.QTD[s * self.num_actions + a] += alpha * delta * (self.lam * trace) if trace else alpha * delta


        return None
    
    def _Q_modelBased(self, s, a):

        '''
        Update the Q-value table of mb
        Input arguments:
            s     -- initial state
            a     -- chosen action
        '''

        if s != 0:
            self.QMB[s * self.num_actions + a] = self.QTD[s * self.num_actions + a]
            return None

        if self.transition_count_b > self.transition_count_c:
            self.transition = self.transition_matrix[0]
        else:
            self.transition = self.transition_matrix[1]

        if s == 0:
            p_a = self.transition[a]
            p_not_a = 1-p_a

            maxQ_s2 = np.max(self.QTD[[2,3]]) 
            maxQ_s3 = np.max(self.QTD[[4,5]])
            self.QMB[s * self.num_actions + a] = p_a*maxQ_s2 +p_not_a*maxQ_s3

        
        return None
    
    def _Q_net(self, s, a):

        '''
        Update the Q-value table of combined td and mb
        Input arguments:
            s     -- state
            a     -- chosen action

        '''
        s_a = s*self.num_actions+a

        if s == 0:
            self.Qnet[s_a] = self.w * self.QMB[s_a] + (1-self.w)*self.QTD[s_a]
        else:
            self.Qnet[s_a] = self.QTD[s_a]

        return None

    
    def _policy(self, s):

        '''
        Agent's policy 
        Input arguments:
            s -- state
        Output:
            a -- index of action to be chosen
        '''
        exp_terms = np.zeros(2)
        
        for a in range(self.num_actions):
            if s == 0:
                beta = self.beta1
                rep = self.last_a == a
            else:
                beta = self.beta2
                rep = 0
            s_a = s*self.num_actions+a

            exp_terms[a] = np.exp(beta*(self.Qnet[s_a] + self.p * rep)) #qnet

        policy =  exp_terms/ exp_terms.sum()

        a = np.random.choice(np.arange(2),p=policy)
        return a
    
    def get_next_state(self,s,a):
        if s == 0:
            p = [self.transition_matrix[0][a], 1-self.transition_matrix[0][a]]
            new_state = np.random.choice([1,2], p=p)

        return new_state

    def get_stay_probabilities(self):

        '''
        Calculate stay probabilities
        '''

        common_r      = 0
        num_common_r  = 0
        common_nr     = 0
        num_common_nr = 0
        rare_r        = 0
        num_rare_r    = 0
        rare_nr       = 0
        num_rare_nr   = 0

        num_trials = self.history.shape[0]
        for idx_trial in range(num_trials-1):
            a, s1, r1 = self.history[idx_trial, :]
            a_next    = self.history[idx_trial+1, 0]

            # common
            if (a == 0 and s1 == 1) or (a == 1 and s1 == 2):
                # rewarded
                if r1 == 1:
                    if a == a_next:
                        common_r += 1
                    num_common_r += 1
                else:
                    if a == a_next:
                        common_nr += 1
                    num_common_nr += 1
            else:
                if r1 == 1:
                    if a == a_next:
                        rare_r += 1
                    num_rare_r += 1
                else:
                    if a == a_next:
                        rare_nr += 1
                    num_rare_nr += 1

        return np.array([common_r/num_common_r, rare_r/num_rare_r, common_nr/num_common_nr, rare_nr/num_rare_nr])

    def simulate(self, num_trials):

        '''
        Main simulation function
        Input arguments:
            num_trials -- number of trials to simulate
        '''   

        self._init_q_values()
        self._init_history()

        for _ in range(num_trials):
            s1 = self.start_state
            # choose action
            a1  = self._policy(s1)

            # get new state
            r1=0
            s2 = self.get_next_state(s1,a1)

            if (s1==1 and a1==0) or (s1==2 and a1==1):
                self.transition_count_b+=1
            elif (s1==2 and a1==0) or (s1==1 and a1==1):
                self.transition_count_c+=1

            self.last_a = a1

            # receive reward
            a2 = self._policy(s2)
            self._Q_td(s1, a1, r1, s2,a2,False)
            p = self.rewards[s2]
            r2 = np.random.choice((0,1), p=(1-p, p))

            # learning
            self._Q_td(s2, a2, r2, _,_,False)
            self._Q_td(s1, a1, r2, s2,a2,True)

            self._Q_modelBased(s1,a1)
            self._Q_modelBased(s2,a2)

            self._Q_net(s1,a1)
            self._Q_net(s2,a2)

            #update history
            self._update_history(a1, s2, r2)
            self.add_noise()
            
        return None