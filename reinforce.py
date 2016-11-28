import gym, tflearn
import scipy.signal
import numpy as np
import tensorflow as tf

class GymSimulator(object):
  def __init__(self, env):
    self.env = env

  def episode(self, agent, max_steps=50, render=False):
    observation = self.env.reset()
    observations, actions, rewards = [], [], []
    for _ in xrange(max_steps):
      if render:
        self.env.render()
      action = agent.act([observation])
      (observation, reward, done, _) = self.env.step(action)
      action_vec = np.zeros(self.num_actions())
      action_vec[action] = 1
      actions.append(action_vec)
      observations.append(observation)
      rewards.append(reward)
      if done:
        break
    return {'observations': observations,
            'actions': actions,
            'rewards': rewards}

  def num_actions(self):
    # TODO: this is for inverted pendulum, fix me
    return 2

  def state_space(self):
    # TODO: this is for inverted pendulum, fix me
    return 4

  def discount(self, x, gamma):
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

  def simulate(self, n, agent, max_steps=50, gamma=1.0, render=False):
    episodes = [self.episode(agent, max_steps, render) for _ in xrange(n)]
    observations = np.concatenate([ep["observations"] for ep in episodes])
    actions = np.concatenate([ep["actions"] for ep in episodes])
    rewards = [self.discount(ep['rewards'], gamma) for ep in episodes]
    pr = [np.sum(ep['rewards']) for ep in episodes]
    print "reward: %s +/- %s" % (np.mean(pr), np.std(pr)/np.sqrt(len(pr)))
    padded_rewards = [np.concatenate([r, np.zeros(max_steps-len(r))]) for r in rewards]
    baseline = np.mean(padded_rewards, axis=0)
    advantages = np.concatenate([r - baseline[:len(r)] for r in rewards])
    return observations, actions, advantages


class ReinforceAgent(object):
  def __init__(self, simulator, session):
    self.simulator = simulator
    self.session = session

    # Build the TF graph
    self.build()

  def build(self):
    """
    Construct the TS graph. For ease of access, we'll just save all
    tensor references on self.
    """

    # Forward propagation
    self.states = tf.placeholder(tf.float32,
        shape=(None, self.simulator.state_space()), name="states")
    hidden = tflearn.fully_connected(self.states, 20, activation='tanh')
    self.policy = tflearn.fully_connected(hidden,
        self.simulator.num_actions(), activation='softmax')

    # Training
    self.actions = tf.placeholder(tf.float32,
        shape=(None, self.simulator.num_actions()), name="actions")
    self.advantage = tf.placeholder(tf.float32,
        shape=(None,), name="advantage")

    # self.action_activations = tf.gather_nd(self.policy, self.actions)
    self.action_activations = tf.reduce_sum(self.policy * self.actions,
        reduction_indices=1)
    self.loss = -tf.reduce_mean(tf.log(self.action_activations) * self.advantage)

    self.optimizer = tf.train.RMSPropOptimizer(0.05)
    self.train_op = self.optimizer.minimize(self.loss)

  def categorical_sample(self, prob_n):
      """
      Sample from categorical distribution,
      specified by a vector of class probabilities
      """
      prob_n = np.asarray(prob_n)
      csprob_n = np.cumsum(prob_n)
      return (csprob_n > np.random.rand()).argmax()

  def act(self, observation):
    prob = self.session.run(self.policy, feed_dict={self.states: observation})
    return self.categorical_sample(prob)

  def train(self, saver=None, max_iter=100, batch_size=64, max_steps=50):
    # saver.restore(self.session, "carpole.ckpt-100")
    num_iter = 0
    while True:
      num_iter += 1
      if max_iter is not None and num_iter > max_iter:
        break
      print "iter: %s, episodes: %s" % (num_iter, batch_size * num_iter)
      if saver is not None and num_iter % 20 == 0:
        saver.save(self.session, "carpole.ckpt", global_step=num_iter)
      observations, actions, advantages = self.simulator.simulate(batch_size,
          self, max_steps=max_steps)
      loss, _ = session.run([self.loss, self.train_op],
          feed_dict={self.states: observations,
                     self.actions: actions,
                     self.advantage: advantages})
      print "loss: %s" % loss
      observations, actions, advantages = self.simulator.simulate(1, self,
          max_steps=max_steps, render=True)
      print

  def evaluate(self, saver, max_steps=200):
    saver.restore(self.session, "carpole.ckpt-20")
    for _ in xrange(20):
      observations, actions, advantages = self.simulator.simulate(1, self,
          max_steps=max_steps, render=True)


if __name__ == "__main__":
  with tf.Session() as session:
    env = gym.make('CartPole-v0')
    #env.monitor.start('/tmp/cartpole-experiment-1', force=True)
    simulator = GymSimulator(env)
    agent = ReinforceAgent(simulator, session)
    saver = tf.train.Saver(max_to_keep=5)
    session.run(tf.initialize_all_variables())
    agent.train(saver, max_iter=None, batch_size=200, max_steps=200)
    # agent.evaluate(saver)
    #env.monitor.close()
