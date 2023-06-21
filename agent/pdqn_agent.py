from . import BaseAgent

class PDQNAgent(BaseAgent):
    def __init__(self, action_space, ob_generator, reward_generator,reward_generator2):
        super().__init__(action_space)
        self.ob_generator = ob_generator
        self.reward_generator = reward_generator
        self.reward_generator2 = reward_generator2

    def get_ob(self):
        return self.ob_generator.generate()

    def get_reward(self):
        reward = self.reward_generator.generate()
        reward2 = self.reward_generator2.generate()
        assert len(reward) == 1
        return reward,reward2

    def get_action(self, ob):
        return self.action_space.sample()