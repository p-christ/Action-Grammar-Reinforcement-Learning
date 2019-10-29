from environments.Open_AI_Wrappers import *
import gym

def make_atari(env_id, seed, done_on_lost_life):
    """Creates a wrapped Atari environment"""
    env = gym.make(env_id)
    env.seed(seed)
    if hasattr(gym.spaces, 'prng'):
        gym.spaces.prng.seed(seed)
    assert "Deterministic" in env_id
    assert "v4" in env.spec.id
    if not "SpaceInvaders" in env.spec.id:
        assert env.unwrapped.frameskip == 4
    if done_on_lost_life:
        env = EpisodicLifeEnv(env, play_full_game=False)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = TimeLimit(env, max_episode_steps=27000)
    return env

def create_atari_environments(env_id, seed):
    """Creates a training and an evaluation Atari environment"""
    training_env = make_atari(env_id, seed, done_on_lost_life=True)
    eval_env =  make_atari(env_id, seed, done_on_lost_life=False)
    return training_env, eval_env