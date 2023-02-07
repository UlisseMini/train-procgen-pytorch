# %%
import matplotlib.pyplot as plt
from procgen import ProcgenGym3Env
import numpy as np
from common.env.procgen_wrappers import VecExtractDictObs, DirectActionsWrapper, DirectGridFrame
from gym3 import ToBaselinesVecEnv

# %%

venv = ProcgenGym3Env(
    num=1,
    env_name='maze',
    start_level=0,
    num_levels=1,
    num_threads=1,
    render_mode="rgb_array",
)

venv = ToBaselinesVecEnv(venv)
venv = VecExtractDictObs(venv, "rgb")
venv = DirectActionsWrapper(venv)
venv = DirectGridFrame(venv)

# %%

obs = venv.reset()
assert obs[0].shape == venv.observation_space.shape, f'got {obs[0].shape} != want {venv.observation_space.shape}'
plt.imshow(obs[0].transpose(1,2,0), origin='lower')

# %%

obs, rew, done, info = venv.step(np.array([3]))
plt.imshow(obs[0].transpose(1,2,0), origin='lower')

# %%

venv.observation_space.shape
# %%
