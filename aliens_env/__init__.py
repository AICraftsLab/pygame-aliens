from gymnasium.envs.registration import register

register(
    id="Aliens",
    entry_point="aliens_env.envs:AliensEnv"
)