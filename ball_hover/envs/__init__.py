from gym.envs.registration import register

register(id="BallHoverEnv-v0", entry_point="envs.ball_hover_env_dir:BallHoverEnv")

register(id="BallControlEnv-v0", entry_point="envs.ball_control_env_dir:BallControlEnv")

register(id="BallSpeedEnv-v0", entry_point="envs.ball_speed_env_dir:BallSpeedEnv")
