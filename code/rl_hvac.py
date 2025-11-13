import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
import matplotlib.animation as animation
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os
import random

# 랜덤 시드 설정
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

class HVACEnv(gym.Env):
    def __init__(self, room_size=(20, 20), time_steps=100, target_temp=25, temp_tolerance=1.5):
        super(HVACEnv, self).__init__()
        self.room_size = room_size
        self.time_steps = time_steps
        self.target_temp = target_temp
        self.temp_tolerance = temp_tolerance
        self.current_step = 0
        
        # 액션 공간: [에어컨 강도, 환풍기 강도]
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([10, 10]), dtype=np.float32)
        
        # 상태 공간: 온도만 포함
        self.observation_space = spaces.Box(low=20, high=35, shape=(room_size[0], room_size[1]), dtype=np.float32)
        
        # 벽 설정
        self.walls = np.zeros(room_size, dtype=bool)
        self.walls[0:8, 8] = True  # 예시 벽
        
        # 에어컨과 환풍기 위치 설정
        self.ac_position = (0, 18)
        self.fan_position = (19, 10)
        self.ac_direction = np.array([1, 0])  # 에어컨 바람 방향 (x축)
        
        # 초기 온도 설정
        self.initial_temp = self.create_initial_temperature()
        self.total_energy = 0.0

        self.reset()

    def create_initial_temperature(self):
        temp = np.ones(self.room_size) * 28
        hot_spot = (3, 3)
        x, y = np.meshgrid(np.arange(self.room_size[0]), np.arange(self.room_size[1]))
        temp += 5 * np.exp(-0.1 * ((x - hot_spot[0])**2 + (y - hot_spot[1])**2))
        return temp

    def step(self, action):
        ac_strength, fan_power = action

        prev_temp = self.current_temp.copy()  # 이전 온도를 저장
        # 온도 및 공기 흐름 시뮬레이션
        self.current_temp, self.airflow = self.simulate_temperature_and_airflow(
            self.ac_position, self.ac_direction, ac_strength, 
            self.fan_position, fan_power, self.current_temp, 
            self.walls, time_steps=1
        )
        
        # 리워드 계산
        reward = self.calculate_reward(prev_temp, self.current_temp, ac_strength, fan_power)

        self.current_step += 1
        terminated = np.all(np.abs(self.current_temp - self.target_temp) <= self.temp_tolerance) or self.current_step >= self.time_steps
        truncated = False

        # 관찰값 온도
        obs = self.current_temp

        info = {
            "mean_temp": np.mean(self.current_temp),
            "temp_std": np.std(self.current_temp),
            "total_energy": self.total_energy,
            "steps": self.current_step,
            "goal_achieved": terminated and self.current_step < self.time_steps
        }
        
        return obs, float(reward), terminated, truncated, info

    def calculate_reward(self, prev_temp, current_temp, ac_strength, fan_power):
        # 온도 개선 정도
        prev_diff = np.abs(prev_temp - self.target_temp)
        curr_diff = np.abs(current_temp - self.target_temp)
        temp_improvement = np.mean(prev_diff - curr_diff)
        
        # 온도 균일성
        temp_uniformity = -np.std(current_temp)
        
        # 에너지 소비량 계산
        energy_consumption = 0.5 * ac_strength + 0.3 * fan_power
        self.total_energy += energy_consumption
        
        # 목표 달성 보너스
        goal_bonus = 100 if np.all(curr_diff <= self.temp_tolerance) else 0
        
        # 시간에 따른 패널티 추가
        time_penalty = -0.1 * self.current_step

        # 최종 리워드 계산
        reward = (
            50 * temp_improvement +
            10 * temp_uniformity +
            -5 * energy_consumption +
            goal_bonus +
            time_penalty
        )
        
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_temp = self.initial_temp.copy()
        self.total_energy = 0.0
        return self.current_temp, {}

    def simulate_temperature_and_airflow(self, ac_position, ac_direction, ac_strength, fan_position, fan_power, initial_temp, walls, time_steps=10):
        # 방 크기와 공기 흐름 초기화
        room_size = initial_temp.shape
        temp = initial_temp.copy()
        airflow = np.zeros(room_size + (2,))  # 공기 흐름 (x, y 방향)

        for _ in range(time_steps):
            # 자연적인 열 확산 (벽이 있는 경우 확산 줄어듦)
            laplacian = np.zeros_like(temp)
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = np.roll(np.roll(temp, dx, axis=0), dy, axis=1)
                mask = np.roll(np.roll(walls, dx, axis=0), dy, axis=1)
                laplacian += np.where(mask, temp, neighbor) - temp
            
            temp += 0.05 * laplacian  # 자연 열 확산

            # 에어컨 효과 (거리에 따른 영향 감소)
            ac_effect = ac_strength * np.exp(-0.05 * np.sqrt(
                (np.arange(room_size[0])[:, None] - ac_position[0])**2 +
                (np.arange(room_size[1])[None, :] - ac_position[1])**2
            ))
            temp -= 0.2 * ac_effect

            # 환풍기 효과 (거리에 따른 영향 증가)
            fan_effect = fan_power * 0.2 * np.exp(-0.03 * np.sqrt(
                (np.arange(room_size[0])[:, None] - fan_position[0])**2 +
                (np.arange(room_size[1])[None, :] - fan_position[1])**2
            ))
            temp -= 0.2 * fan_effect

            # 공기 흐름 업데이트
            airflow *= 0.8  # 감쇠 계수
            airflow[ac_position[0], ac_position[1]] += ac_strength * ac_direction
            fan_direction = np.array([-1, 0])  # 환풍기가 공기를 -x 방향으로 보냄
            airflow[fan_position[0], fan_position[1]] += fan_power * 0.4 * fan_direction

            # 벽에 의한 공기 흐름 방해
            airflow[walls] = 0

            # 대류 효과 (온도 기울기를 따라 공기 흐름)
            grad_x, grad_y = np.gradient(temp)
            temp -= 0.2 * (airflow[:,:,0] * grad_x + airflow[:,:,1] * grad_y)

            # 온도를 합리적인 범위로 제한
            temp = np.clip(temp, 20, 35)

        return temp, airflow

def baseline_simulation(env, time_steps=100):
    obs = env.reset()
    baseline_temp_history = []
    baseline_ac_strengths = []
    baseline_fan_powers = []
    
    for _ in range(time_steps):
        ac_strength = 2.0
        fan_power = 2.0
        action = np.array([[ac_strength, fan_power]])  # Wrap in an extra list for vectorized env
        
        obs, reward, done, info = env.step(action)
        
        # Extract the temperature array from the observation
        temp = env.get_attr('current_temp')[0]
        baseline_temp_history.append(temp)
        baseline_ac_strengths.append(ac_strength)
        baseline_fan_powers.append(fan_power)
        
        if done[0]:
            break
    
    return np.array(baseline_temp_history), baseline_ac_strengths, baseline_fan_powers

def visualize_3d_comparison(env, model, baseline_temp_history, baseline_ac_strengths, baseline_fan_powers):
    obs = env.reset()
    trained_temp_history = []
    trained_ac_strengths = []
    trained_fan_powers = []
    total_energy = 0

    for _ in range(len(baseline_temp_history)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        # Extract the temperature array from the observation
        temp = env.get_attr('current_temp')[0]
        trained_temp_history.append(temp)
        trained_ac_strengths.append(action[0][0])
        trained_fan_powers.append(action[0][1])
        total_energy += info[0].get('total_energy', 0)  # Access the first element of info

        if done[0]:
            break

    # Visualization code
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(range(env.get_attr('room_size')[0][0]), range(env.get_attr('room_size')[0][1]))

    def update_plot(frame):
        ax.clear()

        trained_temp = trained_temp_history[frame]

        # 환경(env)에서 벽 위치 마스크 가져오기
        wall_mask = env.get_attr('walls')[0]

        # 온도 데이터에서 벽 위치를 NaN으로 설정하여 벽이 반영되도록 함
        temp_filtered = trained_temp.copy()
        temp_filtered[wall_mask] = np.nan  # 벽 위치의 온도 값을 NaN으로 설정하여 제거

        # 온도 분포를 먼저 그리기 (벽 뒤에 있는 부분도 시각적으로 표현)
        Y, X = np.meshgrid(np.arange(trained_temp.shape[1]), np.arange(trained_temp.shape[0]))
        surf = ax.plot_surface(X, Y, temp_filtered, cmap='coolwarm', vmin=20, vmax=35, rstride=1, cstride=1,
                            norm=colors.Normalize(vmin=20, vmax=35), edgecolor='none', alpha=0.8)

        # 벽 그리기 (벽이 온도 분포를 가리도록 설정)
        wall_x, wall_y = np.where(wall_mask)
        ax.bar3d(wall_x, wall_y - 1, np.full_like(wall_x, 20), 1, 1, 15, color='gray', alpha=0.8)

        # 제목 설정 (글자 크기 크게)
        ax.set_title(f"Trained AI \n\n Total energy consumption: {total_energy:.2f} \n AC intensity: {trained_ac_strengths[frame]:.2f}\n Fan intensity: {trained_fan_powers[frame]:.2f}",
                    fontsize=16)
        ax.set_zlim(20, 35)

        # 에어컨 및 팬 위치 표시 (글자 크기 크게)
        ax.scatter(*env.get_attr('ac_position')[0], 34, color='blue', s=100, label='AC')
        ax.scatter(*env.get_attr('fan_position')[0], 34, color='purple', s=100, label='Fan')
        ax.quiver(*env.get_attr('ac_position')[0], 34, *env.get_attr('ac_direction')[0], 0, color='blue', length=2)

        # 축 설정 (글자 크기 설정 및 숫자 제거)
        ax.set_xlabel('X', fontsize=14)
        ax.set_ylabel('Y', fontsize=14)
        ax.set_zlabel('Temperature (°C)', fontsize=14)

        # 축 눈금 제거 (격자는 유지하되 숫자는 제거)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # 격자 유지
        ax.grid(True)

        # 범례 설정 (글자 크기 크게)
        ax.legend(fontsize=12)

        return [surf]

    ani = animation.FuncAnimation(fig, update_plot, frames=len(trained_temp_history), blit=False, repeat=False)
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=20, vmax=35))
    sm.set_array([])
    cax = fig.add_axes([0.1, 0.05, 0.8, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal', aspect=40)
    cbar.set_label('Temperature (°C)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    ani.save("hvac_agent_performance.gif", writer="pillow", fps=5)
    plt.close(fig)

if __name__ == "__main__":
    # 환경 생성
    env = Monitor(HVACEnv())
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 로그 디렉토리 생성
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)

    # 모델 생성
    model = PPO("MlpPolicy", env,
                learning_rate=1e-3,
                n_steps=1024,
                batch_size=32,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                verbose=1,
                seed=RANDOM_SEED)

    # 콜백 설정
    eval_env = Monitor(HVACEnv())
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    eval_callback = EvalCallback(eval_env, best_model_save_path=log_dir,
                                 log_path=log_dir, eval_freq=1000,
                                 deterministic=True, render=False)
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir,
                                             name_prefix='hvac_model')

    # 모델 학습
    model.learn(total_timesteps=33000, callback=[eval_callback, checkpoint_callback])

    # 학습된 모델 저장
    model.save(log_dir + "hvac_ppo_model_final")

    print("Model training completed and saved.")

    # 학습된 모델 로드
    loaded_model = PPO.load(log_dir + "hvac_ppo_model_final")

    # 베이스라인 시뮬레이션 실행
    print("Running baseline simulation...")
    baseline_temp_history, baseline_ac_strengths, baseline_fan_powers = baseline_simulation(env)

    # 시각화
    print("Generating visualization...")
    visualize_3d_comparison(env, loaded_model, baseline_temp_history, baseline_ac_strengths, baseline_fan_powers)
