import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend تعاملی
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO  # برای یادگیری تقویتی
import gym  # برای تعریف محیط سفارشی

# تعریف محیط سفارشی
class CustomGridEnv(gym.Env):
    def __init__(self, grid_size=10):
        super(CustomGridEnv, self).__init__()
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size, grid_size))  # محیط خالی سه‌بعدی

        # قرار دادن منابع انرژی
        self.num_resources = 5
        self.resource_positions = np.random.randint(0, grid_size, (self.num_resources, 3))
        for pos in self.resource_positions:
            self.grid[pos[0], pos[1], pos[2]] = 1  # 1 نشان‌دهنده منبع انرژی است

        # قرار دادن موانع
        self.num_obstacles = 10
        self.obstacle_positions = np.random.randint(0, grid_size, (self.num_obstacles, 3))
        for pos in self.obstacle_positions:
            self.grid[pos[0], pos[1], pos[2]] = -1  # -1 نشان‌دهنده موانع است

        # حالت اولیه ربات
        self.agent_pos = np.random.randint(0, grid_size, size=3)
        self.agent_energy = 100

        # فضای عملیات و مشاهده
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)  # حرکت در سه بعد
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(low=0, high=self.grid_size - 1, shape=(3,), dtype=np.float32),
            "grid": gym.spaces.Box(low=-1, high=1, shape=(self.grid_size, self.grid_size, self.grid_size), dtype=np.float32)
        })

    def reset(self):
        # بازنشانی موقعیت ربات و انرژی
        self.agent_pos = np.random.randint(0, self.grid_size, size=3)
        self.agent_energy = 100
        return self._get_obs(), {}

    def step(self, action):
        dx, dy, dz = action
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy
        new_z = self.agent_pos[2] + dz

        reward = -1  # جایزه پایه
        done = False

        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 0 <= new_z < self.grid_size:
            if self.grid[new_x, new_y, new_z] == 1:  # یافتن منبع
                reward = 10
                self.grid[new_x, new_y, new_z] = 0  # منبع مصرف شده
            elif self.grid[new_x, new_y, new_z] == -1:  # برخورد با موانع
                reward = -10
                done = True
            else:
                self.agent_pos = np.array([new_x, new_y, new_z])

        self.agent_energy -= 1  # کاهش انرژی
        if self.agent_energy <= 0:
            done = True

        return self._get_obs(), reward, done, False, {}

    def _get_obs(self):
        return {
            "agent": self.agent_pos,
            "grid": self.grid
        }

    def render(self, mode="human"):
        if mode == "human":
            print(f"Agent Position: {self.agent_pos}, Energy: {self.agent_energy}")
        elif mode == "rgb_array":
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.set_title("3D Environment")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_xlim(0, self.grid_size)
            ax.set_ylim(0, self.grid_size)
            ax.set_zlim(0, self.grid_size)

            # نمایش منابع انرژی
            for pos in self.resource_positions:
                ax.scatter(pos[0], pos[1], pos[2], color='green', s=100, label='Resource')

            # نمایش موانع
            for pos in self.obstacle_positions:
                ax.scatter(pos[0], pos[1], pos[2], color='black', s=100, label='Obstacle')

            # نمایش ربات
            ax.scatter(self.agent_pos[0], self.agent_pos[1], self.agent_pos[2], color='red', s=100, label='Agent')

            plt.legend()
            plt.show()

# تعریف کلاس ربات با یادگیری تقویتی
class Robot:
    def __init__(self, x, y, z, env, grid_size):
        self.x = x
        self.y = y
        self.z = z
        self.energy = 100
        self.path = [(x, y, z)]  # مسیر حرکت ربات
        self.env = env
        self.grid_size = grid_size
        self.model = PPO("MultiInputPolicy", env, verbose=0)  # استفاده از MultiInputPolicy برای فضای Dict

    def move(self, new_x, new_y, new_z):
        new_x = int(new_x)
        new_y = int(new_y)
        new_z = int(new_z)
        if 0 <= new_x < self.grid_size and 0 <= new_y < self.grid_size and 0 <= new_z < self.grid_size:
            if self.env.grid[new_x, new_y, new_z] != -1:  # بررسی موانع
                self.x = new_x
                self.y = new_y
                self.z = new_z
                self.energy -= 1  # کاهش انرژی با هر حرکت
                self.path.append((new_x, new_y, new_z))  # اضافه کردن موقعیت جدید به مسیر

    def sense_resources(self):
        resources = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dz in [-1, 0, 1]:
                    if 0 <= self.x + dx < self.grid_size and 0 <= self.y + dy < self.grid_size and 0 <= self.z + dz < self.grid_size:
                        if self.env.grid[self.x + dx, self.y + dy, self.z + dz] == 1:
                            resources.append((self.x + dx, self.y + dy, self.z + dz))
        return resources

    def learn_and_move(self):
        action, _ = self.model.predict(self._get_obs())  # استفاده از مدل برای تصمیم‌گیری
        dx, dy, dz = action
        self.move(self.x + int(dx), self.y + int(dy), self.z + int(dz))

    def _get_obs(self):
        return {
            "agent": np.array([self.x, self.y, self.z]),
            "grid": self.env.grid
        }

# ایجاد محیط سه‌بعدی
grid_size = 10
environment = np.zeros((grid_size, grid_size, grid_size))  # محیط خالی سه‌بعدی

# ایجاد ربات‌ها
robots = [
    Robot(np.random.randint(0, grid_size), np.random.randint(0, grid_size), np.random.randint(0, grid_size), CustomGridEnv(grid_size), grid_size)
    for _ in range(2)
]

# شبیه‌سازی
time_steps = 100
for t in range(time_steps):
    print(f"Time step {t}")
    for robot in robots:
        if robot.energy <= 0:
            continue  # ربات انرژی ندارد و نمی‌تواند حرکت کند

        # یادگیری و حرکت
        robot.learn_and_move()

        # تشخیص منابع در نزدیکی
        resources = robot.sense_resources()
        if resources:
            target = resources[0]
            if (robot.x, robot.y, robot.z) == target:
                robot.energy += 20
                environment[target[0], target[1], target[2]] = 0  # منبع مصرف شده

        print(f"Robot at ({robot.x}, {robot.y}, {robot.z}) with energy {robot.energy}")

    # بررسی پایان شبیه‌سازی
    if all(robot.energy <= 0 for robot in robots):
        print("All robots are out of energy!")
        break

# نمایش گراف سه‌بعدی تعاملی
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("3D Robot Movement Visualization", fontsize=16)

# نمایش منابع انرژی
for pos in robots[0].env.resource_positions:
    ax.scatter(pos[0], pos[1], pos[2], color='green', s=100, label='Resource' if pos[0] == robots[0].env.resource_positions[0][0] else "")

# نمایش موانع
for pos in robots[0].env.obstacle_positions:
    ax.scatter(pos[0], pos[1], pos[2], color='black', s=100, label='Obstacle' if pos[0] == robots[0].env.obstacle_positions[0][0] else "")

# نمایش مسیر حرکت ربات‌ها
colors = ['red', 'blue']  # رنگ‌های مختلف برای ربات‌ها
for i, robot in enumerate(robots):
    path = np.array(robot.path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], marker='o', color=colors[i], label=f'Robot {i+1}')
    ax.scatter(robot.path[-1][0], robot.path[-1][1], robot.path[-1][2], marker='X', color=colors[i], s=100)  # موقعیت نهایی

# تنظیمات گراف
ax.set_xlabel("X Coordinate", fontsize=12)
ax.set_ylabel("Y Coordinate", fontsize=12)
ax.set_zlabel("Z Coordinate", fontsize=12)
ax.set_xticks(np.arange(0, grid_size, 1))
ax.set_yticks(np.arange(0, grid_size, 1))
ax.set_zticks(np.arange(0, grid_size, 1))
ax.grid(True)
ax.legend(fontsize=12)

plt.ion()  # فعال کردن حالت تعاملی
plt.show()

# حلقه تعاملی برای نمایش نمودار
try:
    while True:
        plt.pause(0.1)  # به‌روزرسانی نمودار در هر 0.1 ثانیه
except KeyboardInterrupt:
    print("Exiting visualization...")