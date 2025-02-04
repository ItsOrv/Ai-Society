import numpy as np

# تعریف محیط
grid_size = 10
environment = np.zeros((grid_size, grid_size))  # محیط خالی

# قرار دادن منابع انرژی به صورت تصادفی
num_resources = 5
resource_positions = np.random.randint(0, grid_size, (num_resources, 2))
for pos in resource_positions:
    environment[pos[0], pos[1]] = 1  # 1 نشان‌دهنده منبع انرژی است

# تعریف ربات‌ها
class Robot:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.energy = 100

    def move(self, new_x, new_y):
        if 0 <= new_x < grid_size and 0 <= new_y < grid_size:
            self.x = new_x
            self.y = new_y
            self.energy -= 1  # کاهش انرژی با هر حرکت

    def sense_resources(self, environment):
        resources = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if 0 <= self.x + dx < grid_size and 0 <= self.y + dy < grid_size:
                    if environment[self.x + dx, self.y + dy] == 1:
                        resources.append((self.x + dx, self.y + dy))
        return resources

# ایجاد ربات‌ها
robots = [Robot(np.random.randint(0, grid_size), np.random.randint(0, grid_size)) for _ in range(2)]

# شبیه‌سازی
time_steps = 100
for t in range(time_steps):
    print(f"Time step {t}")
    for robot in robots:
        if robot.energy <= 0:
            continue  # ربات انرژی ندارد و نمی‌تواند حرکت کند

        # تشخیص منابع در نزدیکی
        resources = robot.sense_resources(environment)
        if resources:
            target = resources[0]  # به سمت اولین منبع حرکت کند
            dx = target[0] - robot.x
            dy = target[1] - robot.y
            if dx != 0:
                robot.move(robot.x + np.sign(dx), robot.y)
            elif dy != 0:
                robot.move(robot.x, robot.y + np.sign(dy))

            # اگر به منبع رسید، انرژی افزایش یابد
            if (robot.x, robot.y) == target:
                robot.energy += 20
                environment[target[0], target[1]] = 0  # منبع مصرف شده است

        print(f"Robot at ({robot.x}, {robot.y}) with energy {robot.energy}")

    # بررسی پایان شبیه‌سازی
    if all(robot.energy <= 0 for robot in robots):
        print("All robots are out of energy!")
        break