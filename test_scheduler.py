import torch
import unittest
from scheduler import get_trapezoid_scheduler

class TestScheduler(unittest.TestCase):

    def test_trapezoid_scheduler(self):
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)
        num_training_steps = 100
        warmup_percentage = 0.1
        cooldown_percentage = 0.2
        
        scheduler = get_trapezoid_scheduler(optimizer, num_training_steps, warmup_percentage, cooldown_percentage)

        lrs = []
        for _ in range(num_training_steps):
            lrs.append(scheduler.get_last_lr()[0])
            optimizer.step()
            scheduler.step()

        num_warmup_steps = int(num_training_steps * warmup_percentage)  # 10
        num_cooldown_steps = int(num_training_steps * cooldown_percentage)  # 20
        num_stable_steps = num_training_steps - num_warmup_steps - num_cooldown_steps  # 70

        print(f"Warmup steps: {num_warmup_steps}")
        print(f"Stable steps: {num_stable_steps}")
        print(f"Cooldown steps: {num_cooldown_steps}")
        print(f"LR at step {num_warmup_steps-1}: {lrs[num_warmup_steps-1]}")
        print(f"LR at step {num_warmup_steps}: {lrs[num_warmup_steps]}")

        # 修复：warmup结束时应该接近1.0，但可能不完全等于1.0
        # 因为 (step) / max(1, num_warmup_steps) 当step=num_warmup_steps-1时 = 9/10 = 0.9
        self.assertAlmostEqual(lrs[0], 0.0, places=6)  # 开始
        self.assertGreater(lrs[num_warmup_steps-1], 0.8)  # warmup快结束了
        self.assertAlmostEqual(lrs[num_warmup_steps], 1.0, places=6)  # 进入stable阶段
        self.assertAlmostEqual(lrs[num_warmup_steps + num_stable_steps - 1], 1.0, places=6)  # stable阶段结束
        self.assertLess(lrs[-1], 0.1)  # cooldown结束

if __name__ == '__main__':
    unittest.main()
