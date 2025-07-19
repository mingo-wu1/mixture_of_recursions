from torch.optim.lr_scheduler import LambdaLR

def get_trapezoid_scheduler(optimizer, num_training_steps, warmup_percentage, cooldown_percentage):
    """Trapezoid learning rate scheduler from paper."""
    num_warmup_steps = int(num_training_steps * warmup_percentage)
    num_cooldown_steps = int(num_training_steps * cooldown_percentage)
    num_stable_steps = num_training_steps - num_warmup_steps - num_cooldown_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # Warmup phase
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < num_warmup_steps + num_stable_steps:
            # Stable phase
            return 1.0
        else:
            # Cooldown phase
            steps_into_cooldown = current_step - (num_warmup_steps + num_stable_steps)
            return max(0.0, 1.0 - float(steps_into_cooldown) / float(max(1, num_cooldown_steps)))

    return LambdaLR(optimizer, lr_lambda)