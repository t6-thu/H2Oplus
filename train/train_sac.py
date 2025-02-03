import wandb
import argparse
from algos.SAC import SAC


def main():
    wandb.init(project="when-to-trust-your-simulator", entity="t6-thu")

    # Parameters
    parser = argparse.ArgumentParser(description='Solve HalfCheetah-v2 with SAC')
    parser.add_argument(
        '--lr_actor', type=float, default=3e-4, metavar='lr_actor', help='discount factor (default: 1e-5)')
    parser.add_argument(
        '--lr_critic', type=float, default=3e-4, metavar='lr_critic', help='discount factor (default: 1e-3)')
    parser.add_argument(
        '--batch_size', type=int, default=256, metavar='BS', help='batch_size (default: 256)')
    parser.add_argument(
        '--warmup_steps', type=int, default=1000000, metavar='warmup_steps', help='warmup_steps (default: 30000)')
    parser.add_argument(
        '--num_hidden', type=int, default=256, metavar='hidden', help='num_hidden (default: 256)')
    parser.add_argument(
        '--threshold', type=float, default=8, metavar='ε', help='threshold (default: 2)')
    parser.add_argument(
        '--alpha', type=float, default=0.2, metavar='alpha', help='alpha (default: 100)')
    parser.add_argument('--auto_alpha', default=True, help='auto-update the hyper-parameter alpha (default: True)')
    parser.add_argument('--device', default='cuda', help='cuda or cpu')
    parser.add_argument('--importance', default=False)
    parser.add_argument('--env_name', help='choose your mujoco env')
    args = parser.parse_args()
    wandb.config.update(args)

    # setup mujoco environment and SBAC agent
    env_name = args.env_name
    agent_SBAC = SBAC(env_name=env_name,
                    num_hidden=args.num_hidden,
                    lr_actor=args.lr_actor,
                    lr_critic=args.lr_critic,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    auto_alpha=args.auto_alpha,
                    epsilon=args.threshold,
                    alpha=args.alpha,
                    device=args.device,
                    Use_W=args.importance
                    )

    agent_SBAC.learn()


if __name__ == '__main__':
    main()