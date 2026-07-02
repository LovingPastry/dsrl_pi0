import argparse
import sys
from examples.train_sim import main
from jaxrl2.utils.launch_util import parse_training_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', default=42, help='Random seed.', type=int)
    parser.add_argument('--launch_group_id', default='', help='group id used to group runs in logging.')
    parser.add_argument('--eval_episodes', default=10,help='Number of episodes used for evaluation.', type=int)
    parser.add_argument('--env', default='libero', help='name of environment')
    parser.add_argument('--log_interval', default=1000, help='Logging interval.', type=int)
    parser.add_argument('--eval_interval', default=5000, help='Eval interval.', type=int)
    parser.add_argument('--checkpoint_interval', default=-1, help='checkpoint interval.', type=int)
    parser.add_argument('--batch_size', default=16, help='Mini batch size.', type=int)
    parser.add_argument('--max_steps', default=int(1e6), help='Number of training steps.', type=int)
    parser.add_argument('--add_states', default=1, help='whether to add low-dim states to the obervations', type=int)
    parser.add_argument('--tb_project', default='cql_sim_online', help='tensorboard project / run log subdir name')
    parser.add_argument('--start_online_updates', default=1000, help='number of steps to collect before starting online updates', type=int)
    parser.add_argument('--algorithm', default='pixel_sac', help='type of algorithm')
    parser.add_argument('--prefix', default='', help='prefix to use for the run name / logging')
    parser.add_argument('--suffix', default='', help='suffix to use for the run name / logging')
    parser.add_argument('--multi_grad_step', default=1, help='Number of graident steps to take per environment step, aka UTD', type=int)
    parser.add_argument('--resize_image', default=-1, help='the size of image if need resizing', type=int)
    parser.add_argument('--query_freq', default=-1, help='query frequency', type=int)
    parser.add_argument('--task_suite', default='libero_90', help='LIBERO task suite (e.g. libero_90, libero_10, libero_goal)', type=str)
    parser.add_argument('--task_id', default=57, help='task index within the LIBERO suite', type=int)
    parser.add_argument('--warmup_trajs', default=-1, help='collect exactly this many Gaussian-noise warmup trajs before updates (-1 = legacy start_online_updates threshold)', type=int)
    parser.add_argument('--dual_buffer', default=0, help='1 = frozen warmup buffer + growing online buffer, 50/50 batches', type=int)
    parser.add_argument('--early_stop_success', default=1, help='1 = stop as soon as an eval hits 100%% success', type=int)
    parser.add_argument('--vlm_dim', default=2048, help='dim of the pi0 PaliGemma prefix feature (obs_mode=vlm)', type=int)
    parser.add_argument('--qa_critic_lr', default=3e-4, help='learning rate of the DSRL-NA action-space critic', type=float)

    train_args_dict = dict(
        actor_lr=1e-4,
        critic_lr= 3e-4,
        temp_lr=3e-4,
        hidden_dims= (128, 128, 128),
        cnn_features= (32, 32, 32, 32),
        cnn_strides= (2, 1, 1, 1),
        cnn_padding= 'VALID',
        latent_dim= 50,
        discount= 0.999,
        tau= 0.005,
        critic_reduction = 'mean',
        dropout_rate=0.0,
        aug_next=1,
        use_bottleneck=True,
        encoder_type='small',
        encoder_norm='group',
        use_spatial_softmax=True,
        softmax_temperature=-1,
        target_entropy='auto',
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        obs_mode='pixels',
        )

    variant, args = parse_training_args(train_args_dict, parser)
    print(variant)
    main(variant)
    sys.exit()
    