
def get_qmix_args(args):
    args.n_actions = 9
    args.n_agents = 12
    args.state_shape = 481
    args.obs_shape = 481
    args.last_action = True
    args.reuse_network = True
    args.rnn_hidden_dim = 64
    args.qmix_hidden_dim = 32
    args.model_dir = "./models/qmix"
    args.date_dir = "20191021"
    args.optimizer = "RMS"
    args.lr = 5e-4
    args.epsilon = 0.9
    args.anneal_epsilon = 0.00034
    args.min_epsilon = 0.02
    args.gamma = 0.99
    args.n_epoch = 100000
    args.n_episodes = 1
    args.episode_limit = 10
    args.step_control = 400
    args.buffer_size = 8000
    args.batch_size = 64
    args.train_steps = 2
    args.target_update_cycle = 200
    args.save_cycle = 1
    args.start_train = 20
    args.evaluate_cycle = 200
    args.evaluate_epoch = 10
    args.threshold = 0.75
    return args
