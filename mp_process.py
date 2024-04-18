import gym
def main2():
    """
    第二种实现多进程训练的思路：
        1.多个进程不训练网络，只是拿到主进程的网络后去探索环境，并将transition通过pipe传回主进程
        2.主进程将所有子进程的transition打包为一个buffer后供网络训练
        3.将更新后的net再传到子进程，回到1
    """
    env = gym.make('LunarLanderContinuous-v2')
    net = GlobalNet(env.observation_space.shape[0], env.action_space.shape[0])
    ppo = AgentPPO(deepcopy(net))
    process_num = 4
    pipe_dict = dict((i, (pipe1, pipe2)) for i in range(process_num) for pipe1, pipe2 in (multiprocessing.Pipe(),))
    child_process_list = []
    for i in range(process_num):
        pro = multiprocessing.Process(target=child_process2, args=(pipe_dict[i][1],))
        child_process_list.append(pro)
    [p.start() for p in child_process_list]

    rewardList = list()
    MAX_EPISODE = 30
    batch_size = 128
    for episode in range(MAX_EPISODE):
        [pipe_dict[i][0].send(net) for i in range(process_num)]
        reward = 0
        buffer_list = list()
        for i in range(process_num):
            receive = pipe_dict[i][0].recv()        # 这句带同步子进程的功能，收不到子进程的数据就都不会走到for之后的语句
            data = receive[0]
            buffer_list.append(data)
            reward += receive[1]
        ppo.update_policy_mp(batch_size,8,buffer_list)
        net.act.load_state_dict(ppo.act.state_dict())
        net.cri.load_state_dict(ppo.cri.state_dict())

        reward /= process_num
        rewardList.append(reward)
        print(f'episode:{episode}  reward:{reward}')

    [p.terminate() for p in child_process_list]
    painter = Painter(load_csv=True, load_dir='../figure.csv')
    painter.addData(rewardList, 'MP-PPO-Mod')
    painter.saveData('../figure.csv')
    painter.drawFigure()


def child_process2(pipe):
    env = gym.make('LunarLanderContinuous-v2')
    while True:
        net = pipe.recv()  # 收主线程的net参数，这句也有同步的功能
        ppo = AgentPPO(net)
        rewards, steps = ppo.update_buffer(env, 5000, 1)
        transition = ppo.buffer.sample_all()
        r = transition.reward
        m = transition.mask
        a = transition.action
        s = transition.state
        log = transition.log_prob
        data = (r,m,s,a,log)
        """pipe不能直接传输buffer回主进程，可能是buffer内有transition，因此将数据取出来打包回传"""
        pipe.send((data,rewards))
