import torch

class CNNSharedNet(nn.Module):
    def __init__(self, observation_space, hidden_sizes):
        super(CNNSharedNet, self).__init__()
        pretrained_CNN = 'resnet'+str(hidden_sizes[0])
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', pretrained_CNN, pretrained=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        if observation_space.shape[0] == 1:
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, hidden_sizes[1])

    def forward(self, x):
        return self.resnet(x)

class CNNActorCritic(nn.Module):
    def __init__(self, observation_space, action_space,
                 hidden_sizes=(18,64,64), activation=nn.Tanh):
        super().__init__()
        # shared network
        self.shared = CNNSharedNet(observation_space, hidden_sizes)
        hidden_sizes.pop(0)
        dummy_obs_dim_to_be_replaced = 1
        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(dummy_obs_dim_to_be_replaced, action_space.shape[0], hidden_sizes, activation)
            self.pi.mu_net[0] = self.shared
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(dummy_obs_dim_to_be_replaced, action_space.n, hidden_sizes, activation)
            self.pi.logits_net[0] = self.shared
        # build value function
        self.v  = MLPCritic(dummy_obs_dim_to_be_replaced, hidden_sizes, activation)
        self.v.v_net[0] = self.shared

    def step(self, obs):
        obs = obs
        with torch.no_grad():
            pi = self.pi._distribution(obs.unsqueeze(0))
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            a = a[0]
            v = self.v(obs.unsqueeze(0))
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

