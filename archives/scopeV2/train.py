from archives.scopeV2.parseConfig import parseConfig
from archives.scopeV2.trainer import Trainer


def train(sys_argv=None):
    config = parseConfig(sys_argv)
    Trainer(config).main()


if __name__ == '__main__':
    train()
