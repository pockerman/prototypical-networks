from tqdm import tqdm


class Engine(object):
    """
    Network training engine
    """
    def __init__(self):
        hook_names = ['on_start', 'on_start_epoch', 'on_sample', 'on_forward',
                      'on_backward', 'on_end_epoch', 'on_update', 'on_end']

        self.hooks = {}
        for hook_name in hook_names:
            self.hooks[hook_name] = lambda state: None

        self._state = None

    @property
    def state(self):
        """
        Returns the state of the engine
        """
        return self._state

    def train(self, **kwargs):
        state = {
            'model': kwargs['model'],
            'loader': kwargs['loader'],
            'optim_method': kwargs['optim_method'],
            'optim_config': kwargs['optim_config'],
            'max_epoch': kwargs['max_epoch'],
            'epoch': 0,  # epochs done so far
            't': 0,  # samples seen so far
            'batch': 0,  # samples seen in current epoch
            'stop': False
        }

        state['optimizer'] = state['optim_method'](state['model'].parameters(), **state['optim_config'])

        self.hooks['on_start'](state)
        while state['epoch'] < state['max_epoch'] and not state['stop']:

            # Let the model know that we are training
            # see the thread: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch
            # this does not train the model as it may be implied but
            # it informs the model that it undergoes training
            state['model'].train()

            # actions to perform on the start of an epoch
            self.hooks['on_start_epoch'](state)

            state['epoch_size'] = len(state['loader'])

            for sample in tqdm(state['loader'], desc="Epoch {:d} train".format(state['epoch'] + 1)):

                state['sample'] = sample

                # actions on the sample
                self.hooks['on_sample'](state)

                # zero the optimizer gradient
                # that is zero the parameter gradients
                # gradient buffers had to be manually set to zero using optimizer.zero_grad().
                # because gradients are accumulated in the Backprop step
                state['optimizer'].zero_grad()
                loss, state['output'] = state['model'].loss(state['sample'])
                self.hooks['on_forward'](state)

                # backward propagate
                loss.backward()
                self.hooks['on_backward'](state)

                state['optimizer'].step()

                state['t'] += 1
                state['batch'] += 1
                self.hooks['on_update'](state)

            state['epoch'] += 1
            state['batch'] = 0
            self.hooks['on_end_epoch'](state)

        self.hooks['on_end'](state)
        # finally assign the trained state
        self._state = state
