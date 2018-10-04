# -*- coding: utf-8 -*-
"""lessons_buffer.py: RUDDER lessons buffer for atari games

Author -- Michael Widrich
Contact -- widrich@bioinf.jku.at

"""
import numpy as np
import h5py


class LessonReplayBuffer(object):
    def __init__(self, buffersize, rnd_gen=None, filename='LessonReplayBuffer.hdf5', inmemory=True, block_size=64000,
                 hdf5_dict=None, temperature=1.):
        """Lessons Replay Buffer storing complete game sequences, as described in RUDDER paper;
        
        Each buffer element is a dictionary representing a game or episode.
        
        Following key/value pairs are mandatory:
        'loss': reward redistribution model loss; shape=(1,)
        
        Following additional key/value pairs are mandatory if used with RUDDER example code:
        'states': (possibly stacked) game frames; shape=(batchsize, n_timesteps, x, y, c)
        'actions': actions passed to the environment; shape=(batchsize, n_timesteps, 1)
        'neglogpacs': output values of policy network (negative log-policy-activations); shape=(batchsize, n_timesteps, n_actions)
        'original_rewards': rewards from the environment; shape=(batchsize, n_timesteps, 1)
        'redistributed_reward': rewards redistributed by RUDDER; shape=(batchsize, n_timesteps, 1)
        'rr_quality': quality measure of redistributed reward; shape=(1,)
        'dones': boolean done-flags used to indicate end-of-episode; shape=(batchsize, n_timesteps, 1)
                                                                         
        New custom buffer elements may be added as long as the datatype is supported by hf5py; By default, the
        compressed buffer is stored in memory only until written to disk when saving the model;
        

        Parameters
        ----------
        buffersize : int
            Maximum number of buffer elements to store
        rnd_gen : numpy random generator or None
            If None, creates a new numpy.random.RandomState()
        filename : str
            Filename to store buffer in during usage; Only relevant if inmemory==False;
        inmemory : bool
            Sets h5py's backing_store parameter:
            True: Keep (compressed) h5py file in memory and write to disk only when saving
            False: Keep (compressed) h5py file on disk during usage
        block_size : int
            Sets h5py's block_size parameter
        hdf5_dict : dict ot None
            Dictionary with kwargs for creating h5py datasets; Defaults to lzf compression if None;
        temperature : float
            Temperature to use for softmax on losses for sampling; Higher temperature will lead to more variance in
            sampling, lower temperature will focus buffer elements with higher loss;
        """
        self.buffersize = int(buffersize)
        self.current_entropy = 0
        self.inmemory = inmemory
        if self.inmemory:
            self.bufferfile = h5py.File(filename, 'w-', driver='core', block_size=block_size, backing_store=False)
        else:
            self.bufferfile = h5py.File(filename, 'w-', driver=None, block_size=block_size)
        
        if hdf5_dict is None:
            hdf5_dict = dict(compression="lzf", chunks=True)
        self.hdf5_dict = hdf5_dict
        
        if rnd_gen is None:
            rnd_gen = np.random.RandomState()
        self.rnd_gen = rnd_gen
        self.temperature = float(temperature)
    
    def add_sample(self, sample, id):
        """Add sample to buffer, assign id"""
        id = str(id)
        self.bufferfile.create_group(id)
        for key in sample.keys():
            self.bufferfile[id].create_dataset(key, data=np.asarray(sample[key]), **self.hdf5_dict)
        
    def buffer_from_file(self, filepath):
        """Load buffer from file"""
        readfile = h5py.File(filepath, 'r')
        for group in readfile.keys():
            self.bufferfile.copy(readfile[group], group, name=group)
    
    def buffer_to_file(self, filepath):
        """Write buffer to file"""
        writefile = h5py.File(filepath, 'w-')
        for group in self.bufferfile.keys():
            writefile.copy(self.bufferfile[group], group, name=group)
        writefile.flush()
    
    def consider_adding_sample(self, sample: dict):
        """ Show sample to buffer; Buffer decides whether to add it or not based on sample loss;
        
        Sample must at least contain the key 'loss'; For usage with RUDDER example code, see LessonReplayBuffer class
        docstring;
        """
        if self.get_buffer_len() < self.buffersize:
            # Add sample if buffer is not full
            self.add_sample(sample, id=self.get_buffer_len())
        else:
            # Replace sample with lowest loss in buffer if new sample loss is higher
            keys_losses = self.get_losses()
            low_sample_ind = np.argmin([b[1] for b in keys_losses])
            if sample['loss'] > keys_losses[low_sample_ind][1]:
                id = np.copy(keys_losses[low_sample_ind][0])
                self.del_sample(id)
                self.add_sample(sample, id=id)
    
    def del_sample(self, id):
        """Delete sample by id from buffer"""
        id = str(id)
        del self.bufferfile[id]
    
    def get_buffer_len(self):
        """Get number of current buffer elements"""
        return len(self.bufferfile.keys())
    
    def get_losses(self):
        """Get all losses in buffer"""
        keys_losses = [(k, self.bufferfile[k]["loss"][0]) for k in self.bufferfile.keys()]
        return keys_losses
    
    def get_sample(self):
        """Randomly sample episode or game with from buffer; Sampling probabilities are softmax values of losses in
        buffer;"""
        keys_losses = self.get_losses()
        losses = [b[1] for b in keys_losses]
        ids = [b[0] for b in keys_losses]
        
        probs = self.softmax(losses)
        sample_id = self.rnd_gen.choice(ids, p=probs)
        sample = self.bufferfile[str(sample_id)]
        sample = dict(((k, np.copy(sample[k][:])) for k in sample.keys()))
        sample['id'] = sample_id
        return sample
    
    def softmax(self, logits):
        """Softmax used for sampling from buffer"""
        e_logits = np.exp((logits - np.max(logits)) / self.temperature)
        return e_logits / e_logits.sum()
    
    def update_sample_field(self, key, values, id):
        """Update value of key of sample id"""
        id = str(id)
        self.bufferfile[id][key][:] = values
    
    def update_sample_loss(self, loss, id):
        """Update loss of sample id"""
        id = str(id)
        self.bufferfile[id]['loss'][0] = loss