import numpy as np
import tensorflow as tf


class Agent(tf.Module):
    '''
    Class for the agents in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.
    A single instance of Agent contains both the pro and con agent.
    '''

    def __init__(self, params, judge):
        '''
        Initializes the agents.
        :param params: Dict. Parameters of the experiment.
        :param judge: Judge. Instance of Judge that the agents present arguments to.
        '''
        super().__init__(name='Agent')
        
        self.judge = judge
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.ePAD = tf.constant(params['entity_vocab']['PAD'], dtype=tf.int32)
        self.rPAD = tf.constant(params['relation_vocab']['PAD'], dtype=tf.int32)
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.test_rollouts = params['test_rollouts']
        self.path_length = params['path_length']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']
        self.dummy_start_label = tf.constant(
            np.ones(self.batch_size, dtype='int64') * params['relation_vocab']['DUMMY_START_RELATION'])

        self.hidden_layers = params['layers_agent']
        self.custom_baseline = params['custom_baseline']
        self.use_entity_embeddings = params['use_entity_embeddings']
        if self.use_entity_embeddings:
            self.entity_initializer = tf.keras.initializers.GlorotUniform()
            self.m = 2
        else:
            self.m = 1
            self.entity_initializer = tf.zeros_initializer()

        self.define_embeddings()
        self.define_agents_policy()

    def define_embeddings(self):
        '''
        For both agents, creates and adds the embeddings for the KG's relations and entities to the graph.
        '''
        # Agent 1 embeddings
        self.relation_lookup_table_agent_1 = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[self.action_vocab_size, self.embedding_size]),
            trainable=self.train_relations,
            name="relation_lookup_table_agent_1"
        )
        
        self.entity_lookup_table_agent_1 = tf.Variable(
            self.entity_initializer(shape=[self.entity_vocab_size, self.embedding_size]),
            trainable=self.train_entities,
            name="entity_lookup_table_agent_1"
        )

        # Agent 2 embeddings
        self.relation_lookup_table_agent_2 = tf.Variable(
            tf.keras.initializers.GlorotUniform()(shape=[self.action_vocab_size, self.embedding_size]),
            trainable=self.train_relations,
            name="relation_lookup_table_agent_2"
        )
        
        self.entity_lookup_table_agent_2 = tf.Variable(
            self.entity_initializer(shape=[self.entity_vocab_size, self.embedding_size]),
            trainable=self.train_entities,
            name="entity_lookup_table_agent_2"
        )

    def define_agents_policy(self):
        '''
        Defines the agents' policy using TF 2.x RNN cells.
        '''
        # Agent 1 policy
        cells_1 = []
        for _ in range(self.hidden_layers):
            cells_1.append(
                tf.keras.layers.LSTMCell(self.m * self.embedding_size, use_bias=True))
        self.policy_agent_1 = tf.keras.layers.RNN(cells_1, return_state=True)

        # Agent 2 policy
        cells_2 = []
        for _ in range(self.hidden_layers):
            cells_2.append(
                tf.keras.layers.LSTMCell(self.m * self.embedding_size, use_bias=True))
        self.policy_agent_2 = tf.keras.layers.RNN(cells_2, return_state=True)

    def format_state(self, state):
        '''
        Formats the cell- and hidden-state of the LSTM.
        :param state: Tensor, [hidden_layers_agent, 2, Batch_size, embedding_size * m]
        :return: List of list of two tensors
        '''
        return tf.nest.map_structure(lambda x: tf.unstack(x, 2), tf.unstack(state, self.hidden_layers))

    def get_mem_shape(self):
        '''Returns the shape of the agent's LSTMCell.'''
        return (self.hidden_layers, 2, None, self.m * self.embedding_size)

    def get_init_state_array(self, temp_batch_size):
        '''Returns initial state arrays for both agents.'''
        mem_agent = self.get_mem_shape()
        agent_mem_1 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        agent_mem_2 = np.zeros(
            (mem_agent[0], mem_agent[1], temp_batch_size * self.test_rollouts, mem_agent[3])).astype('float32')
        return agent_mem_1, agent_mem_2

    @tf.function
    def policy(self, input_action, which_agent):
        '''
        Applies the policy for the specified agent.
        '''
        def policy_1():
            output, states = self.policy_agent_1(tf.expand_dims(input_action, 1), 
                                               initial_state=self.state_agent_1)
            return output[:, 0, :], states

        def policy_2():
            output, states = self.policy_agent_2(tf.expand_dims(input_action, 1), 
                                               initial_state=self.state_agent_2)
            return output[:, 0, :], states

        if which_agent == 0:
            output, new_state = policy_1()
        else:
            output, new_state = policy_2()

        new_state_stacked = tf.stack(new_state)
        state_agent_1_stacked = tf.stack(self.state_agent_1)
        state_agent_2_stacked = tf.stack(self.state_agent_2)

        # Update states conditionally
        self.state_agent_1 = self.format_state((1-which_agent)*new_state_stacked + which_agent*state_agent_1_stacked)
        self.state_agent_2 = self.format_state(which_agent*new_state_stacked + (1-which_agent)*state_agent_2_stacked)

        return output

    @tf.function
    def action_encoder_agent(self, next_relations, current_entities, which_agent):
        '''Encodes actions for the specified agent.'''
        if which_agent == 0:
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, current_entities)
        else:
            relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, next_relations)
            entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, current_entities)

        if self.use_entity_embeddings:
            return tf.concat([relation_embedding, entity_embedding], axis=-1)
        return relation_embedding

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''Sets query embeddings for both agents.'''
        self.query_subject_embedding_agent_1 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, query_subject)
        self.query_relation_embedding_agent_1 = tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, query_relation)
        self.query_object_embedding_agent_1 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, query_object)

        self.query_subject_embedding_agent_2 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, query_subject)
        self.query_relation_embedding_agent_2 = tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, query_relation)
        self.query_object_embedding_agent_2 = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, query_object)

    @tf.function
    def step(self, next_relations, next_entities, prev_state_agent_1,
             prev_state_agent_2, prev_relation, current_entities, range_arr, which_agent, random_flag):
        '''Computes a step for an agent during the debate.'''
        self.state_agent_1 = prev_state_agent_1
        self.state_agent_2 = prev_state_agent_2

        # Get state vector
        if which_agent == 0:
            prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table_agent_1, current_entities)
            prev_relation_emb = tf.nn.embedding_lookup(self.relation_lookup_table_agent_1, prev_relation)
        else:
            prev_entity = tf.nn.embedding_lookup(self.entity_lookup_table_agent_2, current_entities)
            prev_relation_emb = tf.nn.embedding_lookup(self.relation_lookup_table_agent_2, prev_relation)

        if self.use_entity_embeddings:
            state = tf.concat([prev_relation_emb, prev_entity], axis=-1)
        else:
            state = prev_relation_emb

        # Get query embeddings based on agent
        if which_agent == 0:
            query_subject_embedding = self.query_subject_embedding_agent_1
            query_relation_embedding = self.query_relation_embedding_agent_1
            query_object_embedding = self.query_object_embedding_agent_1
        else:
            query_subject_embedding = self.query_subject_embedding_agent_2
            query_relation_embedding = self.query_relation_embedding_agent_2
            query_object_embedding = self.query_object_embedding_agent_2

        state_query_concat = tf.concat([state, query_subject_embedding, 
                                      query_relation_embedding, query_object_embedding], axis=-1)

        # Get action embeddings and compute scores
        candidate_action_embeddings = self.action_encoder_agent(next_relations, next_entities, which_agent)
        output = self.policy(state_query_concat, which_agent)
        output_expanded = tf.expand_dims(output, axis=1)
        prelim_scores = tf.reduce_sum(tf.multiply(candidate_action_embeddings, output_expanded), axis=2)

        # Mask PAD actions
        mask = tf.equal(next_relations, self.rPAD)
        dummy_scores = tf.ones_like(prelim_scores) * -99999.0
        scores = tf.where(mask, dummy_scores, prelim_scores)
        uni_scores = tf.where(mask, dummy_scores, tf.ones_like(prelim_scores))

        # Sample action
        if random_flag:
            action = tf.cast(tf.random.categorical(logits=uni_scores, num_samples=1), tf.int32)
        else:
            action = tf.cast(tf.random.categorical(logits=scores, num_samples=1), tf.int32)

        label_action = tf.squeeze(action, axis=1)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=scores, labels=label_action)

        action_idx = tf.squeeze(action)
        indices = tf.stack([range_arr, action_idx], axis=1)
        chosen_relation = tf.gather_nd(next_relations, indices)

        return loss, self.state_agent_1, self.state_agent_2, tf.nn.log_softmax(scores), action_idx, chosen_relation

    @tf.function
    def __call__(self, which_agent, candidate_relation_sequence, candidate_entity_sequence, 
                 current_entities, range_arr, T=3, random_flag=None):
        '''Main call method for conducting a debate.'''
        prev_relation = self.dummy_start_label
        argument = self.judge.action_encoder_judge(prev_relation, prev_relation)

        all_loss = []
        all_logits = []
        action_idx = []
        all_temp_logits_judge = []
        arguments_representations = []
        all_rewards_agents = []
        all_rewards_before_baseline = []

        # Initialize agent states
        prev_state_agent_1 = self.policy_agent_1.get_initial_state(batch_size=self.batch_size)
        prev_state_agent_2 = self.policy_agent_2.get_initial_state(batch_size=self.batch_size)

        for t in range(T):
            next_possible_relations = candidate_relation_sequence[t]
            next_possible_entities = candidate_entity_sequence[t]
            current_entities_t = current_entities[t]
            which_agent_t = which_agent[t]

            loss, prev_state_agent_1, prev_state_agent_2, logits, idx, chosen_relation = \
                self.step(next_possible_relations, next_possible_entities,
                         prev_state_agent_1, prev_state_agent_2, prev_relation,
                         current_entities_t, range_arr=range_arr,
                         which_agent=which_agent_t, random_flag=random_flag)

            all_loss.append(loss)
            all_logits.append(logits)
            action_idx.append(idx)
            prev_relation = chosen_relation

            argument = self.judge.extend_argument(argument, tf.constant(t, dtype=tf.int32), 
                                                idx, candidate_relation_sequence[t], 
                                                candidate_entity_sequence[t], range_arr)

            # Process rewards and logits based on argument completion
            if t % self.path_length != (self.path_length - 1):
                all_temp_logits_judge.append(tf.zeros([self.batch_size, 1]))
                temp_rewards = tf.zeros([self.batch_size, 1])
                all_rewards_before_baseline.append(temp_rewards)
                all_rewards_agents.append(temp_rewards)
            else:
                logits_judge, rep_argu = self.judge.classify_argument(argument)
                rewards = tf.nn.sigmoid(logits_judge)
                all_temp_logits_judge.append(logits_judge)
                arguments_representations.append(rep_argu)
                all_rewards_before_baseline.append(rewards)

                if self.custom_baseline:
                    no_op_arg = self.judge.action_encoder_judge(prev_relation, prev_relation)
                    for i in range(self.path_length):
                        no_op_arg = self.judge.extend_argument(no_op_arg, tf.constant(i, dtype=tf.int32),
                                                            tf.zeros_like(
                                                                idx), candidate_relation_sequence[0],
                                                            candidate_entity_sequence[0], range_arr)
                    no_op_logits, rep_argu = self.judge.classify_argument(no_op_arg)
                    rewards_no_op = tf.nn.sigmoid(no_op_logits)
                    all_rewards_agents.append(rewards - rewards_no_op)


                else:
                    all_rewards_agents.append(rewards)

        loss_judge, final_logit_judge = self.judge.final_loss(
            arguments_representations)

        return loss_judge, final_logit_judge, all_temp_logits_judge, all_loss, all_logits, action_idx, \
            all_rewards_agents, all_rewards_before_baseline
