import tensorflow as tf
import numpy as np

class Judge(tf.keras.Model):
    '''
    Class representing the Judge in R2D2. Adapted from the agent class from https://github.com/shehzaadzd/MINERVA.

    It evaluates the arguments that the agents present and assigns them a score
    that is used to train the agents. Furthermore, it assigns the final prediction score to the whole debate.
    '''

    def __init__(self, params):
        '''
        Initializes the judge.

        :param params: Dict. Parameters of the experiment.
        '''
        super(Judge, self).__init__()
        
        self.path_length = params['path_length']
        self.action_vocab_size = len(params['relation_vocab'])
        self.entity_vocab_size = len(params['entity_vocab'])
        self.embedding_size = params['embedding_size']
        self.hidden_size = params['hidden_size']
        self.train_entities = params['train_entity_embeddings']
        self.train_relations = params['train_relation_embeddings']
        self.hidden_layers = params['layers_judge']
        self.batch_size = params['batch_size'] * (1 + params['false_facts_train']) * params['num_rollouts']
        self.use_entity_embeddings = params['use_entity_embeddings']

        # Initializers
        self.entity_initializer = tf.keras.initializers.GlorotUniform() if params['use_entity_embeddings'] else tf.zeros_initializer()
        
        # Define embeddings
        self.define_embeddings()

    def define_embeddings(self):
        '''
        Creates and adds the embeddings for the KG's relations and entities, as well as assigns operations
        needed when using pre-trained embeddings.
        '''
        self.relation_lookup_table = self.add_weight("relation_lookup_table",
                                                     shape=[self.action_vocab_size, self.embedding_size],
                                                     initializer=tf.keras.initializers.GlorotUniform(),
                                                     trainable=self.train_relations)
        self.entity_lookup_table = self.add_weight("entity_lookup_table",
                                                   shape=[self.entity_vocab_size, self.embedding_size],
                                                   initializer=self.entity_initializer,
                                                   trainable=self.train_entities)

    def set_labels(self, labels):
        '''
        Setter for the labels.
        :param labels: Tensor, [None, 1]. Labels for the episode's query.
        '''
        self.labels = labels

    def action_encoder_judge(self, next_relations, next_entities):
        '''
        Encodes an action into its embedded representation.
        '''
        relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, next_relations)
        entity_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, next_entities)
        
        if self.use_entity_embeddings:
            action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        '''
        Extends an argument by adding the embedded representation of an action.
        '''
        chosen_relation = tf.gather_nd(next_relations, tf.stack([range_arr, action_idx], axis=-1))
        chosen_entities = tf.gather_nd(next_entities, tf.stack([range_arr, action_idx], axis=-1))
        action_embedding = self.action_encoder_judge(chosen_relation, chosen_entities)

        argument = tf.cond(tf.equal(t % self.path_length, 0),
                           lambda: action_embedding,
                           lambda: tf.concat([argument, action_embedding], axis=-1))
        return argument

    def classify_argument(self, argument):
        '''
        Classifies arguments by computing a hidden representation and assigning logits.
        '''
        argument = tf.concat([argument, self.query_relation_embedding, self.query_object_embedding], axis=-1)

        # Reshape argument for the dense layer
        if self.use_entity_embeddings:
            argument = tf.reshape(argument, shape=[-1, self.path_length * 2 * self.embedding_size + 2 * self.embedding_size])
        else:
            argument = tf.reshape(argument, shape=[-1, self.path_length * self.embedding_size + 2 * self.embedding_size])

        hidden = argument
        for i in range(self.hidden_layers - 1):
            hidden = tf.keras.layers.Dense(self.hidden_size, activation='relu', name=f"layer_{i}")(hidden)
        hidden = tf.keras.layers.Dense(self.hidden_size, name=f"layer_{self.hidden_layers - 1}")(hidden)
        
        logits = self.get_logits_argument(hidden)
        return logits, hidden

    def get_logits_argument(self, argument):
        '''
        Assigns logits to an argument.
        '''
        logits = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='classifier_judge_0')(argument)
        logits = tf.keras.layers.Dense(1, name='classifier_judge_1')(logits)
        return logits

    def final_loss(self, rep_argu_list):
        '''
        Computes the final loss and final logits of the debates using all arguments presented.
        '''
        average_argu = tf.reduce_mean(tf.stack(rep_argu_list, axis=-1), axis=-1)
        final_logit = self.get_logits_argument(average_argu)
        final_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=final_logit))
        
        return final_loss, final_logit

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets the judge's query information to the placeholders used by trainer.
        '''
        self.query_subject_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, query_subject)
        self.query_relation_embedding = tf.nn.embedding_lookup(self.relation_lookup_table, query_relation)
        self.query_object_embedding = tf.nn.embedding_lookup(self.entity_lookup_table, query_object)
