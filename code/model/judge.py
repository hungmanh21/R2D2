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

    def extend_argument(self, argume
