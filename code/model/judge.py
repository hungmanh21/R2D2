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

        # Initialize embeddings
        self.initialize_embeddings()
        
        # Initialize dense layers
        self.hidden_layers_list = []
        for i in range(self.hidden_layers - 1):
            self.hidden_layers_list.append(
                tf.keras.layers.Dense(self.hidden_size, activation='relu', name=f"layer_{i}")
            )
        self.hidden_layers_list.append(
            tf.keras.layers.Dense(self.hidden_size, name=f"layer_{self.hidden_layers - 1}")
        )
        
        # Classifier layers
        self.classifier_1 = tf.keras.layers.Dense(self.hidden_size, activation='relu', name='classifier_judge_0')
        self.classifier_2 = tf.keras.layers.Dense(1, name='classifier_judge_1')

    def initialize_embeddings(self):
        '''
        Creates embeddings for the KG's relations and entities
        '''
        # Relation embeddings
        self.relation_lookup_table = tf.keras.layers.Embedding(
            input_dim=self.action_vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer='glorot_uniform',
            trainable=self.train_relations,
            name="relation_lookup_table"
        )

        # Entity embeddings
        entity_initializer = 'glorot_uniform' if self.use_entity_embeddings else 'zeros'
        self.entity_lookup_table = tf.keras.layers.Embedding(
            input_dim=self.entity_vocab_size,
            output_dim=self.embedding_size,
            embeddings_initializer=entity_initializer,
            trainable=self.train_entities,
            name="entity_lookup_table"
        )

    def set_embeddings(self, relation_embeddings, entity_embeddings):
        '''
        Sets pre-trained embeddings
        
        :param relation_embeddings: numpy array of shape [action_vocab_size, embedding_size]
        :param entity_embeddings: numpy array of shape [entity_vocab_size, embedding_size]
        '''
        self.relation_lookup_table.set_weights([relation_embeddings])
        self.entity_lookup_table.set_weights([entity_embeddings])

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
        relation_embedding = self.relation_lookup_table(next_relations)
        entity_embedding = self.entity_lookup_table(next_entities)
        
        if self.use_entity_embeddings:
            action_embedding = tf.concat([relation_embedding, entity_embedding], axis=-1)
        else:
            action_embedding = relation_embedding

        return action_embedding

    @tf.function
    def extend_argument(self, argument, t, action_idx, next_relations, next_entities, range_arr):
        '''
        Extends an argument by adding the embedded representation of an action.
        '''
        indices = tf.stack([range_arr, action_idx], axis=-1)
        chosen_relation = tf.gather_nd(next_relations, indices)
        chosen_entities = tf.gather_nd(next_entities, indices)
        action_embedding = self.action_encoder_judge(chosen_relation, chosen_entities)

        def concat_embedding():
            return tf.concat([argument, action_embedding], axis=-1)
        
        def return_embedding():
            return action_embedding

        return tf.cond(tf.equal(t % self.path_length, 0),
                      return_embedding,
                      concat_embedding)

    def classify_argument(self, argument):
        '''
        Classifies arguments by computing a hidden representation and assigning logits.
        '''
        argument = tf.concat([argument, self.query_relation_embedding, self.query_object_embedding], axis=-1)

        # Reshape argument for the dense layer
        if self.use_entity_embeddings:
            argument = tf.reshape(argument, [-1, self.path_length * 2 * self.embedding_size + 2 * self.embedding_size])
        else:
            argument = tf.reshape(argument, [-1, self.path_length * self.embedding_size + 2 * self.embedding_size])

        # Pass through hidden layers
        hidden = argument
        for layer in self.hidden_layers_list:
            hidden = layer(hidden)
        
        logits = self.get_logits_argument(hidden)
        return logits, hidden

    def get_logits_argument(self, argument):
        '''
        Assigns logits to an argument.
        '''
        logits = self.classifier_1(argument)
        logits = self.classifier_2(logits)
        return logits

    def final_loss(self, rep_argu_list):
        '''
        Computes the final loss and final logits of the debates using all arguments presented.
        '''
        average_argu = tf.reduce_mean(tf.stack(rep_argu_list, axis=-1), axis=-1)
        final_logit = self.get_logits_argument(average_argu)
        final_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(self.labels, final_logit, from_logits=True)
        )
        
        return final_loss, final_logit

    def set_query_embeddings(self, query_subject, query_relation, query_object):
        '''
        Sets the judge's query information.
        '''
        self.query_subject_embedding = self.entity_lookup_table(query_subject)
        self.query_relation_embedding = self.relation_lookup_table(query_relation)
        self.query_object_embedding = self.entity_lookup_table(query_object)
