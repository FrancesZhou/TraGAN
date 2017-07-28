import numpy as np
import tensorflow as tf
import random
from tra_preprocessing import get_all_data, get_all_data2
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT
#from target_lstm import TARGET_LSTM
#import cPickle

#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
EMB_DIM = 32 # embedding dimension
HIDDEN_DIM = 32 # hidden state dimension of lstm cell
PRE_LENGTH = 10
SEQ_LENGTH = 20 # sequence length
START_TOKEN = 0
PRE_EPOCH_NUM = 2 # supervise (maximum likelihood estimation) epochs
SEED = 88
BATCH_SIZE = 64

#########################################################################################
#  Discriminator  Hyper-parameters
#########################################################################################
dis_embedding_dim = 64
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
dis_dropout_keep_prob = 0.75
dis_l2_reg_lambda = 0.2
dis_batch_size = 64

#########################################################################################
#  Basic Training Parameters
#########################################################################################
TOTAL_BATCH = 200
dis_train_num = 10000
# positive_file = 'save/real_data.txt'
# negative_file = 'save/generator_sample.txt'
# eval_file = 'save/eval_file.txt'
# generated_num = 10000


def generate_samples(sess, trainable_model, batch_size, generated_num, gen_loader):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        x = gen_loader.next_batch()
        generated_samples.extend(trainable_model.generate(sess, x))
    return generated_samples


def target_loss(sess, gen_lstm, data_loader):
    # target_loss means the positive log-test tested with the generative model "lstm"
    nll = []
    data_loader.reset_pointer()
    for it in xrange(data_loader.num_batch):
        batch = data_loader.next_batch()
        g_loss = sess.run(gen_lstm.pretrain_loss, {gen_lstm.x: batch})
        nll.append(g_loss)
    return np.mean(nll)


def pre_train_epoch(sess, trainable_model, data_loader):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []
    data_loader.reset_pointer()
    for it in xrange(data_loader.num_batch):
        if it%1000 == 0:
            print 'pre_train_epoch: '+str(it)
        batch = data_loader.next_batch()
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    assert START_TOKEN == 0
    #get_data = get_all_data2('TraData/BeijingTraData/', 'TraData/BJseq/', 20)
    get_data = get_all_data2('TraData/BeijingTraData/', 'TraData/BJseq2/', 20)
    #get_data.process_all(116.0, 116.8, 39.6, 40.2, 0.01)
    pre_train, train, test = get_data.get_train_test_data()
    pre_train = pre_train[:100000] + 1
    train = train[:100000] + 1
    test = test[:100000] + 1
    train = pre_train
    # pre_train, train, test = get_data.process_all() # just as follows
    #get_data.create_sequences()
    #get_data.create_sequences_bound(115.5, 117.5, 39.5, 41, 0.01)
    #get_data.create_sequences_grid(115.5, 117.5, 39.5, 41, 0.01)
    #get_data.create_grid_seq(115.5, 117.5, 39.5, 41, 0.01)
    
    pre_train_data_loader = Gen_Data_loader(BATCH_SIZE)
    train_data_loader = Gen_Data_loader(BATCH_SIZE)
    test_data_loader = Gen_Data_loader(BATCH_SIZE) # For testing
    vocab_size = 80*60+1
    dis_data_loader = Dis_dataloader(BATCH_SIZE)

    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, PRE_LENGTH, START_TOKEN)

    discriminator = Discriminator(sequence_length=20, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # pre_train as the positive samples on pre-training phase
    pre_train_data_loader.create_batches(pre_train)
    train_data_loader.create_batches(train)
    test_data_loader.create_batches(test)
    log = open('save/experiment-log.txt', 'w')
    #  pre-train generator
    print 'Start pre-training generator...'
    
    log.write('pre-training...\n')
    for epoch in xrange(PRE_EPOCH_NUM):
        print 'epoch: '+str(epoch)
        loss = pre_train_epoch(sess, generator, pre_train_data_loader)
        if epoch % 2 == 0 or epoch == PRE_EPOCH_NUM-1:
            #eval_data = generate_samples(sess, generator, BATCH_SIZE, generated_num)
            #test_data_loader.create_batches(test)
            test_loss = target_loss(sess, generator, test_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
    
    # pre-train discriminator
    print 'Start pre-training discriminator...'
    # Train 3 epoch on the generated data and do this for 50 times
    
    for _ in range(1):
        negative_data = generate_samples(sess, generator, BATCH_SIZE, len(pre_train), pre_train_data_loader)
        negative_data = np.array(negative_data)
        #np.save('negative_data.npy',negative_data)
        dis_data_loader.load_train_data(pre_train, negative_data)
        #for _ in range(3):
        dis_data_loader.reset_pointer()
        for it in xrange(dis_data_loader.num_batch):
            if it%1000 == 0:
                print 'pre_train discriminator epoch: '+str(it)
            x_batch, y_batch = dis_data_loader.next_batch()
            feed = {
                discriminator.input_x: x_batch,
                discriminator.input_y: y_batch,
                discriminator.dropout_keep_prob: dis_dropout_keep_prob
            }
            _ = sess.run(discriminator.train_op, feed)
    

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(TOTAL_BATCH):
        print 'total_batch: '+str(total_batch)
        # Train the generator for one step
        print 'train generator...'
        for it in range(1):
            print 'epoch '+str(it)
            #samples = generate_samples(sess, generator, BATCH_SIZE, BATCH_SIZE, train_data_loader)
            index = np.arange(len(train))
            np.random.shuffle(index)
            # get positive samples
            true_samples = train[index[:BATCH_SIZE]]
            # get negative samples
            samples = generator.generate(sess, true_samples)
            rewards = rollout.get_reward(sess, samples, PRE_LENGTH, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            # g_predictions in rewards is just the same as softmax(o_t) when generating samples.
            _ = sess.run(generator.g_updates, feed_dict=feed)
            # teacher-forcing 
            rewards_tf = np.ones((BATCH_SIZE, SEQ_LENGTH-PRE_LENGTH))
            feed_tf = {generator.x: true_samples, generator.rewards: rewards_tf}
            _ = sess.run(generator.g_updates, feed_dict=feed_tf)

        # Test
        if total_batch % 2 == 0 or total_batch == TOTAL_BATCH - 1:
            print 'test...'
            #generate_samples(sess, generator, BATCH_SIZE, generated_num, eval_file)
            #test_data_loader.create_batches(eval_file)
            test_loss = target_loss(sess, generator, test_data_loader)
            buffer = 'epoch:\t' + str(total_batch) + '\tnll:\t' + str(test_loss) + '\n'
            print 'total_batch: ', total_batch, 'test_loss: ', test_loss
            log.write(buffer)
        

        # Update roll-out parameters
        rollout.update_params()

        # Train the discriminator
        print 'train discriminator...'
        for _ in range(2):
            print 'generate negative data...'
            negative_data = generate_samples(sess, generator, BATCH_SIZE, len(train), train_data_loader)
            negative_data = np.array(negative_data)
            #index = np.arange(len(train))
            #np.random.shuffle(index)
            #positive_data = train[index[:dis_train_num]]
            dis_data_loader.load_train_data(train, negative_data)

            for _ in range(3):
                print 'train discriminator for pos/neg data'
                dis_data_loader.reset_pointer()
                for it in xrange(dis_data_loader.num_batch):
                    #print 'dis_data_loader batch '+str(it)
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    _, dis_loss = sess.run([discriminator.train_op, discriminator.loss], feed)
		print 'loss: '+str(dis_loss)

    log.close()


if __name__ == '__main__':
    main()
