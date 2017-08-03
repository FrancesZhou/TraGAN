import numpy as np
import tensorflow as tf
import random
import sys
# for mac debug
sys.path.append('/Users/frances/Documents/DeepLearning/TraGAN/model/')
sys.path.append('/Users/frances/Documents/DeepLearning/TraGAN/util/')
# for server running
sys.path.append('/home/zx/TraGAN/model/')
sys.path.append('./util/')
from tra_preprocessing import get_all_data, get_all_data2
from dataloader import Gen_Data_loader, Dis_dataloader
from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('input_length', 10, """num of input_length""")
tf.app.flags.DEFINE_integer('output_length', 10, """num of output_length""")
tf.app.flags.DEFINE_integer('seq_length', 20, """num of seqence_length""")
tf.app.flags.DEFINE_integer('emb_dim', 32, """dimensionality of embedding""")
tf.app.flags.DEFINE_integer('hidden_dim', 32, """dimensionality of hidden states""")
#tf.app.flags.DEFINE_integer('seq_length', 20, """num of seqence_length""")
tf.app.flags.DEFINE_boolean('if_pre_train', True, """whether to pre-train model or restore from checkpoint""")
tf.app.flags.DEFINE_string('model_save_dir', './save/model_save/', """directory for saving model parameters""")
tf.app.flags.DEFINE_integer('pre_epoch_num', 5, """num of epoch for pre-training""")
tf.app.flags.DEFINE_integer('batch_size', 64, """batch size for training""")
tf.app.flags.DEFINE_integer('dis_batch_size', 64, """batch size for discriminator training""")
tf.app.flags.DEFINE_integer('total_epoch_num', 200, """num of total epoch for training""")
#########################################################################################
#  Generator  Hyper-parameters
######################################################################################
# emb_dim = 32 # embedding dimension
# FLAGS.emb_dim = 32 # hidden state dimension of lstm cell
# FLAGS.input_length = 10
# FLAGS.seq_length = 20 # sequence length
START_TOKEN = 0
# FLAGS.pre_epoch_num = 2 # supervise (maximum likelihood estimation) epochs
SEED = 88
#S batch_size = 64

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
#FLAGS.total_epoch_num = 200
dis_train_num = 5000
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
    
    pre_train_data_loader = Gen_Data_loader(FLAGS.batch_size)
    train_data_loader = Gen_Data_loader(FLAGS.batch_size)
    test_data_loader = Gen_Data_loader(FLAGS.batch_size) # For testing
    vocab_size = 80*60+1
    dis_data_loader = Dis_dataloader(FLAGS.batch_size)

    generator = Generator(vocab_size, FLAGS.batch_size, FLAGS.emb_dim, FLAGS.emb_dim, FLAGS.seq_length, FLAGS.input_length, START_TOKEN)

    discriminator = Discriminator(sequence_length=FLAGS.seq_length, num_classes=2, vocab_size=vocab_size, embedding_size=dis_embedding_dim, 
                                filter_sizes=dis_filter_sizes, num_filters=dis_num_filters, l2_reg_lambda=dis_l2_reg_lambda)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    saver = tf.train.Saver()
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    # pre_train as the positive samples on pre-training phase
    pre_train_data_loader.create_batches(pre_train)
    #train_data_loader.create_batches(train)
    test_data_loader.create_batches(test)
    log = open('save/experiment-log.txt', 'w')
    
    if FLAGS.if_pre_train:
        # ======================== pre-train generator =========================
        print 'Start pre-training generator...'
        log.write('pre-training...\n')
        e2 = 100
        e1 = 10
        epoch = 0
        while abs(e2-e1)>0.001:
            #for epoch in xrange(FLAGS.pre_epoch_num):
            e1 = e2
            epoch = epoch + 1
            print '--------------- epoch: '+str(epoch)
            loss = pre_train_epoch(sess, generator, pre_train_data_loader)
            #if epoch % 1 == 0 or epoch == FLAGS.pre_epoch_num-1:
            test_loss = target_loss(sess, generator, test_data_loader)
            print 'pre-train epoch ', epoch, 'test_loss ', test_loss
            buffer = 'epoch:\t'+ str(epoch) + '\tnll:\t' + str(test_loss) + '\n'
            log.write(buffer)
            e2 = test_loss
        # ======================== pre-train discriminator =========================
        print 'Start pre-training discriminator...'
        # Train 3 epoch on the generated data and do this for pre_epoch_num/2 times
        accu = 0
        epoch = 0
        while accu < 0.85:
            #for epoch in range(FLAGS.pre_epoch_num/2):
            epoch = epoch + 1
            print '--------------- epoch: '+str(epoch)
            negative_data = generate_samples(sess, generator, FLAGS.batch_size, len(pre_train), pre_train_data_loader)
            negative_data = np.array(negative_data)
            #np.save('negative_data.npy',negative_data)
            dis_data_loader.load_train_data(pre_train, negative_data)
            accu_list = np.zeros(3)
            for a_i in range(3):
                dis_data_loader.reset_pointer()
                result_list = np.zeros([dis_data_loader.num_batch, 2])
                #accuracy_list = []
                for it in xrange(dis_data_loader.num_batch):
                    if it%1000 == 0:
                        print 'pre_train discriminator epoch: '+str(it)
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    #_ = sess.run(discriminator.train_op, feed)
                    _, dis_loss, dis_accuracy = sess.run([discriminator.train_op, discriminator.loss, discriminator.accuracy], feed)
                    result_list[it] = [dis_loss, dis_accuracy]
                # print 'pre-train discriminator result max'+str(np.max(result_list,0))
                # print 'pre-train discriminator result min'+str(np.min(result_list,0))
                print '********* pre-train discriminator result mean'+str(np.mean(result_list,0))
                accu_list[a_i] = np.mean(result_list, 0)[-1]
            accu = np.max(accu_list)
        # save pre_trained model
        save_name = FLAGS.model_save_dir+'pre_train_model.ckpt'
        save_path = saver.save(sess, save_name)
        print('Pre_trained Model saved in file: %s' % save_path)
    else:
        saver.restore(sess, FLAGS.model_save_dir+'pre_train_model.ckpt')
        print('Pre_trained Model restored.')

    rollout = ROLLOUT(generator, 0.8)

    print '#########################################################################'
    print 'Start Adversarial Training...'
    log.write('adversarial training...\n')
    for total_batch in range(FLAGS.total_epoch_num):
        print 'total_batch: '+str(total_batch)
        # Train the generator for one step
        print 'train generator...'
        for it in range(1):
            print 'epoch '+str(it)
            #samples = generate_samples(sess, generator, FLAGS.batch_size, FLAGS.batch_size, train_data_loader)
            index = np.random.randint(len(train), size=FLAGS.batch_size)
            # get positive samples
            true_samples = train[index]
            # get negative samples
            samples = generator.generate(sess, true_samples)
            rewards = rollout.get_reward(sess, samples, FLAGS.input_length, 16, discriminator)
            feed = {generator.x: samples, generator.rewards: rewards}
            # g_predictions in rewards is just the same as softmax(o_t) when generating samples.
            _ = sess.run(generator.g_updates, feed_dict=feed)
            # teacher-forcing 
            rewards_tf = np.ones((FLAGS.batch_size, FLAGS.seq_length-FLAGS.input_length))
            feed_tf = {generator.x: true_samples, generator.rewards: rewards_tf}
            _ = sess.run(generator.g_updates, feed_dict=feed_tf)

        # Test
        if total_batch % 2 == 0 or total_batch == FLAGS.total_epoch_num - 1:
            print 'test...'
            #generate_samples(sess, generator, FLAGS.batch_size, generated_num, eval_file)
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
            index = np.random.randint(len(train), size=dis_train_num)
            positive_data = train[index]
            train_data_loader.create_batches(positive_data)
            negative_data = generate_samples(sess, generator, FLAGS.batch_size, dis_train_num, train_data_loader)
            #negative_data = generator.generate(sess, positive_data)
            negative_data = np.array(negative_data)
            dis_data_loader.load_train_data(positive_data, negative_data)
            #for _ in range(3):
            accuracy = 0
            while accuracy < 0.85:
                print 'train discriminator for pos/neg data'
                dis_data_loader.reset_pointer()
                #result_list = np.zeros([dis_data_loader.num_batch, 2])
                result_list = np.empty((0))
                for it in xrange(dis_data_loader.num_batch):
                    #print 'dis_data_loader batch '+str(it)
                    x_batch, y_batch = dis_data_loader.next_batch()
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: dis_dropout_keep_prob
                    }
                    batch_accuracy = sess.run(discriminator.accuracy, feed)
                    if batch_accuracy>0.99:
                        #accuracy = batch_accuracy
                        continue
                    else:
                        _, dis_loss, dis_accuracy = sess.run([discriminator.train_op, discriminator.loss, discriminator.accuracy], feed)
                        result_list = np.append(result_list, dis_accuracy)
                if len(result_list)==0:
                    break
                accuracy = np.mean(result_list)
                # print 'pre-train discriminator result max'+str(np.max(result_list,0))
                # print 'pre-train discriminator result min'+str(np.min(result_list,0))
            print '********* discriminator accuracy mean'+str(accuracy)
		# print 'loss: '+str(dis_loss)
        # print 'accuracy: '+str(dis_accuracy)

    log.close()


if __name__ == '__main__':
    main()
