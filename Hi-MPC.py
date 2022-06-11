import numpy as np
import tensorflow as tf
import os, sys
from utils import process
from utils.faiss_rerank import compute_jaccard_distance
from tensorflow.python.layers.core import Dense
from sklearn.preprocessing import label_binarize
from sklearn.cluster import DBSCAN
import torch
import torch.nn.functional as F
import collections
from sklearn.metrics import average_precision_score
from sklearn import metrics as mr
from sklearn.metrics.cluster import adjusted_mutual_info_score as AMI_score
import gc
from functools import partial
from collections import Counter

dataset = ''
probe = ''
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

nb_nodes = 20
ft_size = 3  # originial node feature dimension (D)
time_step = 6  # sequence length (F)

# training params
batch_size = 256
nb_epochs = 100000
patience = 250  # patience for early stopping

k1, k2 = 20, 6  # parameters to compute feature distance matrix

tf.app.flags.DEFINE_string('save_model', '1', "")  # save best model
tf.app.flags.DEFINE_string('batch_size', '256', "")

tf.app.flags.DEFINE_string('model_size', '0', "")  # output model size and computational complexity

tf.app.flags.DEFINE_string('dataset', 'KS20', "Dataset: IAS, KS20, BIWI, CASIA-B or KGBD")
tf.app.flags.DEFINE_string('probe', 'probe', "for testing probe")
tf.app.flags.DEFINE_string('length', '6', "4, 6, 8 or 10")

tf.app.flags.DEFINE_string('H', '256', "")  # embedding size (h) for skeleton representations
tf.app.flags.DEFINE_string('M', '8', "")  # number (M) of meta-transformation heads
tf.app.flags.DEFINE_string('eps', '', "distance parameter in DBSCAN")
tf.app.flags.DEFINE_string('min_samples', '', "minimum sample number in DBSCAN")
tf.app.flags.DEFINE_string('gpu', '0', "GPU number")
tf.app.flags.DEFINE_string('probe_type', '', "probe.gallery")  # probe and gallery setting for CASIA-B
tf.app.flags.DEFINE_string('patience', '50', "epochs for early stopping")
tf.app.flags.DEFINE_string('mode', 'Train', "Training (Train) or Evaluation (Eval)")
tf.app.flags.DEFINE_string('lr', '0.00035', "learning rate")

tf.app.flags.DEFINE_string('k1', '20', "")
tf.app.flags.DEFINE_string('k2', '6', "")

tf.app.flags.DEFINE_string('focus', '1', "")

FLAGS = tf.app.flags.FLAGS

k1, k2 = int(FLAGS.k1), int(FLAGS.k2)

# check parameters
if FLAGS.dataset not in ['IAS', 'KGBD', 'KS20', 'BIWI', 'CASIA_B']:
	raise Exception('Dataset must be IAS, KGBD, KS20, BIWI or CASIA B.')
if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'
	if FLAGS.length not in ['40', '50', '60']:
		raise Exception('Length number must be 40, 50 or 60')
else:
	if FLAGS.length not in ['4', '6', '8', '10']:
		raise Exception('Length number must be 4, 6, 8 or 10')
if FLAGS.probe not in ['probe', 'Walking', 'Still', 'A', 'B']:
	raise Exception('Dataset probe must be "A" (for IAS-A), "B" (for IAS-B), "probe" (for KS20, KGBD).')

if FLAGS.mode not in ['Train', 'Eval']:
	raise Exception('Mode must be Train or Eval.')

os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
dataset = FLAGS.dataset

# optimal paramters
if dataset == 'KGBD':
	FLAGS.lr = '0.00035'
	if FLAGS.min_samples == '':
		FLAGS.min_samples = '4'

elif dataset == 'CASIA_B':
	FLAGS.lr = '0.00035'
	if FLAGS.min_samples == '':
		FLAGS.min_samples = '2'
	if FLAGS.eps == '':
		FLAGS.eps = '0.75'

else:
	FLAGS.lr = '0.00035'
if dataset == 'KS20' or dataset == 'IAS':

	if FLAGS.eps == '':
		FLAGS.eps = '0.8'
elif dataset == 'BIWI':
	if FLAGS.probe == 'Walking':
		if FLAGS.eps == '':
			FLAGS.eps = '0.8'

T_eps = FLAGS.eps
T_min_a = FLAGS.min_samples

if FLAGS.eps == '':
	FLAGS.eps = '0.6'

if FLAGS.min_samples == '':
	FLAGS.min_samples = '2'

eps = float(FLAGS.eps)
min_samples = int(FLAGS.min_samples)

time_step = int(FLAGS.length)
probe = FLAGS.probe
patience = int(FLAGS.patience)
batch_size = int(FLAGS.batch_size)

# not used
global_att = False
nhood = 1
residual = False
nonlinearity = tf.nn.elu

pre_dir = 'ReID_Models/'
# Customize the [directory] to save models with different hyper-parameters
change = '_Hi-MPC_Formal'

# [directory] = [pre_dir] + [dataset] + '/' + [probe] + [change] + '/' + 'best.ckpt'
# e.g., ReID_Models/BIWI/Walking_Hi-MPC/best.ckpt

if FLAGS.probe_type != '':
	change += '_CME'

try:
	os.mkdir(pre_dir)
except:
	pass

if dataset == 'KS20':
	nb_nodes = 25

if dataset == 'CASIA_B':
	nb_nodes = 14

if FLAGS.dataset == 'CASIA_B':
	FLAGS.length = '40'

print('----- Model hyperparams -----')

print('batch_size: ' + str(batch_size))
print('M: ' + FLAGS.M)
print('H: ' + FLAGS.H)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('seqence_length: ' + str(time_step))
print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	if dataset == 'CASIA_B':
		print('Probe.Gallery: ', FLAGS.probe_type.split('.')[0], FLAGS.probe_type.split('.')[1])
	else:
		print('Probe: ' + FLAGS.probe)

"""
 Codes from our project of SPC-MGR
 We use joint-level (J), component-level (P), and limb-level (B) skeleton data
"""

norm = True

if FLAGS.probe_type == '':
	if FLAGS.dataset == 'KS20':
		nb_nodes = 25
	X_train_J, X_train_P, X_train_B, _, _, y_train, X_test_J, X_test_P, X_test_B, _, _, y_test, \
	adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_test_J_D = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
							   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size, norm=norm)
	del _
	gc.collect()



else:
	from utils import process_cme_L3 as process

	X_train_J, X_train_P, X_train_B, _, _, y_train, X_test_J, X_test_P, X_test_B, _, _, y_test, \
	adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
		process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
							   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
							   PG_type=FLAGS.probe_type.split('.')[0])
	print('## [Probe].[Gallery]', FLAGS.probe_type)
	del _
	gc.collect()

all_ftr_size = int(FLAGS.H)
loaded_graph = tf.Graph()
joint_num = X_train_J.shape[2]

train_epochs = 15000
display = 80

imp_val, imp_val_P, imp_val_B = None, None, None

if FLAGS.mode == 'Train':
	loaded_graph = tf.Graph()
	with loaded_graph.as_default():
		with tf.name_scope('Input'):
			H = int(FLAGS.H)

			J_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, joint_num, ft_size))
			P_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 10, ft_size))
			B_in = tf.placeholder(dtype=tf.float32, shape=(batch_size * time_step, 5, ft_size))

			pseudo_lab_J = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_J = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))

			pseudo_lab_P = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_P = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))

			pseudo_lab_B = tf.placeholder(dtype=tf.int32, shape=(batch_size,))
			seq_cluster_ftr_B = tf.placeholder(dtype=tf.float32, shape=(None, all_ftr_size))

			lbl_s = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_classes))

		with tf.name_scope("Encoder"), tf.variable_scope("", reuse=tf.AUTO_REUSE):

			H = int(FLAGS.H)

			W_1 = tf.get_variable('W_1', shape=[joint_num * ft_size, H],
								  initializer=tf.glorot_uniform_initializer())
			W_1_P = tf.get_variable('W_1_P', shape=[10 * ft_size, H],
									initializer=tf.glorot_uniform_initializer())
			W_1_B = tf.get_variable('W_1_B', shape=[5 * ft_size, H],
									initializer=tf.glorot_uniform_initializer())

			b_1 = tf.Variable(tf.zeros(shape=[H, ]))
			b_1_P = tf.Variable(tf.zeros(shape=[H, ]))
			b_1_B = tf.Variable(tf.zeros(shape=[H, ]))

			W_2 = tf.get_variable('W_2', shape=[H, H], initializer=tf.glorot_uniform_initializer())
			W_2_P = tf.get_variable('W_2_P', shape=[H, H], initializer=tf.glorot_uniform_initializer())
			W_2_B = tf.get_variable('W_2_B', shape=[H, H], initializer=tf.glorot_uniform_initializer())
			b_2 = tf.Variable(tf.zeros(shape=[H, ]))
			b_2_P = tf.Variable(tf.zeros(shape=[H, ]))
			b_2_B = tf.Variable(tf.zeros(shape=[H, ]))

			inputs = tf.reshape(J_in, [time_step * batch_size, -1])

			s_rep = tf.matmul(tf.nn.relu(tf.matmul(inputs, W_1) + b_1), W_2) + b_2

			inputs_P = tf.reshape(P_in, [time_step * batch_size, -1])
			inputs_B = tf.reshape(B_in, [time_step * batch_size, -1])

			s_rep_P = tf.matmul(tf.nn.relu(tf.matmul(inputs_P, W_1_P) + b_1_P), W_2_P) + b_2_P
			s_rep_B = tf.matmul(tf.nn.relu(tf.matmul(inputs_B, W_1_B) + b_1_B), W_2_B) + b_2_B

			seq_ftr = tf.reshape(s_rep, [batch_size, time_step, -1])
			seq_ftr_P = tf.reshape(s_rep_P, [batch_size, time_step, -1])
			seq_ftr_B = tf.reshape(s_rep_B, [batch_size, time_step, -1])

			seq_ftr_frames = seq_ftr

			seq_ftr = tf.reduce_mean(seq_ftr, axis=1)
			seq_ftr = tf.reshape(seq_ftr, [batch_size, -1])

			seq_ftr_P_frames = seq_ftr_P

			seq_ftr_P = tf.reduce_mean(seq_ftr_P, axis=1)
			seq_ftr_P = tf.reshape(seq_ftr_P, [batch_size, -1])

			seq_ftr_B_frames = seq_ftr_B

			seq_ftr_B = tf.reduce_mean(seq_ftr_B, axis=1)
			seq_ftr_B = tf.reshape(seq_ftr_B, [batch_size, -1])

		with tf.name_scope("Hi_MPC"), tf.variable_scope("Hi_MPC", reuse=tf.AUTO_REUSE):

			def Hi_MPC_hard(t, pseudo_lab, all_ftr, cluster_ftr, pseudo_lab_P, all_ftr_P, cluster_ftr_P, pseudo_lab_B,
							all_ftr_B, cluster_ftr_B):
				global imp_val, imp_val_P, imp_val_B

				M = int(FLAGS.M)

				concat_heads = tf.zeros_like(seq_ftr)
				concat_heads_clu = tf.zeros_like(cluster_ftr)
				W_head = lambda: tf.Variable(tf.random_normal([H, H]))
				all_ftr_mean = tf.reduce_mean(all_ftr, axis=1)
				all_ftr_P_mean = tf.reduce_mean(all_ftr_P, axis=1)
				all_ftr_B_mean = tf.reduce_mean(all_ftr_B, axis=1)

				for i in range(M):

					W_q_head = W_k_head = tf.Variable(initial_value=W_head)
					W_q_head_P = W_k_head_P = tf.Variable(initial_value=W_head)
					W_q_head_B = W_k_head_B = tf.Variable(initial_value=W_head)

					all_ftr_trans = tf.matmul(all_ftr, W_q_head)
					all_ftr_trans_mean = tf.matmul(all_ftr_mean, W_q_head)
					cluster_ftr_trans = tf.matmul(cluster_ftr, W_k_head)

					all_ftr_trans_P = tf.matmul(all_ftr_P, W_q_head_P)
					all_ftr_trans_P_mean = tf.matmul(all_ftr_P_mean, W_q_head_P)
					cluster_ftr_trans_P = tf.matmul(cluster_ftr_P, W_k_head_P)

					all_ftr_trans_B = tf.matmul(all_ftr_B, W_q_head_B)
					all_ftr_trans_B_mean = tf.matmul(all_ftr_B_mean, W_q_head_B)
					cluster_ftr_trans_B = tf.matmul(cluster_ftr_B, W_k_head_B)

					pred_lbl = tf.argmax(tf.matmul(all_ftr_trans_mean, tf.transpose(cluster_ftr_trans)) / np.sqrt(H),
										 -1)
					pred_lbl_P = tf.argmax(
						tf.matmul(all_ftr_trans_P_mean, tf.transpose(cluster_ftr_trans_P)) / np.sqrt(H), -1)
					pred_lbl_B = tf.argmax(
						tf.matmul(all_ftr_trans_B_mean, tf.transpose(cluster_ftr_trans_B)) / np.sqrt(H), -1)

					# importance inference
					logits = tf.matmul(all_ftr_trans, tf.transpose(cluster_ftr_trans)) / np.sqrt(H)
					logits_P = tf.matmul(all_ftr_trans_P, tf.transpose(cluster_ftr_trans_P)) / np.sqrt(H)
					logits_B = tf.matmul(all_ftr_trans_B, tf.transpose(cluster_ftr_trans_B)) / np.sqrt(H)

					pred_lbl_frames = tf.reshape(tf.tile(tf.reshape(pred_lbl, [-1, 1]), [1, time_step]), [-1])
					pred_lbl_P_frames = tf.reshape(tf.tile(tf.reshape(pred_lbl_P, [-1, 1]), [1, time_step]), [-1])
					pred_lbl_B_frames = tf.reshape(tf.tile(tf.reshape(pred_lbl_B, [-1, 1]), [1, time_step]), [-1])

					pred_lbl_frames = tf.cast(pred_lbl_frames, tf.int32)
					pred_lbl_P_frames = tf.cast(pred_lbl_P_frames, tf.int32)
					pred_lbl_B_frames = tf.cast(pred_lbl_B_frames, tf.int32)

					# [batch_size, time_step]
					pseudo_lab_frames = tf.reshape(tf.tile(tf.reshape(pseudo_lab, [-1, 1]), [1, time_step]), [-1])
					pseudo_lab_P_frames = tf.reshape(tf.tile(tf.reshape(pseudo_lab_P, [-1, 1]), [1, time_step]), [-1])
					pseudo_lab_B_frames = tf.reshape(tf.tile(tf.reshape(pseudo_lab_B, [-1, 1]), [1, time_step]), [-1])

					# [batch_size, time_step]
					# If pred. is true, focus on less-score frames, otherwise focus on high-score (wrong label) frames
					indices = tf.concat([tf.reshape(tf.range(0, batch_size * time_step), [-1, 1]),
										 tf.reshape(pred_lbl_frames, [-1, 1])], axis=-1)
					indices_P = tf.concat([tf.reshape(tf.range(0, batch_size * time_step), [-1, 1]),
										   tf.reshape(pred_lbl_P_frames, [-1, 1])], axis=-1)
					indices_B = tf.concat([tf.reshape(tf.range(0, batch_size * time_step), [-1, 1]),
										   tf.reshape(pred_lbl_B_frames, [-1, 1])], axis=-1)

					imp_frames_unorm = tf.gather_nd(tf.reshape(logits, [batch_size * time_step, -1]), indices)
					imp_P_frames_unorm = tf.gather_nd(tf.reshape(logits_P, [batch_size * time_step, -1]), indices_P)
					imp_B_frames_unorm = tf.gather_nd(tf.reshape(logits_B, [batch_size * time_step, -1]), indices_B)

					imp_frames_unorm = tf.reshape(imp_frames_unorm, [-1, time_step])
					imp_P_frames_unorm = tf.reshape(imp_P_frames_unorm, [-1, time_step])
					imp_B_frames_unorm = tf.reshape(imp_B_frames_unorm, [-1, time_step])

					ones = tf.ones_like(pseudo_lab_frames, dtype=tf.float32)

					if FLAGS.focus == '1':
						imp_frames = tf.nn.softmax(tf.reshape(
							tf.reshape(imp_frames_unorm, [-1]) * tf.where(tf.equal(pred_lbl_frames, pseudo_lab_frames),
																		  -ones, ones), [batch_size, time_step]),
							axis=-1)
						imp_P_frames = tf.nn.softmax(tf.reshape(tf.reshape(imp_P_frames_unorm, [-1]) * tf.where(
							tf.equal(pred_lbl_P_frames, pseudo_lab_P_frames),
							-ones, ones), [batch_size, time_step]), axis=-1)
						imp_B_frames = tf.nn.softmax(tf.reshape(tf.reshape(imp_B_frames_unorm, [-1]) * tf.where(
							tf.equal(pred_lbl_B_frames, pseudo_lab_B_frames),
							-ones, ones), [batch_size, time_step]), axis=-1)


					elif FLAGS.focus == '-1':
						imp_frames = tf.nn.softmax(tf.reshape(tf.reshape(imp_frames_unorm, [-1]) * tf.where(
							tf.equal(pred_lbl_frames, pseudo_lab_frames),
							ones, -ones), [batch_size, time_step]), axis=-1)
						imp_P_frames = tf.nn.softmax(tf.reshape(tf.reshape(imp_P_frames_unorm, [-1]) * tf.where(
							tf.equal(pred_lbl_P_frames, pseudo_lab_P_frames),
							ones, -ones), [batch_size, time_step]), axis=-1)
						imp_B_frames = tf.nn.softmax(tf.reshape(tf.reshape(imp_B_frames_unorm, [-1]) * tf.where(
							tf.equal(pred_lbl_B_frames, pseudo_lab_B_frames),
							ones, -ones), [batch_size, time_step]), axis=-1)
					else:
						imp_frames = tf.ones([batch_size, time_step])
						imp_P_frames = tf.ones([batch_size, time_step])
						imp_B_frames = tf.ones([batch_size, time_step])

					W_att_int = all_ftr_trans_mean
					W_att_int_P = all_ftr_trans_P_mean
					W_att_int_B = all_ftr_trans_B_mean

					pseudo_lab_frames = tf.reshape(pseudo_lab_frames, [batch_size, time_step])
					pseudo_lab_P_frames = tf.reshape(pseudo_lab_P_frames, [batch_size, time_step])
					pseudo_lab_B_frames = tf.reshape(pseudo_lab_B_frames, [batch_size, time_step])

					if i == 0:

						# weighted loss for all frames in each sequece, and then average all sequences

						loss = tf.reduce_mean(tf.reduce_sum(
							imp_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab_frames,
																						logits=logits), axis=-1))
						loss_P = tf.reduce_mean(tf.reduce_sum(
							imp_P_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab_P_frames,
																						  logits=logits_P), axis=-1))
						loss_B = tf.reduce_mean(tf.reduce_sum(
							imp_B_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab_B_frames,
																						  logits=logits_B), axis=-1))

						W_att_int_ave = W_att_int
						W_att_int_P_ave = W_att_int_P
						W_att_int_B_ave = W_att_int_B

						# imp for evaluation
						imp_val = tf.nn.softmax(logits, axis=-1)
						imp_val_P = tf.nn.softmax(logits_P, axis=-1)
						imp_val_B = tf.nn.softmax(logits_B, axis=-1)


					else:

						loss = loss + tf.reduce_mean(tf.reduce_sum(
							imp_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=pseudo_lab_frames,
																						logits=logits), axis=-1))
						loss_P = loss_P + tf.reduce_mean(tf.reduce_sum(
							imp_P_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(
								labels=pseudo_lab_P_frames, logits=logits_P), axis=-1))
						loss_B = loss_B + tf.reduce_mean(tf.reduce_sum(
							imp_B_frames * tf.nn.sparse_softmax_cross_entropy_with_logits(
								labels=pseudo_lab_B_frames, logits=logits_B), axis=-1))
						W_att_int_ave = W_att_int_ave + W_att_int
						W_att_int_P_ave = W_att_int_P_ave + W_att_int_P
						W_att_int_B_ave = W_att_int_B_ave + W_att_int_B
						# imp for evaluation
						imp_val = imp_val + tf.nn.softmax(logits, axis=-1)
						imp_val_P = imp_val_P + tf.nn.softmax(logits_P, axis=-1)
						imp_val_B = imp_val_B + tf.nn.softmax(logits_B, axis=-1)

				# imp for evaluation
				imp_val = imp_val / M
				imp_val_P = imp_val_P / M
				imp_val_B = imp_val_B / M

				loss_P = loss_P / M
				loss_B = loss_B / M
				W_att_int_ave = W_att_int_ave / M
				W_att_int_P_ave = W_att_int_P_ave / M
				W_att_int_B_ave = W_att_int_B_ave / M
				return loss, loss_P, loss_B, W_att_int_ave, W_att_int_P_ave, W_att_int_B_ave


			loss_J, loss_P, loss_B, W_att_int_ave, W_att_int_P_ave, W_att_int_B_ave = Hi_MPC_hard(
				np.sqrt(H), pseudo_lab_J, seq_ftr_frames, seq_cluster_ftr_J, pseudo_lab_P, seq_ftr_P_frames,
				seq_cluster_ftr_P, pseudo_lab_B, seq_ftr_B_frames, seq_cluster_ftr_B)

			Hi_MPC_loss = (loss_J + loss_P + loss_B) / 3

			seq_ftr_int = tf.concat([W_att_int_ave, W_att_int_P_ave, W_att_int_B_ave], axis=-1)

			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))

			optimizer = tf.train.AdamOptimizer(learning_rate=float(FLAGS.lr))

			train_op = optimizer.minimize(Hi_MPC_loss)

		saver = tf.train.Saver()
		init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

		with tf.Session(config=config) as sess:
			sess.run(init_op)
			if FLAGS.model_size == '1':
				# compute model size (M) and computational complexity (GFLOPs)
				def stats_graph(graph):
					flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
					params = tf.profiler.profile(graph,
												 options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
					print('FLOPs: {} GFLOPS;    Trainable params: {} M'.format(flops.total_float_ops / 1e9,
																			   params.total_parameters / 1e6))


				stats_graph(loaded_graph)
				exit()

			# only initialization
			cluster_features = np.random.random([batch_size, H])
			cluster_features_P = np.random.random([batch_size, H])
			cluster_features_B = np.random.random([batch_size, H])


			def train_loader(X_train_J, X_train_P, X_train_B, y_train):

				tr_step = 0
				tr_size = X_train_J.shape[0]
				train_labels_all = []
				train_features_all_int = []
				train_features_all = []
				train_features_all_P = []
				train_features_all_B = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]

					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_input_B.reshape([-1, 5, 3])

					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]

					[all_features_int, all_features, all_features_P, all_features_B] = sess.run(
						[seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B, ],
						feed_dict={
							J_in: X_input_J,
							P_in: X_input_P,
							B_in: X_input_B,
						})
					train_features_all_P.extend(all_features_P.tolist())
					train_features_all_B.extend(all_features_B.tolist())
					train_features_all_int.extend(all_features_int.tolist())
					train_features_all.extend(all_features.tolist())
					train_labels_all.extend(labels.tolist())
					tr_step += 1

				train_features_all_int = np.array(train_features_all_int).astype(np.float32)
				train_features_all_int = torch.from_numpy(train_features_all_int)

				train_features_all = np.array(train_features_all).astype(np.float32)
				train_features_all = torch.from_numpy(train_features_all)

				train_features_all_P = np.array(train_features_all_P).astype(np.float32)
				train_features_all_P = torch.from_numpy(train_features_all_P)

				train_features_all_B = np.array(train_features_all_B).astype(np.float32)
				train_features_all_B = torch.from_numpy(train_features_all_B)

				return train_features_all_int, train_features_all, train_features_all_P, train_features_all_B, train_labels_all


			def gal_loader(X_train_J, X_train_P, X_train_B, y_train):
				tr_step = 0
				tr_size = X_train_J.shape[0]
				gal_logits_all = []
				gal_labels_all = []
				gal_features_all_int = []
				gal_features_all = []
				gal_features_all_P = []
				gal_features_all_B = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]

					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_input_B.reshape([-1, 5, 3])

					labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]

					[Seq_features_int, Seq_features, Seq_features_P, Seq_features_B] = sess.run(
						[seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B],
						feed_dict={
							J_in: X_input_J,
							P_in: X_input_P,
							B_in: X_input_B,

						})

					gal_features_all_int.extend(Seq_features_int.tolist())
					gal_features_all.extend(Seq_features.tolist())
					gal_features_all_P.extend(Seq_features_P.tolist())
					gal_features_all_B.extend(Seq_features_B.tolist())
					# gal_features_tea.extend(rep_tea_.tolist())
					gal_labels_all.extend(labels.tolist())
					tr_step += 1

				return gal_features_all_int, gal_features_all, gal_features_all_P, gal_features_all_B, gal_labels_all


			def evaluation():
				vl_step = 0
				vl_size = X_test_J.shape[0]
				pro_labels_all = []
				pro_features_all = []

				pro_features_tea = []
				pro_features_tea_2 = []

				vl_step = 0
				vl_size = X_test_J.shape[0]
				pro_labels_all = []
				pro_features_all_int = []
				pro_features_all = []
				pro_features_all_P = []
				pro_features_all_B = []

				while vl_step * batch_size < vl_size:
					if (vl_step + 1) * batch_size > vl_size:
						break
					X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
					X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]

					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_input_B.reshape([-1, 5, 3])

					labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]

					[Seq_features_int, Seq_features, Seq_features_P, Seq_features_B] = sess.run(
						[seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B],
						feed_dict={
							J_in: X_input_J,
							P_in: X_input_P,
							B_in: X_input_B,

						})

					pro_labels_all.extend(labels.tolist())
					pro_features_all_int.extend(Seq_features_int.tolist())
					pro_features_all.extend(Seq_features.tolist())
					pro_features_all_P.extend(Seq_features_P.tolist())
					pro_features_all_B.extend(Seq_features_B.tolist())

					vl_step += 1

				X = np.array(gal_features_all)

				X_int = np.array(gal_features_all_int)
				X_P = np.array(gal_features_all_P)
				X_B = np.array(gal_features_all_B)

				y = np.array(gal_labels_all)

				t_X = np.array(pro_features_all)
				t_X_int = np.array(pro_features_all_int)
				t_X_P = np.array(pro_features_all_P)
				t_X_B = np.array(pro_features_all_B)

				t_y = np.array(pro_labels_all)

				t_y = np.argmax(t_y, axis=-1)
				y = np.argmax(y, axis=-1)

				def mean_ap(distmat, query_ids=None, gallery_ids=None,
							query_cams=None, gallery_cams=None):
					# distmat = to_numpy(distmat)
					m, n = distmat.shape
					# Fill up default values
					if query_ids is None:
						query_ids = np.arange(m)
					if gallery_ids is None:
						gallery_ids = np.arange(n)
					if query_cams is None:
						query_cams = np.zeros(m).astype(np.int32)
					if gallery_cams is None:
						gallery_cams = np.ones(n).astype(np.int32)
					# Ensure numpy array
					query_ids = np.asarray(query_ids)
					gallery_ids = np.asarray(gallery_ids)
					query_cams = np.asarray(query_cams)
					gallery_cams = np.asarray(gallery_cams)
					# Sort and find correct matches
					indices = np.argsort(distmat, axis=1)
					matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
					# Compute AP for each query
					aps = []
					if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(1, m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))

							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							y_score[np.isnan(y_score)] = 0
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					else:
						for i in range(m):
							valid = ((gallery_ids[indices[i]] != query_ids[i]) |
									 (gallery_cams[indices[i]] != query_cams[i]))
							y_true = matches[i, valid]
							y_score = -distmat[i][indices[i]][valid]
							# discard nan
							# y_score = np.nan_to_num(y_score)
							if not np.any(y_true): continue
							aps.append(average_precision_score(y_true, y_score))
					if len(aps) == 0:
						raise RuntimeError("No valid query")
					return np.mean(aps)

				def metrics(X, y, t_X, t_y):
					# compute Euclidean distance
					if dataset != 'CASIA_B':
						a, b = torch.from_numpy(t_X), torch.from_numpy(X)
						m, n = a.size(0), b.size(0)
						a = a.view(m, -1)
						b = b.view(n, -1)
						dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
								 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
						dist_m.addmm_(1, -2, a, b.t())
						dist_m = (dist_m.clamp(min=1e-12)).sqrt()
						mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
						_, dist_sort = dist_m.sort(1)
						dist_sort = dist_sort.numpy()
					else:
						X = np.array(X)
						t_X = np.array(t_X)
						dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_m = np.array(dist_m)
						mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
						dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
						dist_sort = np.array(dist_sort)

					top_1 = top_5 = top_10 = 0
					probe_num = dist_sort.shape[0]
					if (FLAGS.probe_type == 'nm.nm' or
							FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
						for i in range(probe_num):

							if t_y[i] in y[dist_sort[i, 1:2]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, 1:6]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, 1:11]]:
								top_10 += 1
					else:
						for i in range(probe_num):

							if t_y[i] in y[dist_sort[i, :1]]:
								top_1 += 1
							if t_y[i] in y[dist_sort[i, :5]]:
								top_5 += 1
							if t_y[i] in y[dist_sort[i, :10]]:
								top_10 += 1
					return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

				mAP_int, top_1_int, top_5_int, top_10_int = metrics(X_int, y, t_X_int, t_y)
				mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
				mAP_P, top_1_P, top_5_P, top_10_P = metrics(X_P, y, t_X_P, t_y)
				mAP_B, top_1_B, top_5_B, top_10_B = metrics(X_B, y, t_X_B, t_y)
				del X, y, t_X, t_y, X_P, t_X_P, X_B, t_X_B, pro_labels_all, pro_features_all
				gc.collect()

				return mAP_int, top_1_int, top_5_int, top_10_int, \
					   mAP, top_1, top_5, top_10, \
					   mAP_P, top_1_P, top_5_P, top_10_P, mAP_B, top_1_B, top_5_B, top_10_B


			max_acc_1 = 0
			max_acc_2 = 0
			top_5_max = 0
			top_10_max = 0

			max_acc_1_int = 0
			max_acc_2_int = 0
			top_5_max_int = 0
			top_10_max_int = 0

			max_acc_1_P = 0
			max_acc_2_P = 0
			top_5_max_P = 0
			top_10_max_P = 0

			max_acc_1_B = 0
			max_acc_2_B = 0
			top_5_max_B = 0
			top_10_max_B = 0

			best_cluster_info_1 = [0, 0]
			best_cluster_info_2 = [0, 0]
			cur_patience = 0
			MI_1s = []
			MI_2s = []
			AMI_1s = []
			AMI_2s = []
			top_1s = []
			top_5s = []
			top_10s = []
			mAPs = []
			Hi_MPC_losses = []
			mACT = []
			mRCL = []

			if dataset == 'KGBD' or dataset == 'KS20':
				if FLAGS.dataset == 'KS20':
					nb_nodes = 25
				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
					process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
										   norm=norm
										   )
				nb_nodes = 20
			elif dataset == 'BIWI':
				if probe == 'Walking':
					X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
						process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, norm=norm
											   )
				else:
					X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
						process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, norm=norm
											   )
			elif dataset == 'IAS':
				if probe == 'A':
					X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
						process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, norm=norm
											   )
				else:
					X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
					adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
						process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
											   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
											   batch_size=batch_size, norm=norm
											   )
			elif dataset == 'CASIA_B':

				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
					process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
										   PG_type=FLAGS.probe_type.split('.')[1])
			del _
			gc.collect()

			for epoch in range(train_epochs):

				train_features_all_int, train_features_all, train_features_all_P, train_features_all_B, train_labels_all = train_loader(
					X_train_J, X_train_P, X_train_B, y_train)

				gal_features_all_int, gal_features_all, gal_features_all_P, gal_features_all_B, gal_labels_all = gal_loader(
					X_gal_J, X_gal_P, X_gal_B, y_gal)

				mAP_int, top_1_int, top_5_int, top_10_int, mAP, top_1, top_5, top_10, mAP_P, top_1_P, top_5_P, top_10_P, \
				mAP_B, top_1_B, top_5_B, top_10_B, = evaluation()

				cur_patience += 1

				if epoch > 0 and top_1 > max_acc_2:
					max_acc_1 = mAP
					max_acc_2 = top_1
					top_5_max = top_5
					top_10_max = top_10

				if epoch > 0 and top_1_int > max_acc_2_int:
					max_acc_1_int = mAP_int
					max_acc_2_int = top_1_int
					top_5_max_int = top_5_int
					top_10_max_int = top_10_int
					if FLAGS.mode == 'Train':
						if FLAGS.dataset != 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + 'best.ckpt'
						elif FLAGS.dataset == 'CASIA_B':
							checkpt_file = pre_dir + dataset + '/' + probe + change + '/' + FLAGS.probe_type + '_best.ckpt'
						print(checkpt_file)
						if FLAGS.save_model == '1':
							saver.save(sess, checkpt_file)
					cur_patience = 0
				if epoch > 0 and top_1_P > max_acc_2_P:
					max_acc_1_P = mAP_P
					max_acc_2_P = top_1_P
					top_5_max_P = top_5_P
					top_10_max_P = top_10_P
				if epoch > 0 and top_1_B > max_acc_2_B:
					max_acc_1_B = mAP_B
					max_acc_2_B = top_1_B
					top_5_max_B = top_5_B
					top_10_max_B = top_10_B
				if epoch > 0:
					# print(
					# 	'[Joint-Level] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f)' % (
					# 		FLAGS.dataset, FLAGS.probe,
					# 		top_1, max_acc_2, top_5, top_5_max, top_10, top_10_max, mAP, max_acc_1,))
					#
					# print(
					# 	'[Component-Level] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f) ' % (
					# 		FLAGS.dataset, FLAGS.probe,
					# 		top_1_P, max_acc_2_P, top_5_P, top_5_max_P, top_10_P, top_10_max_P, mAP_P,
					# 		max_acc_1_P,))
					# print(
					# 	'[Limb-Level] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f) ' % (
					# 		FLAGS.dataset, FLAGS.probe,
					# 		top_1_B, max_acc_2_B, top_5_B, top_5_max_B, top_10_B, top_10_max_B, mAP_B,
					# 		max_acc_1_B,))
					print(
						'[MSMR] %s - %s | Top-1: %.4f (%.4f) | Top-5: %.4f (%.4f) | Top-10: %.4f (%.4f) | mAP: %.4f (%.4f) ' % (
							FLAGS.dataset, FLAGS.probe,
							top_1_int, max_acc_2_int, top_5_int, top_5_max_int, top_10_int, top_10_max_int,
							mAP_int, max_acc_1_int,))
					# print(
					# 	" %.4f-%.4f-%.4f-%.4f \n %.4f-%.4f-%.4f-%.4f \n %.4f-%.4f-%.4f-%.4f \n %.4f-%.4f-%.4f-%.4f" % (
					# 		max_acc_2, top_5_max, top_10_max, max_acc_1,
					# 		max_acc_2_P, top_5_max_P, top_10_max_P, max_acc_1_P, max_acc_2_B, top_5_max_B,
					# 		top_10_max_B, max_acc_1_B,
					# 		max_acc_2_int, top_5_max_int, top_10_max_int, max_acc_1_int))

				if cur_patience == patience:
					break


				def generate_cluster_features(labels, features):
					centers = collections.defaultdict(list)
					for i, label in enumerate(labels):
						if label == -1:
							continue
						centers[labels[i]].append(features[i])

					centers = [
						torch.stack(centers[idx], dim=0).mean(0) for idx in sorted(centers.keys())
					]
					centers = torch.stack(centers, dim=0)
					return centers


				rerank_dist = compute_jaccard_distance(train_features_all, k1=k1, k2=k2)
				rerank_dist_P = compute_jaccard_distance(train_features_all_P, k1=k1, k2=k2)
				rerank_dist_B = compute_jaccard_distance(train_features_all_B, k1=k1, k2=k2)
				cluster = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
				pseudo_labels = cluster.fit_predict(rerank_dist)
				pseudo_labels_P = cluster.fit_predict(rerank_dist_P)
				pseudo_labels_B = cluster.fit_predict(rerank_dist_B)

				X_train_J_new = X_train_J

				cluster_features = generate_cluster_features(pseudo_labels, train_features_all)
				cluster_features = cluster_features.numpy()
				cluster_features = cluster_features.astype(np.float64)

				cluster_features_P = generate_cluster_features(pseudo_labels_P, train_features_all_P)
				cluster_features_P = cluster_features_P.numpy()
				cluster_features_P = cluster_features_P.astype(np.float64)

				cluster_features_B = generate_cluster_features(pseudo_labels_B, train_features_all_B)
				cluster_features_B = cluster_features_B.numpy()
				cluster_features_B = cluster_features_B.astype(np.float64)

				X_train_J_new = X_train_J[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]
				X_train_P_new = X_train_P[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]
				X_train_B_new = X_train_B[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]

				outlier_num = np.sum((pseudo_labels == -1))
				pseudo_labels_new = pseudo_labels[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]
				pseudo_labels_P_new = pseudo_labels_P[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]
				pseudo_labels_B_new = pseudo_labels_B[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]

				train_labels_all = np.array(train_labels_all)
				train_labels_all = train_labels_all[
					np.where((pseudo_labels != -1) & (pseudo_labels_P != -1) & (pseudo_labels_B != -1))]
				pseudo_labels = pseudo_labels_new

				outlier_num_P = np.sum((pseudo_labels_P == -1))

				pseudo_labels_P = pseudo_labels_P_new

				outlier_num_B = np.sum((pseudo_labels_B == -1))

				pseudo_labels_B = pseudo_labels_B_new

				num_cluster = len(set(pseudo_labels)) - (1 if -1 in pseudo_labels else 0)
				num_cluster_P = len(set(pseudo_labels_P)) - (1 if -1 in pseudo_labels_P else 0)
				num_cluster_B = len(set(pseudo_labels_B)) - (1 if -1 in pseudo_labels_B else 0)

				tr_step = 0
				tr_size = X_train_J_new.shape[0]

				batch_Hi_MPC_loss = []

				while tr_step * batch_size < tr_size:
					if (tr_step + 1) * batch_size > tr_size:
						break
					X_input_J = X_train_J_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_P = X_train_P_new[tr_step * batch_size:(tr_step + 1) * batch_size]
					X_input_B = X_train_B_new[tr_step * batch_size:(tr_step + 1) * batch_size]

					X_input_J = X_input_J.reshape([-1, joint_num, 3])
					X_input_P = X_input_P.reshape([-1, 10, 3])
					X_input_B = X_input_B.reshape([-1, 5, 3])

					labels = pseudo_labels[tr_step * batch_size:(tr_step + 1) * batch_size]
					labels_P = pseudo_labels_P[tr_step * batch_size:(tr_step + 1) * batch_size]
					labels_B = pseudo_labels_B[tr_step * batch_size:(tr_step + 1) * batch_size]

					_, loss, loss_J_, loss_P_, loss_B_, Seq_features = sess.run(
						[train_op, Hi_MPC_loss, loss_J, loss_P, loss_B, seq_ftr],
						feed_dict={
							J_in: X_input_J,
							P_in: X_input_P,
							B_in: X_input_B,
							pseudo_lab_J: labels,
							pseudo_lab_P: labels_P,
							pseudo_lab_B: labels_B,
							seq_cluster_ftr_J: cluster_features,
							seq_cluster_ftr_P: cluster_features_P,
							seq_cluster_ftr_B: cluster_features_B})
					Seq_features = torch.from_numpy(Seq_features)
					batch_Hi_MPC_loss.append(loss)

					if tr_step % display == 0:
						print(
							'[%s] Batch num: %d | Loss: %.3f | J/C/L Cluser num: %d, %d, %d  | J/C/L Loss: %.3f, %.3f, %.3f ' %
							(str(epoch), tr_step, loss, num_cluster, num_cluster_P, num_cluster_B, loss_J_, loss_P_, loss_B_))
					tr_step += 1

			sess.close()

elif FLAGS.mode == 'Eval':
	checkpt_file = pre_dir + FLAGS.dataset + '/' + FLAGS.probe + change + '/best.ckpt'

	with tf.Session(graph=loaded_graph, config=config) as sess:
		loader = tf.train.import_meta_graph(checkpt_file + '.meta')
		J_in = loaded_graph.get_tensor_by_name("Input/Placeholder:0")
		P_in = loaded_graph.get_tensor_by_name("Input/Placeholder_1:0")
		B_in = loaded_graph.get_tensor_by_name("Input/Placeholder_2:0")
		pseudo_lab_J = loaded_graph.get_tensor_by_name("Input/Placeholder_3:0")
		seq_cluster_ftr_J = loaded_graph.get_tensor_by_name("Input/Placeholder_4:0")
		pseudo_lab_P = loaded_graph.get_tensor_by_name("Input/Placeholder_5:0")
		seq_cluster_ftr_P = loaded_graph.get_tensor_by_name("Input/Placeholder_6:0")
		pseudo_lab_B = loaded_graph.get_tensor_by_name("Input/Placeholder_7:0")
		seq_cluster_ftr_B = loaded_graph.get_tensor_by_name("Input/Placeholder_8:0")
		lbl_s = loaded_graph.get_tensor_by_name("Input/Placeholder_9:0")
		seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B = loaded_graph.get_tensor_by_name("Hi_MPC/Hi_MPC/concat_24:0"), \
													 loaded_graph.get_tensor_by_name("Encoder/Reshape_6:0"), \
													 loaded_graph.get_tensor_by_name("Encoder/Reshape_7:0"), \
													 loaded_graph.get_tensor_by_name("Encoder/Reshape_8:0")

		init_op = tf.global_variables_initializer()
		sess.run(init_op)
		loader.restore(sess, checkpt_file)
		saver = tf.train.Saver()


		def gal_loader(X_train_J, X_train_P, X_train_B, y_train):
			tr_step = 0
			tr_size = X_train_J.shape[0]
			gal_logits_all = []
			gal_labels_all = []
			gal_features_all_int = []
			gal_features_all = []
			gal_features_all_P = []
			gal_features_all_B = []

			while tr_step * batch_size < tr_size:
				if (tr_step + 1) * batch_size > tr_size:
					break
				X_input_J = X_train_J[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_P = X_train_P[tr_step * batch_size:(tr_step + 1) * batch_size]
				X_input_B = X_train_B[tr_step * batch_size:(tr_step + 1) * batch_size]

				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_input_B.reshape([-1, 5, 3])

				labels = y_train[tr_step * batch_size:(tr_step + 1) * batch_size]

				[Seq_features_int, Seq_features, Seq_features_P, Seq_features_B] = sess.run(
					[seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B],
					feed_dict={
						J_in: X_input_J,
						P_in: X_input_P,
						B_in: X_input_B,

					})

				gal_features_all_int.extend(Seq_features_int.tolist())
				gal_features_all.extend(Seq_features.tolist())
				gal_features_all_P.extend(Seq_features_P.tolist())
				gal_features_all_B.extend(Seq_features_B.tolist())
				# gal_features_tea.extend(rep_tea_.tolist())
				gal_labels_all.extend(labels.tolist())
				tr_step += 1

			return gal_features_all_int, gal_features_all, gal_features_all_P, gal_features_all_B, gal_labels_all


		def evaluation():
			vl_step = 0
			vl_size = X_test_J.shape[0]
			pro_labels_all = []
			pro_features_all = []

			pro_features_tea = []
			pro_features_tea_2 = []

			vl_step = 0
			vl_size = X_test_J.shape[0]
			pro_labels_all = []
			pro_features_all_int = []
			pro_features_all = []
			pro_features_all_P = []
			pro_features_all_B = []

			while vl_step * batch_size < vl_size:
				if (vl_step + 1) * batch_size > vl_size:
					break
				X_input_J = X_test_J[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_P = X_test_P[vl_step * batch_size:(vl_step + 1) * batch_size]
				X_input_B = X_test_B[vl_step * batch_size:(vl_step + 1) * batch_size]

				X_input_J = X_input_J.reshape([-1, joint_num, 3])
				X_input_P = X_input_P.reshape([-1, 10, 3])
				X_input_B = X_input_B.reshape([-1, 5, 3])

				labels = y_test[vl_step * batch_size:(vl_step + 1) * batch_size]

				[Seq_features_int, Seq_features, Seq_features_P, Seq_features_B] = sess.run(
					[seq_ftr_int, seq_ftr, seq_ftr_P, seq_ftr_B],
					feed_dict={
						J_in: X_input_J,
						P_in: X_input_P,
						B_in: X_input_B,

					})

				pro_labels_all.extend(labels.tolist())
				pro_features_all_int.extend(Seq_features_int.tolist())
				pro_features_all.extend(Seq_features.tolist())
				pro_features_all_P.extend(Seq_features_P.tolist())
				pro_features_all_B.extend(Seq_features_B.tolist())

				vl_step += 1

			X = np.array(gal_features_all)

			X_int = np.array(gal_features_all_int)
			X_P = np.array(gal_features_all_P)
			X_B = np.array(gal_features_all_B)

			y = np.array(gal_labels_all)

			t_X = np.array(pro_features_all)
			t_X_int = np.array(pro_features_all_int)
			t_X_P = np.array(pro_features_all_P)
			t_X_B = np.array(pro_features_all_B)

			t_y = np.array(pro_labels_all)

			t_y = np.argmax(t_y, axis=-1)
			y = np.argmax(y, axis=-1)

			def mean_ap(distmat, query_ids=None, gallery_ids=None,
						query_cams=None, gallery_cams=None):
				# distmat = to_numpy(distmat)
				m, n = distmat.shape
				# Fill up default values
				if query_ids is None:
					query_ids = np.arange(m)
				if gallery_ids is None:
					gallery_ids = np.arange(n)
				if query_cams is None:
					query_cams = np.zeros(m).astype(np.int32)
				if gallery_cams is None:
					gallery_cams = np.ones(n).astype(np.int32)
				# Ensure numpy array
				query_ids = np.asarray(query_ids)
				gallery_ids = np.asarray(gallery_ids)
				query_cams = np.asarray(query_cams)
				gallery_cams = np.asarray(gallery_cams)
				# Sort and find correct matches
				indices = np.argsort(distmat, axis=1)
				matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
				# Compute AP for each query
				aps = []
				if (FLAGS.probe_type == 'nm.nm' or FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(1, m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))

						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						y_score[np.isnan(y_score)] = 0
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				else:
					for i in range(m):
						valid = ((gallery_ids[indices[i]] != query_ids[i]) |
								 (gallery_cams[indices[i]] != query_cams[i]))
						y_true = matches[i, valid]
						y_score = -distmat[i][indices[i]][valid]
						# discard nan
						# y_score = np.nan_to_num(y_score)
						if not np.any(y_true): continue
						aps.append(average_precision_score(y_true, y_score))
				if len(aps) == 0:
					raise RuntimeError("No valid query")
				return np.mean(aps)

			def metrics(X, y, t_X, t_y):
				# compute Euclidean distance
				if dataset != 'CASIA_B':
					a, b = torch.from_numpy(t_X), torch.from_numpy(X)
					m, n = a.size(0), b.size(0)
					a = a.view(m, -1)
					b = b.view(n, -1)
					dist_m = torch.pow(a, 2).sum(dim=1, keepdim=True).expand(m, n) + \
							 torch.pow(b, 2).sum(dim=1, keepdim=True).expand(n, m).t()
					dist_m.addmm_(1, -2, a, b.t())
					dist_m = (dist_m.clamp(min=1e-12)).sqrt()
					mAP = mean_ap(distmat=dist_m.numpy(), query_ids=t_y, gallery_ids=y)
					_, dist_sort = dist_m.sort(1)
					dist_sort = dist_sort.numpy()
				else:
					X = np.array(X)
					t_X = np.array(t_X)
					dist_m = [(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_m = np.array(dist_m)
					mAP = mean_ap(distmat=dist_m, query_ids=t_y, gallery_ids=y)
					dist_sort = [np.argsort(np.linalg.norm(X - i, axis=1)).tolist() for i in t_X]
					dist_sort = np.array(dist_sort)

				top_1 = top_5 = top_10 = 0
				probe_num = dist_sort.shape[0]
				if (FLAGS.probe_type == 'nm.nm' or
						FLAGS.probe_type == 'cl.cl' or FLAGS.probe_type == 'bg.bg'):
					for i in range(probe_num):

						if t_y[i] in y[dist_sort[i, 1:2]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, 1:6]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, 1:11]]:
							top_10 += 1
				else:
					for i in range(probe_num):

						if t_y[i] in y[dist_sort[i, :1]]:
							top_1 += 1
						if t_y[i] in y[dist_sort[i, :5]]:
							top_5 += 1
						if t_y[i] in y[dist_sort[i, :10]]:
							top_10 += 1
				return mAP, top_1 / probe_num, top_5 / probe_num, top_10 / probe_num

			mAP_int, top_1_int, top_5_int, top_10_int = metrics(X_int, y, t_X_int, t_y)
			mAP, top_1, top_5, top_10 = metrics(X, y, t_X, t_y)
			mAP_P, top_1_P, top_5_P, top_10_P = metrics(X_P, y, t_X_P, t_y)
			mAP_B, top_1_B, top_5_B, top_10_B = metrics(X_B, y, t_X_B, t_y)
			del X, y, t_X, t_y, X_P, t_X_P, X_B, t_X_B, pro_labels_all, pro_features_all
			gc.collect()

			return mAP_int, top_1_int, top_5_int, top_10_int, \
				   mAP, top_1, top_5, top_10, \
				   mAP_P, top_1_P, top_5_P, top_10_P, mAP_B, top_1_B, top_5_B, top_10_B


		if dataset == 'KGBD' or dataset == 'KS20':
			if FLAGS.dataset == 'KS20':
				nb_nodes = 25
			X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
			adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
				process.gen_train_data(dataset=dataset, split='gallery', time_step=time_step,
									   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
									   norm=norm
									   )
			nb_nodes = 20
		elif dataset == 'BIWI':
			if probe == 'Walking':
				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
					process.gen_train_data(dataset=dataset, split='Still', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, norm=norm
										   )
			else:
				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
					process.gen_train_data(dataset=dataset, split='Walking', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, norm=norm
										   )
		elif dataset == 'IAS':
			if probe == 'A':
				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
					process.gen_train_data(dataset=dataset, split='B', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, norm=norm
										   )
			else:
				X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
				adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes, X_train_J_D, X_gal_J_D = \
					process.gen_train_data(dataset=dataset, split='A', time_step=time_step,
										   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att,
										   batch_size=batch_size, norm=norm
										   )
		elif dataset == 'CASIA_B':

			X_train_J, X_train_P, X_train_B, _, _, y_train, X_gal_J, X_gal_P, X_gal_B, _, _, y_gal, \
			adj_J, biases_J, _, _, _, _, _, _, _, _, nb_classes = \
				process.gen_train_data(dataset=dataset, split=probe, time_step=time_step,
									   nb_nodes=nb_nodes, nhood=nhood, global_att=global_att, batch_size=batch_size,
									   PG_type=FLAGS.probe_type.split('.')[1])
		del _
		gc.collect()

		gal_features_all_int, gal_features_all, gal_features_all_P, gal_features_all_B, gal_labels_all = gal_loader(
			X_gal_J, X_gal_P, X_gal_B, y_gal)

		mAP_int, top_1_int, top_5_int, top_10_int, mAP, top_1, top_5, top_10, mAP_P, top_1_P, top_5_P, top_10_P, \
		mAP_B, top_1_B, top_5_B, top_10_B, = evaluation()

		# print(
		# 	'[Evaluation - J-level] %s - %s | Top-1: %.4f | Top-5: %.4f | Top-10: %.4f | mAP: %.4f ' % (
		# 		FLAGS.dataset, FLAGS.probe,
		# 		top_1, top_5, top_10, mAP))
		# print(
		# 	'[Evaluation - C-level] %s - %s | Top-1: %.4f | Top-5: %.4f | Top-10: %.4f | mAP: %.4f ' % (
		# 		FLAGS.dataset, FLAGS.probe,
		# 		top_1_P, top_5_P, top_10_P, mAP_P,))
		# print(
		# 	'[Evaluation - L-level] %s - %s | Top-1: %.4f | Top-5: %.4f | Top-10: %.4f | mAP: %.4f ' % (
		# 		FLAGS.dataset, FLAGS.probe,
		# 		top_1_B, top_5_B, top_10_B, mAP_B,))
		print(
			'[Evaluation - MSMR] %s - %s | Top-1: %.4f | Top-5: %.4f | Top-10: %.4f | mAP: %.4f ' % (
				FLAGS.dataset, FLAGS.probe,
				top_1_int, top_5_int, top_10_int, mAP_int))

		sess.close()
		exit()

print('End')
print('----- Model hyperparams -----')
print('batch_size: ' + str(batch_size))
print('M: ' + FLAGS.M)
print('H: ' + FLAGS.H)
print('eps: ' + FLAGS.eps)
print('min_samples: ' + FLAGS.min_samples)
print('seqence_length: ' + str(time_step))
print('patience: ' + FLAGS.patience)
print('Mode: ' + FLAGS.mode)

if FLAGS.mode == 'Train':
	print('----- Dataset Information  -----')
	print('Dataset: ' + dataset)
	print('Probe: ' + FLAGS.probe)
